"""
Module which stores main functions used for model training

This module implements the following functionality:
    1. train/test splitting
    2. SFS Feature selection
    3. Hyperparameter tuning
    4. Model evaluation
    5. Model explainability plots
    6. WandB training artifact tracking

Author: Jared Andrews
Date: 6/11/23
"""

import os
import wandb
import warnings
import numpy as np
import pandas as pd
from functools import partial
import matplotlib.pyplot as plt
from optuna import create_study
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner
from optuna.trial import TrialState
from optuna.study import MaxTrialsCallback
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.feature_selection import SequentialFeatureSelector as SFS
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, classification_report, RocCurveDisplay
from src.utils import clean_folder, save_model
from src.config import base_model_features, analysis_plots_fp, model_funcs, sfs_models, base_param_set, \
    target_feature, prob_thresholds, models_fp, scoring_metrics, eda_plots_fp, data_fp
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
import shap


def train_test_split(data, logger):
    """
    Performs time-based train/test split for feature selection, hp-tuning and model training

    :param data: preprocessed data used in training/testing
    :param logger: Logger
    :return:
        train_x: input features dataframe for training
        train_y: target feature series for training
        test_x: input features dataframe for testing
        test_y: target feature series for testing
        train_sw: training sample weight feature
        train_tcvs: training time-based 3-fold cross validation splits
    """
    # Sort data based on application submit date to determine time-based splits
    data = data.sort_values('application_when')
    # Split time-sorted data into train/test
    train, test = data.loc[:int(len(data) * 0.8)].reset_index(drop=True), \
        data.loc[int(len(data) * 0.8) + 1:].reset_index(drop=True)

    # Break train/test datasets into x/y feature sets
    train_x, train_y = train[base_model_features], train['flgGood']
    test_x, test_y = test[base_model_features], test['flgGood']
    # Get sample weight feature for model training
    train_sw = train['SAMPLE_WEIGHT']

    # Time-based CV splits for validation
    train_tcvs = list(TimeSeriesSplit(n_splits=3).split(train))

    logger.info("Preprocessed dataframe split into train/test sets successfully")
    return train_x, train_y, test_x, test_y,  train_sw, train_tcvs


def handle_numeric_cat_feats(train_x, logger, test_x=None, feat_names_final=None):
    """

    :param train_x: input features dataframe for training to be processed via ColumnTransformer
    :param logger: Logger
    :param test_x: input features dataframe for testing to be processed via ColumnTransformer
    :param feat_names_final: the final feature names used when we are applying the feat transformation to the whole
                             dataset for final model training (feature selection has occured)
    :return:
        train_encoded: training dataframe that has been processed through the ColumnTransformer (OHE/MinMaxScaling)
        test_encoded: testing dataframe that has been processed through the ColumnTransformer (OHE/MinMaxScaling)
        encoded_out_feats: feature names produced by the One-hot-encoder
    """
    # Column transformer which one-hot-encodes categorical features and scales numeric features
    preproc = ColumnTransformer(transformers=
                                [("cat", OneHotEncoder(sparse_output=False, handle_unknown='ignore'),
                                  make_column_selector(dtype_include=object)),
                                 ("num", MinMaxScaler(), make_column_selector(dtype_include=[float, int]))],
                                remainder='passthrough')

    # Apply transformations to training dataset
    train_encoded = preproc.fit_transform(train_x)
    # In this case, the train_x is the whole dataset, and we are training the final model (i.e., there is no test set)
    # Since feature selection has occurred during earlier training, we will return the encoded dataframe with the
    # preselected columns
    if isinstance(test_x, type(None)):
        final_feats_out = preproc.get_feature_names_out()
        train_encoded = train_encoded[:, [f in feat_names_final for f in final_feats_out]]
        logger.info("Successfully encoded all data for final model training")
        return train_encoded

    # Apply transformations to test dataset
    test_encoded = preproc.transform(test_x)
    # Get the features produced by the transformations (specifically OHE)
    encoded_out_feats = preproc.get_feature_names_out()
    logger.info("Successfully encoded training data for feature selection (if applicable), HP-tuning and model "
                "evaluation")
    return train_encoded, test_encoded, encoded_out_feats


def sfs_feat_selection(model, train_x, train_y, train_tcvs, test_x, encode_out_feats, logger, scoring_metric='f1'):
    """
    Applied Sequential Forward Feature selection using provided model and returns dataframes with features and the
    feature names

    :param model: Model which is being used in training
    :param train_x: input features dataframe for training
    :param train_y: target feature series for training
    :param train_tcvs: time-based 3-fold cross validation splits used for SFS k-fold cross validation
    :param test_x: input features dataframe for testing
    :param encode_out_feats: features produced by the ColumnTransformer in previous step
    :param logger: Logger
    :param scoring_metric: The scoring metric used to determine feature set quality during SFS
    :return:
        train_x_final: training dataset with features found by SFS
        test_x_final: testing dataset with features found by SFS
        use_feats: features found by SFS
    """

    # Instantiation of SFS instance with provided model, scoring metric and cv splits
    sfs = SFS(model(), scoring=scoring_metric, cv=train_tcvs, n_features_to_select='auto', tol=None)
    # Train SFS using training datasets
    sfs.fit(train_x, train_y)

    # Based on features found by SFS, limit training and testing datasets to features and get feature names
    train_x_final = train_x[:, sfs.support_]
    test_x_final = test_x[:, sfs.support_]
    use_feats = [c for c, s in zip(encode_out_feats, sfs.support_) if s]

    logger.info("Successfully performed Sequential Forward Selection feature selection")
    return train_x_final, test_x_final, use_feats


def hp_tuning_objective(trial, X, y, cvs, model, base_params, sample_weights, scoring_metric='f1'):

    """
    Optuna hyperparameter tuning objective used for each trial within model training

    :param trial: Optuna trial object which comprise a HP-tuning study
    :param X: training input feature dataset
    :param y: training target feature series
    :param cvs: 3-fold cross validations training splits
    :param model: model to be tuned
    :param base_params: Set of parameters (e.g., RandomState) which are required but not tuned
    :param sample_weights: Sample weights which indicate how an individual sample should be weighted during training
    :param scoring_metric: Scoring metric used to determine quality of the current HP configuration
    :return:
        average_cv_score_metric: Average scoring metric across all splits
    """

    # Get model name to determine which hyperparameters to tune
    model_name = model.__name__

    # Setup hp_grid for LogisticRegression
    if model_name == 'LogisticRegression':

        hp_grid = {'C': trial.suggest_float("C", 1e-5, 3, log=True),
                   'tol': trial.suggest_float('tol', 1e-6, 1e-4)}

    # Setup hp_grid for RandomForestClassifier
    elif model_name == 'RandomForestClassifier':
        hp_grid = {'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                   'max_depth': trial.suggest_int('max_depth', 3, 15),
                   'min_samples_split': trial.suggest_int('min_samples_split', 2, 150),
                   'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 60)}

    # Setup hp_grid for XGBoostClassifier
    else:
        hp_grid = {"booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
                   "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
                   "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
                   "n_estimators": trial.suggest_int("n_estimators", 40, 500, step=20)}

        if hp_grid["booster"] in ["gbtree", "dart"]:
            hp_grid["max_depth"] = trial.suggest_int("max_depth", 1, 7)
            hp_grid["eta"] = trial.suggest_float("eta", 1e-5, 0.1, log=True)
            hp_grid["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
            hp_grid["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])
        if hp_grid["booster"] == "dart":
            hp_grid["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
            hp_grid["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
            hp_grid["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 1.0, log=True)
            hp_grid["skip_drop"] = trial.suggest_float("skip_drop", 1e-8, 1.0, log=True)

    # Instantiate model with current HP configuration
    clf = model(**hp_grid, **base_params)

    # Perform k-fold CV using instantiated model to determine quality of hp config
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        score = cross_val_score(clf, X, y, cv=cvs, scoring=scoring_metric, n_jobs=-1,
                                fit_params={'sample_weight': sample_weights})

    # Take the average of the scores across the splits to get final hp training model score
    average_cv_score_metric = score.mean()
    return average_cv_score_metric


def early_stopping_check(study, trial, early_stopping_rounds, min_num_trials=30):
    """
    Applies early stopping when triggered

    :param study: Optuna study object which facilitates Optuna HP-tuning
    :param trial: Optuna trial object which comprise HP-tuning studies
    :param early_stopping_rounds: Number of rounds where the validation metric doesn't improve before hp-tuning stops
    :param min_num_trials: The minimum number of rounds to complete before early_stopping can be triggered
    :return: None
    """
    # Set the current number of trials that have been completed
    current_trial_number = trial.number
    # Set the best trial number
    best_trial_number = study.best_trial.number
    # Condition which looks at if there has been a specified number of trials since the best trial and that the min
    # number of trials has been completed
    should_stop = ((current_trial_number - best_trial_number) >= early_stopping_rounds) and \
                  trial.number >= min_num_trials
    # Stop hp-tuning based on 'should_stop' condition
    if should_stop:
        study.stop()


def perform_hp_tuning(model, train_x, train_y, train_tcvs, base_params, sample_weights, s_metric, logger):
    """
    Function used to perform Optuna HP-tuning

    :param model: model to be tuned
    :param train_x: input features dataframe for training
    :param train_y: target feature series for training
    :param train_tcvs: time-based 3-fold cross validation splits used for HP tuning k-fold cross validation
    :param base_params: Set of parameters (e.g., RandomState) which are required but not tuned
    :param sample_weights: Sample weights which indicate how an individual sample should be weighted during training
    :param s_metric: Scoring metric used to determine quality of the current HP configuration
    :param logger: Logger
    :return:
        best_params: dictionary containing the best hyperparameters found during HP tuning
    """

    # Instantiate the study with Hyperband pruner, sampler using Tree-structured Parzen Estimator algorithm and
    # direction which indicates if the scoring metric should be optimized or minimized
    study = create_study(pruner=HyperbandPruner(reduction_factor=2), sampler=TPESampler(n_startup_trials=20, seed=2),
                         direction="maximize", study_name=f'{model.__name__}_hp_tuning')
    # Initiate study using hp_tuning_objective, early stopping function and MaxTrialsCallback which indicated the
    # max number of trials to use
    study.optimize(lambda trial: hp_tuning_objective(trial, train_x, train_y, train_tcvs, model, base_params,
                                                     sample_weights, scoring_metric=s_metric), n_trials=50,
                   callbacks=[partial(early_stopping_check, early_stopping_rounds=10, min_num_trials=30),
                              MaxTrialsCallback(100, states=(TrialState.COMPLETE,))])

    # Get the best parameters from HP tuning
    best_params = study.best_trial.params
    # Update the best parameters with necessary base params
    best_params.update({**base_params})
    logger.info("Successfully performed hyperparameter tuning")
    return best_params


def model_evaluation(model, x, y, logger, threshold=0.5):
    """
    Function which calculates necessary testing metrics and produced analysis plots

    :param model: trained model to be evaluated
    :param x: testing input feature dataset
    :param y: testing target feature series
    :param logger: Logger
    :param threshold: Threshold used to convert the predicted probabilities from model to binary output indicating
                      predicted loan quality
    :return:
        acc: Testing accuracy
        f1: Testing F1-Score
        auc: Testing AUC-Score
        recall: Testing recall score
        test_probs: Raw predicted probabilities from model for testing dataset samples
        test_preds: Binary predictions produced using threshold and raw predicted probabilities
    """

    # Generate the predicted probabilities for the test set
    test_probs = model.predict_proba(x)[:, 1]
    # Convert the predicted probabilities to a binary output using specified threshold
    test_preds = [1 if p >= threshold else 0 for p in test_probs]

    # Calculate testing accuracy, f1, auc and recall scores
    acc = accuracy_score(y, test_preds)
    f1 = f1_score(y, test_preds)
    auc = roc_auc_score(y, test_probs)
    recall = classification_report(y, test_preds, output_dict=True)['1']['recall']

    # Plot and save the AUC-ROC curve using testing data
    fig, ax = plt.subplots()
    RocCurveDisplay.from_estimator(model, x, y, ax=ax)
    fig_fp = os.path.join(analysis_plots_fp, f'{type(model).__name__}_ROC_AUC_curve.png')
    ax.figure.savefig(fig_fp, dpi=300)
    plt.clf()

    logger.info("Successfully generated test metrics and produced analysis plots")
    return acc, f1, auc, recall, test_probs, test_preds


def feature_importance_plot(model, feat_names, logger):
    """
    Function to generate the feature importance plots for the tree-based models

    :param model: trained model to generate feature importance plot from
    :param feat_names: features used by model
    :param logger: Logger
    :return:
        None
    """

    fig, ax = plt.subplots()
    # Produce series which contains the feature importances for the top 10 most important features
    importances = pd.Series(model.feature_importances_, index=feat_names).sort_values(ascending=False).head(10)
    # Produce a barplot of the most important features and set title and y label
    importances.plot.barh(ax=ax)
    ax.set_title("Feature importances using MDI")
    ax.set_xlabel("Mean decrease in impurity")
    ax.invert_yaxis()
    fig.tight_layout()
    fig_fp = os.path.join(analysis_plots_fp, f'{type(model).__name__}_feat_importance_plot.png')
    # Save barplot to file
    fig.savefig(fig_fp, dpi=500)
    plt.clf()

    logger.info("Successfully generated feature importance plot for model")


def shap_plot(model, test_x, feat_names, logger):
    """
    Function to produce the SHAP plot for each feature based on the testing dataset

    :param model: trained model to generate SHAP plot from
    :param test_x: testing input feature dataset
    :param feat_names: features used by model
    :param logger: Logger
    :return:
        None
    """

    # Get model name to determine how to create SHAP plot
    model_name = type(model).__name__

    # Generate shap plot from testing data
    try:

        if model_name == 'RandomForestClassifier':
            explainer = shap.Explainer(model)
            shap_values = np.array(explainer.shap_values(test_x))[1]

        else:
            explainer = shap.Explainer(model.predict, test_x)
            shap_values = np.array(explainer.shap_values(test_x, silent=True))

    # Due to issues with some versions of SHAP plot library, if error is thrown don't generate plot and continue.
    except AttributeError as err:
        logger.error(err)
        message = "Downgrade to previous version of numpy to enable shap to work properly or manually edit the shap " \
                  "file to remove np.bool-> bool, np.int -> int. No shap value plots produced."
        logger.warn(message)
        print(message)
        return

    # Plot shap values and save to file
    shap.summary_plot(shap_values, pd.DataFrame(test_x, columns=feat_names), show=False)
    fig_fp = os.path.join(analysis_plots_fp, f'{model_name}_SHAP_plot.png')
    plt.savefig(fig_fp, dpi=700)
    plt.clf()

    logger.info("Successfully generated SHAP plot for model")


def train_evaluate_model(args, data, logger):
    """
    Main function which runs all steps of model training, model evaluation and model experiment tracking

    :param args: Args used to get model name and if artifacts should be logged using weights and bias
    :param data: complete dataset used during model training and evaluation
    :param logger: Logger
    :return:
        None
    """

    print("START: Model Training")

    # Setup model_name and if model results should be logged bool vars
    model_name, log_artifacts = args.model, args.wandb_log

    # Setup model obj, base parameters, probability threshold and evaluation metric for specific model used
    model, base_params, p_thresh, s_metric = model_funcs[model_name], base_param_set[model_name], \
        prob_thresholds[model_name], scoring_metrics[model_name]

    # Create train/test datasets and k-fold CV splits
    train_x, train_y, test_x, test_y,  train_sw, train_tcvs = train_test_split(data, logger)

    # One hot-encode cat variables and min/max scale numeric features for training data
    train_x, test_x, encode_out_feats = handle_numeric_cat_feats(train_x, logger, test_x=test_x)

    # Perform Sequential Forward Selection feature selection for non-tree based models (i.e., LogisticRegression)
    if model_name in sfs_models:
        train_x, test_x, final_feats = sfs_feat_selection(model, train_x, train_y, train_tcvs, test_x, encode_out_feats,
                                                          logger, scoring_metric=s_metric)
    else:
        final_feats = encode_out_feats

    # Run hyper-parameter tuning
    best_params = perform_hp_tuning(model, train_x, train_y, train_tcvs, base_params, train_sw, s_metric, logger)

    # Using the best hyperparameters, train model only on train data to get test eval metrics
    m = model(**best_params)
    m.fit(train_x, train_y, sample_weight=train_sw)

    # Get the testing evaluation metrics from model trained only on training data
    acc, f1, auc, recall, test_probs, test_preds = model_evaluation(m, test_x, test_y, logger, threshold=p_thresh)
    print(f"Acc: {acc}, F1: {f1}, AUC: {auc}, Recall: {recall}")

    # Plot the feature names and SHAP plots for model explainability
    plot_feat_names = [f.replace('cat__', '').replace('num__', '') for f in final_feats]
    if hasattr(m, 'feature_importances_'):
        feature_importance_plot(m, plot_feat_names, logger)
    shap_plot(m, test_x, plot_feat_names, logger)

    # Setup data required for final model training on all the data
    data_x, data_y, data_sw = data[base_model_features], data[target_feature], data['SAMPLE_WEIGHT']
    # One hot-encode cat variables and min/max scale numeric features for all the data
    data_x = handle_numeric_cat_feats(data_x, logger, feat_names_final=final_feats)

    # Perform final model training using all the data
    m.fit(data_x, data_y, sample_weight=data_sw)
    final_model_fp = os.path.join(models_fp, f'{model_name}_model.pkl')
    # Save final model; used in production
    save_model(m, final_model_fp)
    logger.info("Final model successfully trained and saved to file locally")

    # If wandb logging is requested, log model and data artificats
    if args.wandb_log:
        logger.info("Using WandB to store and log model training artifacts and results")

        # Remove random state from model config and add model name, probability threshold
        del best_params['random_state']
        best_params.update({**{'model': model_name, 'prob_threshold': p_thresh}})
        # Initialize wandb run for loan_approval_project project, log model hyperparamets as config
        with wandb.init(project="loan_approval_project", name=model_name, job_type='model_training',
                        config=best_params) as run:

            # Save the final trained model to wandb project run
            run.save(final_model_fp)
            logger.info("WandB: final model output logged")

            # Log the evaluation metrics
            run.log({"Accuracy": acc, "F1-Score": f1, "AUC Score": auc, "Recall": recall})
            # Log the model analysis plots
            model_training_analysis_outputs = wandb.Artifact("model_analysis_outputs", type="predictions_plots")
            model_training_analysis_outputs.add_dir(analysis_plots_fp, name='analysis_plots')
            # Log the model's predictions on the test set (both raw probs and binary outputs)
            pred_table = wandb.Table(dataframe=pd.DataFrame(zip(test_probs, test_preds)),
                                     columns=['test_prob_preds', 'test_preds'])
            model_training_analysis_outputs.add(pred_table, "test_pred_table")
            run.log_artifact(model_training_analysis_outputs)
            logger.info("WandB: model analysis outputs and model predictions logged")

            # Log the EDA plot outputs
            eda_outputs = wandb.Artifact("eda_outputs", type="eda_plots")
            eda_outputs.add_dir(eda_plots_fp)
            run.log_artifact(eda_outputs)
            logger.info("WandB: EDA output plots logged")

            # Log the raw, intermediate and final datasets
            data_artifact = wandb.Artifact("data_files", type="data_files")
            data_artifact.add_dir(data_fp)
            run.log_artifact(data_artifact)
            logger.info("WandB: Raw, intermediate and final datasets logged")

    # If user doesn't want to have outputs stored locally, remove all local outputs
    if not args.save_locally:
        for folder in [eda_plots_fp, data_fp, models_fp, analysis_plots_fp]:
            clean_folder(folder)
        logger.info("Removing unwanted training model outputs from being stored locally")

    print("END: Model Training")
