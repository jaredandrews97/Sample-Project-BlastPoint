import os
import wandb
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, classification_report, RocCurveDisplay
from sklearn.feature_selection import SequentialFeatureSelector as SFS
from src.config import base_model_features, analysis_plots_fp, model_funcs, sfs_models, base_param_set, \
    target_feature, prob_thresholds, models_fp, scoring_metrics, eda_plots_fp, data_fp
from optuna.study import MaxTrialsCallback
from optuna import create_study
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner
from optuna.trial import TrialState
from functools import partial
import pickle
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
from src.utils import clean_folder

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

import shap


def train_test_split(data):
    data = data.sort_values('application_when')
    train, test = data.loc[:int(len(data) * 0.8)].reset_index(drop=True), \
        data.loc[int(len(data) * 0.8) + 1:].reset_index(drop=True)

    train_x, train_y = train[base_model_features], train['flgGood']
    test_x, test_y = test[base_model_features], test['flgGood']

    train_sw = train['SAMPLE_WEIGHT']

    # Time-based CV splits for validation
    train_tcvs = list(TimeSeriesSplit(n_splits=3).split(train))
    return train_x, train_y, test_x, test_y,  train_sw, train_tcvs


def handle_numeric_cat_feats(train_x, test_x=None, feat_names_final=None):
    preproc = ColumnTransformer(transformers=
                                [("cat", OneHotEncoder(sparse_output=False, handle_unknown='ignore'),
                                  make_column_selector(dtype_include=object)),
                                 ("num", MinMaxScaler(), make_column_selector(dtype_include=[float, int]))],
                                remainder='passthrough')

    train_encoded = preproc.fit_transform(train_x)
    if isinstance(test_x, type(None)):
        final_feats_out = preproc.get_feature_names_out()
        train_pp = train_encoded[:, [f in feat_names_final for f in final_feats_out]]
        return train_pp

    test_encoded = preproc.transform(test_x)
    encoded_out_feats = preproc.get_feature_names_out()
    return train_encoded, test_encoded, encoded_out_feats


def early_stopping_check(study, trial, early_stopping_rounds):
    current_trial_number = trial.number
    best_trial_number = study.best_trial.number
    should_stop = ((current_trial_number - best_trial_number) >= early_stopping_rounds) and trial.number >= 30
    if should_stop:
        study.stop()


def sfs_feat_selection(model, train_x, train_y, train_tcvs, test_x, encode_out_feats, scoring_metric='f1'):

    sfs = SFS(model(), scoring=scoring_metric, cv=train_tcvs, n_features_to_select='auto', tol=None)
    sfs.fit(train_x, train_y)

    train_x_final = train_x[:, sfs.support_]
    test_x_final = test_x[:, sfs.support_]
    use_feats = [c for c, s in zip(encode_out_feats, sfs.support_) if s]
    return train_x_final, test_x_final, use_feats


def hp_tuning_objective(trial, X, y, cvs, model, base_params, sample_weight, scoring_metric='f1'):

    model_name = model.__name__

    if model_name == 'LogisticRegression':

        hp_grid = {'C': trial.suggest_float("C", 1e-5, 3, log=True),
                   'tol': trial.suggest_float('tol', 1e-6, 1e-4)}

    elif model_name == 'RandomForestClassifier':
        hp_grid = {'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                   'max_depth': trial.suggest_int('max_depth', 3, 15),
                   'min_samples_split': trial.suggest_int('min_samples_split', 2, 150),
                   'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 60)}

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

    clf = model(**hp_grid, **base_params)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        score = cross_val_score(clf, X, y, cv=cvs, scoring=scoring_metric, n_jobs=-1,
                                fit_params={'sample_weight': sample_weight})

    average_cv_score_metric = score.mean()
    return average_cv_score_metric


def perform_hp_tuning(model, train_x, train_y, train_tcvs, base_params, sample_weights, s_metric):

    study = create_study(pruner=HyperbandPruner(reduction_factor=2), sampler=TPESampler(n_startup_trials=20, seed=2),
                         direction="maximize", study_name=f'{model.__name__}_hp_tuning')
    study.optimize(lambda trial: hp_tuning_objective(trial, train_x, train_y, train_tcvs, model, base_params,
                                                     sample_weights, scoring_metric=s_metric), n_trials=50,
                   callbacks=[partial(early_stopping_check, early_stopping_rounds=10),
                              MaxTrialsCallback(100, states=(TrialState.COMPLETE,))])

    best_params = study.best_trial.params
    best_params.update({**base_params})
    return best_params


def model_evaluation(model, x, y, threshold=0.5):

    model_name = type(model).__name__

    test_probs = model.predict_proba(x)[:, 1]
    test_preds = [1 if p >= threshold else 0 for p in test_probs]

    acc = accuracy_score(y, test_preds)
    f1 = f1_score(y, test_preds)
    auc = roc_auc_score(y, test_probs)
    recall = classification_report(y, test_preds, output_dict=True)['1']['recall']

    fig, ax = plt.subplots()
    RocCurveDisplay.from_estimator(model, x, y, ax=ax)
    fig_fp = os.path.join(analysis_plots_fp, f'{model_name}_ROC_AUC_curve.png')
    ax.figure.savefig(fig_fp, dpi=300)
    plt.clf()

    return acc, f1, auc, recall, test_probs, test_preds


def feature_importance_plot(model, feat_names):

    model_name = type(model).__name__

    fig, ax = plt.subplots()
    importances = pd.Series(model.feature_importances_, index=feat_names).sort_values(ascending=False).head(10)
    importances.plot.bar(ax=ax, rot=60)

    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()
    fig_fp = os.path.join(analysis_plots_fp, f'{model_name}_feat_importance_plot.png')
    fig.savefig(fig_fp, dpi=300)
    plt.clf()


def shap_plot(model, test_x, feat_names):

    model_name = type(model).__name__

    if model_name == 'RandomForestClassifier':
        explainer = shap.Explainer(model)
        shap_values = np.array(explainer.shap_values(test_x))[1]

    else:
        explainer = shap.Explainer(model.predict, test_x)
        shap_values = np.array(explainer.shap_values(test_x, silent=True))

    shap.summary_plot(shap_values, pd.DataFrame(test_x, columns=feat_names), show=False)
    fig_fp = os.path.join(analysis_plots_fp, f'{model_name}_SHAP_plot.png')
    plt.savefig(fig_fp, dpi=700)
    plt.clf()


def save_model(model, model_fp):
    """
    Helper function to save trained model files
    input:
            model: fit model to be saved
            model_fp: file path where model is saved to
    output:
            None
    """
    with open(model_fp, 'wb') as file_path:
        pickle.dump(model, file_path)


def train_evaluate_model(args, data, logger):

    print("START: Model Training")

    model_name, log_artifacts = args.model, args.wandb_log

    model, base_params, p_thresh, s_metric = model_funcs[model_name], base_param_set[model_name], \
        prob_thresholds[model_name], scoring_metrics[model_name]

    train_x, train_y, test_x, test_y,  train_sw, train_tcvs = train_test_split(data)
    train_x, test_x, encode_out_feats = handle_numeric_cat_feats(train_x, test_x)

    if model_name in sfs_models:
        train_x, test_x, final_feats = sfs_feat_selection(model, train_x, train_y, train_tcvs, test_x, encode_out_feats,
                                                          scoring_metric=s_metric)
    else:
        final_feats = encode_out_feats

    best_params = perform_hp_tuning(model, train_x, train_y, train_tcvs, base_params, train_sw, s_metric=s_metric)

    m = model(**best_params)
    m.fit(train_x, train_y, sample_weight=train_sw)

    acc, f1, auc, recall, test_probs, test_preds = model_evaluation(m, test_x, test_y, threshold=p_thresh)
    print(f"Acc: {acc}, F1: {f1}, AUC: {auc}, Recall: {recall}")

    plot_feat_names = [f.replace('cat__', '').replace('num__', '') for f in final_feats]
    if hasattr(m, 'feature_importances_'):
        feature_importance_plot(m, plot_feat_names)
    shap_plot(m, test_x, plot_feat_names)

    data_x, data_y, data_sw = data[base_model_features], data[target_feature], data['SAMPLE_WEIGHT']
    data_x = handle_numeric_cat_feats(data_x, feat_names_final=final_feats)

    m.fit(data_x, data_y, sample_weight=data_sw)
    final_model_fp = os.path.join(models_fp, f'{model_name}_model.pkl')
    save_model(m, final_model_fp)

    if args.wandb_log:
        del best_params['random_state']
        best_params.update({**{'model': model_name, 'prob_threshold': p_thresh}})
        with wandb.init(project="loan_approval_project", name=model_name, job_type='model_training',
                        config=best_params) as run:

            run.save(final_model_fp)
            run.log({"Accuracy": acc, "F1-Score": f1, "AUC Score": auc, "Recall": recall})
            model_training_analysis_outputs = wandb.Artifact("model_analysis_outputs", type="predictions_plots")
            model_training_analysis_outputs.add_dir(analysis_plots_fp, name='analysis_plots')
            pred_table = wandb.Table(dataframe=pd.DataFrame(zip(test_probs, test_preds)),
                                     columns=['test_prob_preds', 'test_preds'])
            model_training_analysis_outputs.add(pred_table, "test_pred_table")
            run.log_artifact(model_training_analysis_outputs)

            eda_outputs = wandb.Artifact("eda_outputs", type="eda_plots")
            eda_outputs.add_dir(eda_plots_fp)
            run.log_artifact(eda_outputs)

            data_artifact = wandb.Artifact("data_files", type="data_files")
            data_artifact.add_dir(data_fp)
            run.log_artifact(data_artifact)

    if not args.save_locally:
        for folder in [eda_plots_fp, data_fp, models_fp, analysis_plots_fp]:
            clean_folder(folder)

    print("END: Model Training")

