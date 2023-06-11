import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, classification_report, RocCurveDisplay
from sklearn.feature_selection import SequentialFeatureSelector as SFS
from src.config import base_model_features, analysis_plots_fp, model_funcs, sfs_models, base_param_set, target_feature, \
    prob_thresholds, models_fp
from optuna.study import MaxTrialsCallback
from optuna import create_study
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner
from optuna.trial import TrialState
from functools import partial
import pickle
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning

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

    train_pp = preproc.fit_transform(train_x)
    if isinstance(test_x, type(None)):
        final_feats_out = preproc.get_feature_names_out()
        train_pp = train_pp[:, [f in feat_names_final for f in final_feats_out]]
        return train_pp

    train_pp = preproc.fit_transform(train_x)
    test_pp = preproc.transform(test_x)
    preproc_out_feats = preproc.get_feature_names_out()
    return train_pp, test_pp, preproc_out_feats


def early_stopping_check(study, trial, early_stopping_rounds):
    current_trial_number = trial.number
    best_trial_number = study.best_trial.number
    should_stop = ((current_trial_number - best_trial_number) >= early_stopping_rounds) and trial.number >= 30
    if should_stop:
        study.stop()


def sfs_feat_selection(model, train_x, train_y, train_tcvs, test_x, encode_out_feats, scoring_metric='f1'):

    sfs = SFS(model(), scoring=scoring_metric, cv=train_tcvs)
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
                   'max_depth': trial.suggest_int('max_depth', 4, 12),
                   'min_samples_split': trial.suggest_int('min_samples_split', 2, 150),
                   'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 60)}

    else:
        hp_grid = {"booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
                   "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
                   "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
                   "n_estimators": trial.suggest_int("n_estimators", 50, 500, step=50)}

        if hp_grid["booster"] in ["gbtree", "dart"]:
            hp_grid["max_depth"] = trial.suggest_int("max_depth", 1, 7)
            hp_grid["eta"] = trial.suggest_float("eta", 1e-8, 1.0, log=True)
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


def perform_hp_tuning(model, train_x, train_y, train_tcvs, base_params, sample_weights):

    study = create_study(pruner=HyperbandPruner(reduction_factor=2), sampler=TPESampler(n_startup_trials=20, seed=2),
                         direction="maximize", study_name='xgboost_hp_tuning')
    study.optimize(lambda trial: hp_tuning_objective(trial, train_x, train_y, train_tcvs, model, base_params,
                                                     sample_weights), n_trials=50,
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
    auc = roc_auc_score(y, test_preds)
    recall = classification_report(y, test_preds, output_dict=True)['1']['recall']

    fig, ax = plt.subplots()
    RocCurveDisplay.from_estimator(model, x, y, ax=ax)
    fig_fp = os.path.join(analysis_plots_fp, f'{model_name}_ROC_AUC_curve.png')
    ax.figure.savefig(fig_fp, dpi=300)
    plt.clf()

    return acc, f1, auc, recall, test_preds


def feature_importance_plot(model, feat_names):

    model_name = type(model).__name__

    fig, ax = plt.subplots()
    importances = model.feature_importances_
    if model_name != 'XGBClassifier':
        importance_std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
        imp_df = pd.DataFrame(zip(importances, importance_std),
                              index=feat_names).sort_values(0, ascending=False).head(10)
        importances, importance_std = imp_df[0], imp_df[1]
        importances.plot.bar(yerr=importance_std, ax=ax)
    else:
        importances = pd.Series(importances, index=feat_names).sort_values(ascending=False).head(10)
        importances.plot.bar(ax=ax)

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

    model_name, log_artifacts = args.model, args.wandb_log

    model = model_funcs[model_name]
    base_params = base_param_set[model_name]
    p_thresh = prob_thresholds[model_name]

    train_x, train_y, test_x, test_y,  train_sw, train_tcvs = train_test_split(data)
    train_x, test_x, encode_out_feats = handle_numeric_cat_feats(train_x, test_x)

    if model_name in sfs_models:
        train_x, test_x, final_feats = sfs_feat_selection(model, train_x, train_y, train_tcvs, test_x, encode_out_feats)
    else:
        final_feats = encode_out_feats

    best_params = perform_hp_tuning(model, train_x, train_y, train_tcvs, base_params, train_sw)

    m = model(**best_params)
    m.fit(train_x, train_y, sample_weight=train_sw)

    acc, f1, auc, recall, test_preds = model_evaluation(m, train_x, train_y, threshold=p_thresh)

    if hasattr(m, 'feature_importances_'):
        feature_importance_plot(m, final_feats)
    shap_plot(m, test_x, final_feats)

    data_x, data_y, data_sw = data[base_model_features], data[target_feature], data['SAMPLE_WEIGHT']
    data_x = handle_numeric_cat_feats(data_x, feat_names_final=final_feats)

    m.fit(data_x, data_y, sample_weight=data_sw)
    final_model_fp = os.path.join(models_fp, f'{model_name}_model.pkl')
    save_model(m, final_model_fp)
