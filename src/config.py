import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


input_cols = ['customer_id', 'amount_requested', 'birth_date', 'status', 'residence_rent_or_own',
              'monthly_rent_amount', 'bank_account_direct_deposit', 'application_when', 'loan_duration', 'payment_ach',
              'num_payments', 'payment_amount', 'amount_approved', 'duration_approved', 'payment_amount_approved',
              'address_zip', 'email', 'bank_routing_number', 'email_duration', 'residence_duration',
              'bank_account_duration', 'payment_frequency', 'home_phone_type', 'how_use_money', 'monthly_income_amount',
              'raw_l2c_score', 'raw_FICO_telecom', 'raw_FICO_retail', 'raw_FICO_bank_card', 'raw_FICO_money', 'idLoan',
              'flgGood']

target_feature = 'flgGood'

date_cols = ['birth_date', 'application_when']

eda_drop_feats = ['customer_id',  # unique identifier
                  'status',  # all approved loans (constant feat)
                  'email',  # no info provided by user email
                  'home_phone_type',  # no info provided by user email
                  'how_use_money',  # too many non-descript categories, high instance of 'Other', 'Bills', etc.
                  'idLoan',  # unique identifier,
                  'birth_date']  # potential legal ramifications for using age to determine if someone gets a loan

base_model_features = ['amount_requested', 'monthly_income_after_rent', 'residence_rent_or_own',
                       'bank_account_direct_deposit', 'loan_duration', 'num_payments', 'payment_amount',
                       'amount_approved', 'duration_approved', 'address_zip', 'bank_routing_number', 'email_duration',
                       'residence_duration', 'bank_account_duration', 'payment_frequency', 'raw_l2c_score',
                       'raw_FICO_telecom', 'raw_FICO_retail', 'raw_FICO_bank_card', 'raw_FICO_money', 'pays_rent']

data_fp = './data'

raw_data_fp = os.path.join(data_fp, 'Homework_Data_Scientist.xlsx')

intermediate_data_fp = os.path.join(data_fp, 'intermediate_data.csv')

preprocessed_data_fp = os.path.join(data_fp, 'preprocessed_data.csv')

logs_fp = './logs/results.log'

models_fp = './models'

results_fp = './results'

eda_plots_fp = os.path.join(results_fp, 'eda_plots/')

analysis_plots_fp = os.path.join(results_fp, 'analysis_plots/')

model_funcs = {'LogisticRegression': LogisticRegression, 'RandomForest': RandomForestClassifier,
               'XGBoost': XGBClassifier}

sfs_models = ['LogisticRegression']

base_param_set = {'LogisticRegression': {'penalty': 'l2', 'max_iter': 200, "verbose": False, "random_state": 1},
                  'RandomForest': {"random_state": 1},
                  'XGBoost': {"objective": "binary:logistic", "verbosity": 0, "random_state": 1}}

prob_thresholds = {'LogisticRegression': 0.4, 'RandomForest': 0.4, 'XGBoost': 0.36}
