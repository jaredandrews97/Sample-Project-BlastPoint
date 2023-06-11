# Loan Approval Project

- **Loan Approval Project** for sample project presentation

## Project Description
The purpose of this project is to utilize historical loan approval information to predict if a new customer will pay 
their loan requested loan back. The data used to train the model was provided by Kaggle and contains 33 features for 650 
unique customers. Three models, a logistic regression model, a random forest classifier and a XGBoost classifier, were 
developed and analyzed, with the random forest classifier performing the best (based on predefined evaluation metrics). 
The data-preprocessing, feature engineering, model training and model analysis steps utilizes the high-level steps 
listed below:
1. Reading in Data
2. Initial Preprocessing
3. EDA
4. Feature Engineering
5. Model Training
6. Model Evaluation
7. Model Tracking


## Files and data description
```
.
├── data
│   ├── Homework_Data_Scientist.xlsx                # Input data
│   ├── intermediate_data.csv                       # Data after initial preprocessing
│   └── preprocessed_data.csv                       # Data after feature engineering
├── logs
│   └── results.log                                 # Log produced during model training
├── models
│   ├── LogisticRegression_model.pkl                # Pickled file containing trained logistic regression model
│   ├── RandomForest_model.pkl                      # Pickled file containing trained random forest model
│   └── XGBoost_model.pkl                           # Pickled file containing trained XGBoost model
├── notebooks
│   └── Project_code.ipynb                          # Jupyter-notebook used during development of training pipeline
├── reports
│   └── ...                                         # Reports developed during presentations of project
├── results
│   ├── analysis_plots                              # Folder containing ROC_AUC, SHAP and feature importance plots 
│   └── eda_plots                                   # Folder containing histograms and scatterplots during EDA
├── src                                               
│   ├── config.py                                   # File containing hard-coded variables and filepaths
│   ├── data.py                                     # File containing functions involved in data parsing and intial preprocessing
│   ├── eda.py                                      # File containing functions involved in EDA plot generation
│   ├── feature_engineering.py                      # File containing functions involved in feature engineering
│   ├── train.py                                    # File containing functions involved in the training of the model
│   └── utils.py                                    # File containing helper functions
├── main.py                                         # Main file which the complete model training process is initiated from
├── README.md                                       # README file
└── conda.yml                                       # YML file utilized to produce conda env for training
```
## Running Files
1. Clone repo
2. Set up environment (using conda):

        conda env create --file conda.yml --name loan_project_env
3. Perform data preprocessing and/or model training:

        python3 main.py --data_fp --wandb_log --preprocess_data --train_model --save_locally

### Parameters
- **data_fp**: File path for the raw training data
  - Type: string
  - Required: No
  - Default: './data/Homework_Data_Scientist.xlsx'
- **wandb_log**: Boolean indicating if results should be logged via WandB run
  - Type: boolean
  - Required: No
  - Default: False
- **preprocess_data**: Boolean indicating if data preprocessing should be performed (or if locally stored preprocessed file should be used)
  - Type: boolean
  - Required: Yes
- **train_model**: Boolean indicating if model training should be performed
  - Type: boolean
  - Required: Yes
- **model**: ML model algorithm used
  - Type: string
  - Required: Yes
  - choices= ['LogisticRegression', 'RandomForest', 'XGBoost']
- **save_locally**: Boolean indicating if outputs should be saved locally
  - Type: boolean
  - Required: No
  - Default: True
   



