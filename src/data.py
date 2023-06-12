"""
Module which stores data parsing/preprocessing functions

This module implements the following functionality:
    1. parse_data
    2. clean_nans
    3. handle_booleans
    4. datatype_casting
    5. credit_score_filtering
    6. parse_preprocess_data

Author: Jared Andrews
Date: 6/9/23
"""

import logging
import warnings
import os
import pandas as pd
from src.config import date_cols, input_cols, intermediate_data_fp, target_feature, data_fp
from src.utils import clean_folder

warnings.filterwarnings("ignore", category=UserWarning)


def parse_data(fp, logger):
    """
    Parsing in of data used in model training

    :param fp: File path to read raw data from
    :param logger: Logger
    :return: parsed raw data
    """

    assert os.path.exists(fp), "Specified file path does not exist"

    clean_folder(data_fp)

    try:
        # Read in application data (descriptors of a loan)
        app_data = pd.read_excel(fp, engine='openpyxl', sheet_name='Application Data',
                                 parse_dates=date_cols)

        # Read in loan performance data (binary indicator of loan quality)
        loan_performance = pd.read_excel(fp, engine='openpyxl', sheet_name='Loan Performance')

    except ValueError as err:
        logging.critical(err, exc_info=True)
        raise err

    loan_performance['customer_id'] = loan_performance['idLoan'].str.split('-').str[0].str.lower()
    # Merge app data and loan performance data; note merge occurs on customer_id and not on loanid; 3 customers had
    # good and bad loan performances and so impossible to determine if app data is for good bad loan
    data = app_data.merge(loan_performance, on=['customer_id'])
    # Drop unnecessary customer_id and idLoan features (unique identifiers)
    data.drop(['customer_id', 'idLoan'], axis=1, inplace=True)
    # Determine if missing columns exist in input data
    missing_cols = list(set(input_cols) - set(data.columns))
    assert len(missing_cols) == 0, f"Input data is missing following features: {str(missing_cols)}"

    logger.info(f"Raw data successfully read with dimension: {data.shape}")
    return data


def clean_nans(data, logger):
    """
    Handling of nans in input data

    :param data: Input data (raw)
    :param logger: Logger
    :return: dataframe with nans removed
    """

    # Determine which columns contains nans so that these columns can be dealt with on an individual basis
    na_vals = data.isna()
    na_cols = data.columns[data.isna().any()]
    # Utilize dataframe to create logging info on columns and their nan counts (if nans exist)
    nan_df = pd.DataFrame(zip(na_vals[na_cols].sum(), na_vals[na_cols].mean()),
                          columns=['nan_count', 'proportion_nan'], index=na_cols).sort_values('nan_count')

    logger.info("NaNs present in raw data:")
    for (c, count, prop) in list(zip(nan_df.index, nan_df['nan_count'], nan_df['proportion_nan'])):
        logger.info(f"\tCol: {c}, Count: {count}, Proportion: {prop}")

    # bank_account_duration: nan because payment_ach = 0 (additional context required on payment_ach and
    # bank_account_duration fields), since only 1 row will drop
    data = data[data['bank_account_duration'].notnull()]

    # how_use_money: dropping nan for bank_account_duration dropped 1 out of 2 how_use_money nan rows; since only 1
    # row will drop
    data = data[data['how_use_money'].notnull()]

    # payment_amount_approved: this is a feature that occurs after approval and will not exist in data in production,
    # drop feature as a result
    # other_phone_type: not an important feature, will drop feature
    data.drop(['payment_amount_approved', 'other_phone_type'], axis=1, inplace=True)

    logger.info("Successfully handled NaN values")
    return data


def handle_booleans(data, logger):
    """
    Handling of booleans in input data

    :param data: Input data (nans handled)
    :param logger: Logger
    :return: dataframe with booleans handled
    """

    # Convert bool to 0/1 feature for model
    bool_cols = data.select_dtypes('bool').columns
    logger.info(f"Bool Features:\n\t{', '.join(list(bool_cols) + [target_feature])}")
    data[bool_cols] = data[bool_cols].astype(int)
    # Convert target variable (flgGood) to 0/1 feature for model
    data[target_feature] = data[target_feature].map({'Good': 1, 'Bad': 0})
    return data


def datatype_casting(data, logger):
    """
    Handling of proper data types in input data

    :param data: Input data (nans & booleans handled)
    :param logger: Logger
    :return: dataframe with proper datatypes handled
    """

    # Zip code, bank routing number are numeric representations of categorical features
    logger.info(f"Converting {', '.join(['address_zip', 'bank_routing_number'])} to strings")
    data['address_zip'] = data['address_zip'].astype(str)
    data['bank_routing_number'] = data['bank_routing_number'].astype(str)
    return data


def credit_score_filtering(data, logger):
    """
    Handling of proper data types in input data

    :param data: Input data (nans & booleans handled, datatypes cast)
    :param logger: Logger
    :return: dataframe with incorrect FICO/L2C scores removed
    """
    # Scores should be in range [300, 850]; filter out any rows where FICO/L2C scores are outside of this range
    data_len = len(data)
    data = data[(data[[c for c in data.columns if 'FICO' in c]] > 300).all(axis=1) &
                (data[[c for c in data.columns if 'FICO' in c]] < 850).all(axis=1) &
                (data['raw_l2c_score'] > 300) & (data['raw_l2c_score'] < 850)].reset_index(drop=True)
    num_rows_filtered = data_len - len(data)
    logger.info(f"{num_rows_filtered} rows removed by FICO/L2C filtering")
    return data


def parse_preprocess_data(args, logger):
    """
    Main data parsing/preprocessing function used before EDA is performed

    :param args:
        data_fp: contains File path to read raw data from
    :param logger: Logger
    :return: Preprocessed dataframe
    """
    print("START: Parsing and initial data preprocessing")
    data = parse_data(args.data_fp, logger)
    data = clean_nans(data, logger)
    data = handle_booleans(data, logger)
    data = datatype_casting(data, logger)
    data = credit_score_filtering(data, logger)

    logger.info(f"Intermediate data successfully processed with dimension: {data.shape}")

    logger.info(f"Saving intermediate data to file")
    data.to_csv(intermediate_data_fp, index=False)
    print("DONE: Parsing and initial data preprocessing")

    return data
