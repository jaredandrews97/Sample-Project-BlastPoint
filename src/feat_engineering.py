"""
Module which stores main functions used during the feature engineering step of model training

Author: Jared Andrews
Date: 6/11/23
"""

import numpy as np
from src.config import preprocessed_data_fp


def shorted_zipcode(df, logger):
    """
    Function which shortens zipcodes by retaining only the first two numbers in the zip. Previous zips were too low
    volume and so only keeping the first two digits encoded locality while also producing a low cardinality feature.

    :param df: data to be preprocessed
    :param logger: Logger
    :return:
        df: data with adjustment to zipcode feature
    """

    # Get original number of zip codes for logging
    orig_n_zips = df['address_zip'].nunique()
    # Shorten zipcodes to 2 digits
    df['address_zip'] = df['address_zip'].str[:2]
    # Get new number of zip codes feature values for logging
    new_n_zips = df['address_zip'].nunique()
    logger.info(f"Original number of zipcodes: {orig_n_zips}, New number of shortened zipcodes: {new_n_zips}")
    return df


def group_bank_routing_nums(df, logger):
    """
    Function which only keeps high instance values for the bank routing number. Many brns had only one or two instances
    and so consolidating low instance brns helps the model.

    :param df: data to be preprocessed
    :param logger: Logger
    :return:
        df: data with consolidate bank routing number feature
    """

    # Get original number of bank routing numbers for logging
    orig_n_rn = df['bank_routing_number'].nunique()
    # Get value counts for each bank routing number in the dataset
    bank_rn_vc = df['bank_routing_number'].value_counts()
    # Only keep bank routing number values where there are at least 15 instances; encode all others as 'default'
    bank_rn_retain = bank_rn_vc[bank_rn_vc >= 15].index
    df['bank_routing_number'] = df['bank_routing_number'].apply(lambda rn: rn if rn in bank_rn_retain else 'default')
    # Get new number of bank routing numbers for logging
    new_n_rn = df['bank_routing_number'].nunique()
    logger.info(f"Original number of routing numbers: {orig_n_rn}, New number of grouped routing numbers: {new_n_rn}")
    return df


def group_email_dur(df, logger):
    """
    Function which combines the 'months' email duration denominations into a 0-12 months value

    :param df: data to be preprocessed
    :param logger: Logger
    :return:
        df: data with email duration values grouped
    """

    df['email_duration'] = df['email_duration'].apply(lambda v: v if v == '1 year or more' else '0-12 months')
    logger.info("Grouped 0-6 months and 7-12 months email duration categories")
    return df


def group_residence_bank_acc_durs(df, logger):
    """
    Function which combines the 'months' residence duration and bank account duration denominations into 0-12 months
    values.

    :param df: data to be preprocessed
    :param logger: Logger
    :return:
        df: data with residence duration values and bank account duration values grouped
    """
    # Combine residence duration values
    df['residence_duration'] = df['residence_duration'].apply(lambda v: v if v in ['1-2 years', '3+ years']
                                                              else '0-12 months')
    # Combine bank account duration values
    df['bank_account_duration'] = df['bank_account_duration'].apply(lambda v: v if v in ['1-2 years', '3+ years']
                                                                    else '0-12 months')
    logger.info("Grouped 0-3 months and 4-12 months residence duration/bank account duration categories")
    return df


def calc_income_rent_feats(df, logger):
    """
    Function which calculates monthly income after rent has been paid. Since some people don't pay rent, a feature which
    incorporates these features seems logical to include. In addition, a pays rent feature is created to indicate if
    someone pays rent

    :param df: data to be preprocessed
    :param logger: Logger
    :return:
        df: dataframe with income/rent features incorporated
    """
    # Calculate the income after rent feature
    df['monthly_income_after_rent'] = df['monthly_income_amount'] - df['monthly_rent_amount']
    # Calculate boolean indicator if person pays rent
    df['pays_rent'] = df['monthly_rent_amount'].apply(lambda v: 0 if v == 0 else 1)
    logger.info("Created monthly income after rent and pays rent features")
    return df


def create_time_sample_weights(df, logger):
    """
    Create time-based sample weights so that newer samples affect the resulting model more

    :param df: data to be preprocessed
    :param logger: Logger
    :return:
        df: dataframe with sample weights included
    """
    # Create feature which contains year_month combinations from the application date
    df['year_month'] = df['application_when'].dt.to_period('M')
    # Create a sample weight dict where the samples range from 1-3 based on how old the sample is
    # (newest have a weight of 3 and oldest have a weight of 1)
    sample_weight_dict = dict(zip(df['year_month'].unique(), np.linspace(1, 3, num=df['year_month'].nunique())))
    # Map application month/year values to sample weight value
    df['SAMPLE_WEIGHT'] = df['year_month'].replace(sample_weight_dict)
    # drop year_month col created for sample_weights feat
    df.drop(['year_month'], axis=1, inplace=True)

    logger.info("Created time based sample weights")
    return df


def process_model_training_data(df, logger):
    """

    :param df: data to be preprocessed
    :param logger: Logger
    :return:
    """
    print("START: Feature engineering")

    # Shorten zip codes
    df = shorted_zipcode(df, logger)
    # Consolidate bank routing numbers
    df = group_bank_routing_nums(df, logger)
    # Group email durations
    df = group_email_dur(df, logger)
    # Group residence durations and banking account durations
    df = group_residence_bank_acc_durs(df, logger)
    # Calculate income rent/features
    df = calc_income_rent_feats(df, logger)
    # Create sample weights
    df = create_time_sample_weights(df, logger)

    logger.info(f"Preprocessed data successfully read with dimension: {df.shape}. Saving preprocessed data to file")
    df.to_csv(preprocessed_data_fp, index=False)
    print("END: Feature engineering")
    return df
