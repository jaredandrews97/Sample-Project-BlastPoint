import numpy as np
from src.config import preprocessed_data_fp


def shorted_zipcode(df, logger):

    orig_n_zips = df['address_zip'].nunique()
    df['address_zip'] = df['address_zip'].str[:2]
    new_n_zips = df['address_zip'].nunique()
    logger.info(f"Original number of zipcodes: {orig_n_zips}, New number of shortened zipcodes: {new_n_zips}")
    return df


def group_bank_routing_nums(df, logger):

    orig_n_rn = df['bank_routing_number'].nunique()
    bank_rn_vc = df['bank_routing_number'].value_counts()
    bank_rn_retain = bank_rn_vc[bank_rn_vc >= 15].index
    df['bank_routing_number'] = df['bank_routing_number'].apply(lambda rn: rn if rn in bank_rn_retain else 'default')
    new_n_rn = df['bank_routing_number'].nunique()
    logger.info(f"Original number of routing numbers: {orig_n_rn}, New number of grouped routing numbers: {new_n_rn}")
    return df


def group_email_dur(df, logger):

    df['email_duration'] = df['email_duration'].apply(lambda v: v if v == '1 year or more' else '0-12 months')
    logger.info("Grouped 0-6 months and 7-12 months email duration categories")
    return df


def group_residence_bank_acc_durs(df, logger):
    df['residence_duration'] = df['residence_duration'].apply(lambda v: v if v in ['1-2 years', '3+ years']
                                                              else '0-12 months')
    df['bank_account_duration'] = df['bank_account_duration'].apply(lambda v: v if v in ['1-2 years', '3+ years']
                                                                    else '0-12 months')
    logger.info("Grouped 0-3 months and 4-12 months residence duration/bank account duration categories")
    return df


def calc_income_feats(df, logger):
    df['monthly_income_after_rent'] = df['monthly_income_amount'] - df['monthly_rent_amount']
    df['pays_rent'] = df['monthly_rent_amount'].apply(lambda v: 0 if v == 0 else 1)
    df['monthly_income_after_rent'].hist(bins=30)

    logger.info("Created monthly income after rent and pays rent features")
    return df


def create_time_sample_weights(df, logger):
    df['year_month'] = df['application_when'].dt.to_period('M')
    sample_weight_dict = dict(zip(df['year_month'].unique(), np.linspace(1, 3, num=df['year_month'].nunique())))
    df['SAMPLE_WEIGHT'] = df['year_month'].replace(sample_weight_dict)
    df.drop('year_month', axis=1, inplace=True)

    logger.info("Created time based sample weights")
    return df


def process_model_training_data(df, logger):
    print("START: Feature engineering")

    df = shorted_zipcode(df, logger)
    df = group_bank_routing_nums(df, logger)
    df = group_email_dur(df, logger)
    df = group_residence_bank_acc_durs(df, logger)
    df = calc_income_feats(df, logger)
    df = create_time_sample_weights(df, logger)

    logger.info(f"Preprocessed data successfully read with dimension: {df.shape}. Saving preprocessed data to file")
    df.to_csv(preprocessed_data_fp, index=False)
    print("END: Feature engineering")
    return df
