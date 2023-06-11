import os
import logging
import argparse
import pandas as pd
from src.data import parse_preprocess_data
from src.eda import generate_eda_plots
from src.config import raw_data_fp, logs_fp, preprocessed_data_fp
from src.feat_engineering import process_model_training_data
from src.train import train_evaluate_model
from src.utils import bool_func


if __name__ == '__main__':
    logging.basicConfig(filename=logs_fp, level=logging.INFO, filemode='w',
                        format='%(funcName)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description="Parser for running model training")

    parser.add_argument("--data_fp", type=str, help="File path for the raw training data", required=False,
                        default=raw_data_fp)

    parser.add_argument("--wandb_log", type=bool_func,
                        help="Boolean indicating if results logged as weights and bias run", required=False,
                        default=False, choices=[True, False])

    parser.add_argument("--preprocess_data", type=bool_func,
                        help="Indicate if you only want to perform data preprocessing or to also train a model",
                        required=True, choices=[True, False])

    parser.add_argument("--train_model", type=bool_func,
                        help="Indicate if you only want to perform data preprocessing or to also train a model",
                        required=True, choices=[True, False])

    parser.add_argument("--model", type=str, help="ML model algorithm to train",
                        choices=['LogisticRegression', 'RandomForest', 'XGBoost'], required=False, default=False)

    parser.add_argument("--save_locally", type=bool_func, help="Indicates if outputs should be saved to local machine",
                        choices=[True, False], required=False, default=True)
    args = parser.parse_args()

    if args.preprocess_data:
        # Run parsing and initial pre-processing of data
        preprocessed_data = parse_preprocess_data(args, logger)
        # Generate EDA plots for analysis
        generate_eda_plots(preprocessed_data, logger)

        # Process data so that it is ready to be used during model training
        preprocessed_df = process_model_training_data(preprocessed_data, logger)

    else:

        # Read in saved preprocessed data
        assert os.path.exists(preprocessed_data_fp), "Preprocessed data not saved to file. " \
                                                     "Please run data preprocessing steps"
        preprocessed_df = pd.read_csv(preprocessed_data_fp)

    if args.train_model:

        # Train and evaluate model
        train_evaluate_model(args, preprocessed_df, logger)
