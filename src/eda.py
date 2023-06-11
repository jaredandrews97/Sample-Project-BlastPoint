"""
Module which stores main functions used during the feature engineering step of model training

This module implements the following functionality:
    1. univariate analysis plot generation
    2. bivariate analysis plot generation

Author: Jared Andrews
Date: 6/11/23
"""

import os
import seaborn as sns
import matplotlib.pyplot as plt
from src.config import eda_plots_fp, target_feature, eda_drop_feats
from src.utils import clean_folder


def univariate_analysis_plot(col, dt, logger):
    """
    Graphing of histograms for each feature in processed dataset

    :param col: col to graph histogram for
    :param dt: Data type of the column
    :param logger: Logger
    :return: None
    """
    # If col is of type object (str) and has lots of unique values, only keep 10 most frequent values so that hist
    # isn't too crowded
    if dt == object and col.nunique() > 10:
        top_value_by_vc = col.value_counts().head(10).index
        hist_col = col[col.isin(top_value_by_vc)]
    else:
        hist_col = col.copy()

    col_name = hist_col.name
    col_name_hist = col_name.replace('_', ' ').title()
    # Plot histogram
    ax = hist_col.hist()
    # Add title, xlabel, ylabel and ensure xlabels are clear
    plt.title(f"{col_name_hist} for Individuals Requesting Loans")
    plt.xlabel(col_name_hist)
    plt.ylabel("Count")
    plt.xticks(rotation=0, ha='center')

    # Save figure to file locally
    fig_fp = os.path.join(eda_plots_fp, f'{col_name}_hist.png')
    ax.figure.savefig(fig_fp, dpi=300)
    # clear plot for next image
    plt.clf()

    logger.info(f"Histogram for {col_name_hist} successfully generated and saved to file")


def bivariate_analysis_plot(df, col_name, dt, logger):
    """
    Create bi-variate plots for each feature in processed dataset

    :param df: processed dataset
    :param col_name: column to be used in plot
    :param dt: datatype of column to be used in plot
    :param logger: Logger
    :return: None
    """

    # Set colum plot name and file path to save plot to
    col_name_plot = col_name.replace('_', ' ').title()
    fig_fp = os.path.join(eda_plots_fp, f'{col_name}_scatter.png')

    # If column is an integer, plot scatter plot of col x target feature
    if dt in [float, int]:
        plot_type = 'Scatterplot'
        # Plot scatterplot
        ax = df.plot.scatter(x=col_name, y=target_feature)

        # Set title, xlabel, ylabel and ensure xlabels formatted properly
        plt.title(f"{col_name_plot} vs. Loan Quality")
        plt.xlabel(f"{col_name_plot}")
        plt.ylabel("Loan Quality")
        plt.xticks(rotation=0, ha='center')

    # If col is of type object (str) and has lots of unique values, only keep 10 most frequent values so that barplot
    # isn't too crowded
    elif dt == object:
        if df[col_name].nunique() > 10:
            top_value_by_vc = df[col_name].value_counts().head(10).index
            df_plot = df[df[col_name].isin(top_value_by_vc)]
        else:
            df_plot = df.copy()

        plot_type = 'Barplot'
        # Plot barplot
        ax = sns.barplot(x=col_name, y=target_feature, data=df_plot)
        # Set xlabel, ylabel and title
        ax.set(xlabel=col_name_plot, ylabel='Loan Quality', title=f"Loan Quality for {col_name_plot}")

    # Create plots for datetime columns
    else:
        # To remove too many individual app dates, only keep the year/month for application date plot
        if col_name == 'application_when':
            new_col_name = f'{col_name}_ym'
            df[new_col_name] = df[col_name].dt.to_period('M')

        # To remove too many individual birthdates, only keep the decade for application date plot
        else:
            new_col_name = f'{col_name}_decade'
            df[new_col_name] = df[col_name].dt.year.apply(lambda v: v - (v % 10))

        plot_type = 'Barplot'
        # Plot barplot
        ax = sns.barplot(x=new_col_name, y=target_feature, data=df)
        # Set xlabel, ylabel and title
        ax.set(xlabel=col_name_plot, ylabel='Loan Quality', title=f"Loan Quality for {col_name_plot}")

        # Drop the newly derived date column used for plotting
        df.drop(new_col_name, axis=1, inplace=True)

    # Save figure to file locally
    ax.figure.savefig(fig_fp, dpi=300)
    # clear plot for next image
    plt.clf()

    logger.info(f"{plot_type} for {col_name_plot} successfully generated and saved to file")


def generate_eda_plots(data, logger):
    """
    Main function which triggers the generation of all plots

    :param data: processed dataframe
    :param logger: Logger
    :return: None
    """

    print("START: Generation of EDA plots")
    clean_folder(eda_plots_fp)

    # Generate plots for
    for c in data.columns:
        col_dtype = data[c].dtype
        if col_dtype != '<M8[ns]':
            univariate_analysis_plot(data[c], col_dtype, logger)
        if c != target_feature:
            bivariate_analysis_plot(data, c, col_dtype, logger)

    logger.info("Plots for all features successfully generated")

    data.drop(eda_drop_feats, axis=1, inplace=True)
    logger.info(f"From EDA analysis, features dropped: \n\t{', '.join(eda_drop_feats)}")
    print("END: Generation of EDA plots")
    return data
