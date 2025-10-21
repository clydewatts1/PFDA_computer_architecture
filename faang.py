#!/usr/bin/env python3
#==============================================================================
# File: faang.py
# Description: FAANG Stock Data Downloader and Plotter
# Author: Clyde Watts
# Date: 2025-10-04
# Project: PFDA_computer_architecture
# Lecture : Ian McLoughlin
#==============================================================================
# Note: This script downloads stock data for FAANG stocks and creates a plot.

import argparse
import logging
import os
import glob
from datetime import datetime, timedelta
import yfinance
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

## Problem 3: Script
#
#Create a Python script called `faang.py` in the root of your repository.
#Copy the above functions into it and it so that whenever someone at the terminal types `./faang.py`, the script runs, downloading the data and creating the plot.
#Note that this will require a shebang line and the script to be marked executable.
#Explain the steps you took in your notebook.

# The Developer : Clyde Watts using github copilot to assist

# prototype for extracting stock data
tickers = ["META", "AAPL", "AMZN", "NFLX", "GOOG"]
#------------------------------------------------------------------------------
# Functions
#------------------------------------------------------------------------------
# Function to get stock data from yfinance
# It will save the data to a CSV file in the specified data_path
# It will return the data as a DataFrame
# If start_date is None, it will default to 6 days ago
# If end_date is None, it will default to yesterday
#------------------------------------------------------------------------------
def get_data(tickers = tickers,start_date=None, end_date=None,interval="1h",data_path="./data/"):
    """
    Function to get stock data from yfinance

    Parameters:
    tickers (list): List of stock tickers to download data for
    start_date (str): Start date for data in format "YYYY-MM-DD". If None, defaults to 6 days ago.
    end_date (str): End date for data in format "YYYY-MM-DD". If None, defaults to yesterday.
    interval (str): Data interval. Default is "1h".
    data_path (str): Path to save the data. Default is "./data/".
    Returns:
    df_data (DataFrame): DataFrame containing the stock data
    file_name (str): Name of the file where data is saved
    """
    ## TODO: start with the end date , and then go back 5 trading days to get the start date , if start_date is None
    # is not configured
    if start_date is None:
        # Variation of the fence posting to get last 7 days of data
        start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        # create file name based on current date and time
        file_name = f"{data_path}{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    else:
        # convert start_date to datetime object
        start_date_time = datetime.strptime(start_date, "%Y-%m-%d")
        # create file name based on start time and 23:59:59 of end date
        start_date_str = start_date_time.strftime("%Y%m%d") + "_235959"
        file_name = f"{data_path}{start_date_str}.csv"
    # if end_date is None , set to today - 0 days this means yesterday's data inclusive
    if end_date is None:
        end_date = (datetime.now() - timedelta(days=0)).strftime("%Y-%m-%d")
    else: # convert end_date to datetime object
        end_date_time = datetime.strptime(end_date, "%Y-%m-%d")
        # add 1 day to end_date to make it inclusive
        end_date = (end_date_time + timedelta(days=1)).strftime("%Y-%m-%d")
    # check if directory exists
    if not os.path.exists(data_path):
        logging.info(f"Creating directory: {data_path}")
        os.makedirs(data_path)
    # if file exists then delete it
    if os.path.exists(file_name):
        logging.info(f"Deleting existing file: {file_name}")
        os.remove(file_name)
    logging.info(f"Start Date: {start_date}, End Date: {end_date}")
    df_data = yfinance.download(tickers, interval=interval, group_by='ticker',start=start_date, end=end_date)
    # Save the data to a CSV file
    df_data.to_csv(file_name)
    return df_data
#------------------------------------------------------------------------------
# Function to get the latest file from the data_path
# It will return the file name
# ------------------------------------------------------------------------------
def get_the_latest_file(data_path="./data/"):
    """get_the_latest_file

    Args:
        data_path (str): The path to the directory containing the data files.

    Returns:
        str: The path to the latest data file, or None if no files are found.
    """

    logging.info(f"Getting the latest file from {data_path}")
    # File pattern
    file_pattern = "20[0-9][0-9][0-1][0-9][0-3][0-9]_[0-9][0-9][0-9][0-9][0-9][0-9].csv"
    # Add path to file pattern
    file_pattern = os.path.join(data_path, file_pattern)
    list_of_files = glob.glob(file_pattern) 
    if not list_of_files:
        logging.warning(f"No files found in {data_path} matching pattern {file_pattern}")
        return None
    latest_file = max(list_of_files, key=os.path.getctime)
    logging.info(f"Latest file: {latest_file}")
    return latest_file
#------------------------------------------------------------------------------
# Function to load a CSV file into a pandas DataFrame
# It will return the DataFrame
# ------------------------------------------------------------------------------
def load_file_into_dataframe(file):
    """load_file_into_dataframe

    Args:
        file (str): The path to the data file.

    Returns:
        pd.DataFrame: The data as a pandas DataFrame.
    """
    if file is None:
        logging.error("No file provided to load into dataframe.")
        return None
    df = pd.read_csv(file, header=[0,1], index_col=0, parse_dates=True)
    return df
#------------------------------------------------------------------------------
# Function to plot the data
# It will save the plot to a PNG file
# ------------------------------------------------------------------------------
def plot_data(df, png_file_path):
    """plot_data

    Args:
        df (pd.DataFrame): The data as a pandas DataFrame.
        png_file_path (str): The path to save the plot image.

    Returns:
        None
    """
    if df is None or png_file_path is None:
        logging.error("DataFrame or PNG file path is None.")
        return
    plt.figure(figsize=(14, 8))
    for ticker in tickers:
        plt.plot(df.index, df[(ticker, 'Close')], label=ticker)
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.title('Stock Prices Over Time')
    plt.legend()
    plt.savefig(png_file_path)
    logging.info(f"Plot saved to {png_file_path}")
#------------------------------------------------------------------------------
# Main function to parse arguments and call other functions
#------------------------------------------------------------------------------
def main():
    # Get command line arguments 
    # it will use defaults to project requirements
    parser = argparse.ArgumentParser(description="FAANG Stock Data Downloader and Plotter")
    parser.add_argument('--start_date', type=str, help='Start date for data in format "YYYY-MM-DD". If not provided, defaults to 6 days ago.')
    parser.add_argument('--end_date', type=str, help='End date for data in format "YYYY-MM-DD". If not provided, defaults to yesterday.')
    parser.add_argument('--interval', type=str, default='1h', help='Data interval. Default is "1h".')
    parser.add_argument('--data_path', type=str, default='./data/', help='Path to save the data. Default is "./data/".')
    parser.add_argument('--tickers', type=str, nargs='+', default=["META", "AAPL", "AMZN", "NFLX", "GOOG"], help='List of stock tickers to download data for. Default is FAANG stocks.')
    parser.add_argument('--log_level', type=str, default='INFO', help='Logging level. Default is "INFO".')
    parser.add_argument('--log_path', type=str, default='./logs/', help='Path to save the logs. Default is "./logs/".') 
    parser.add_argument('--report_name', type=str, default='standard', help='This is the report to be generated. standard is the default based on problem/project requirements.')
    # Note : the --help is implied and automatically added by argparse
    args = parser.parse_args()
    # setup variables from args
    start_date = args.start_date
    end_date = args.end_date
    if end_date is None:
        end_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        logging.info(f"End date not provided. Defaulting to yesterday: {end_date}")
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=6)).strftime("%Y-%m-%d")
        logging.info(f"Start date not provided. Defaulting to 6 days ago: {start_date}")
    interval = args.interval
    data_path = args.data_path
    log_level = args.log_level
    tickers = args.tickers
    log_path = args.log_path
    report_name = args.report_name
    png_file_path = None
    # setup logging , default to INFO
    # the default will be to logs directory with the file name faang_YYYYMMDD_HHMMSS.log and to console
    log_file = os.path.join(log_path, f"faang_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(level=log_level, handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ])
    logging.info("Logging setup complete.")
    # Log the arguments for debugging, one per line
    logging.info(f"start_date={start_date}")
    logging.info(f"end_date={end_date}")
    logging.info(f"interval={interval}")
    logging.info(f"data_path={data_path}")
    logging.info(f"tickers={tickers}")
    logging.info(f"log_level={log_level}")
    logging.info(f"log_path={log_path}")
    logging.info(f"report_name={report_name}")
    logging.info("Starting to Extract Data from Source yfinance")
    df_data = get_data(tickers, start_date=start_date, end_date=end_date, interval=interval, data_path=data_path)
    logging.info("Extract Complete")
    logging.info("Starting to Load File")
    latest_file = get_the_latest_file(data_path=data_path)
    df_loaded = load_file_into_dataframe(latest_file)
    logging.info("Load File Complete")
    logging.info("Starting to Generate Report")
    # convert latest_file to png file name
    if latest_file is None:
        logging.error("No latest file found. Cannot generate report.")
        return
    png_file_path = latest_file.replace('.csv', '.png')
    plot_data(df_loaded, png_file_path)
    logging.info("Report Complete")
    logging.info("Complete Main")

#------------------------------------------------------------------------------
# MAIN Section
#------------------------------------------------------------------------------

if __name__ == '__main__':
    main()

#------------------------------------------------------------------------------
# The End
#------------------------------------------------------------------------------