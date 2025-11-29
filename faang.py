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


import logging
from datetime import datetime, timedelta
import os
import pathlib as Path
import glob
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates 
import seaborn as sns
import numpy as np
import yfinance as yf

## Problem 3: Script
#
#Create a Python script called `faang.py` in the root of your repository.
#Copy the above functions into it and it so that whenever someone at the terminal types `./faang.py`, the script runs, downloading the data and creating the plot.
#Note that this will require a shebang line and the script to be marked executable.
#Explain the steps you took in your notebook.

# The Developer : Clyde Watts using github copilot to assist

# prototype for extracting stock data
tickers = ["META", "AAPL", "AMZN", "NFLX", "GOOG"]


def get_data(tickers = tickers,start_date=None, end_date=None,interval="1h",data_path="./data/",once_only=True)-> tuple[int, str, str, pd.DataFrame]:
    """
    Function to get stock data from yfinance

    Parameters:
    tickers (list): List of stock tickers to download data for
    start_date (str): Start date for data in format "YYYY-MM-DD". If None, defaults to 6 days ago.
    end_date (str): End date for data in format "YYYY-MM-DD". If None, defaults to yesterday.
    interval (str): Data interval. Default is "1h".
    data_path (str): Path to save the data. Default is "./data/".
    once_only (bool): If True, download data only once for a date and do not overwrite existing files. Default is True.
    If set to false it will delete existing files and download again.
    TODO: add only once functionality
    Returns:
       return_code : 0 for success, -1 for failure
       return_message : message indicating success or failure
       file_name (str): Name of the file where data is saved
    """
    return_code = 0
    return_message = "Success"
    file_name = None
    df_data = None
    # TODO : implement once_only functionality
    # TODO : Sort out logic of start_date and end_date for once only check , simplify
    # Get current date and time and keep it constant
    now_dttm = datetime.now()
    # if start_date is None , set to today - 7 days
    start_date_dttm =(now_dttm - timedelta(days=7)) if start_date is None else datetime.strptime(start_date, "%Y-%m-%d")
    start_date = start_date_dttm.strftime("%Y-%m-%d") if start_date is None else start_date
    start_date_dttm = datetime.strptime(start_date, "%Y-%m-%d")
    # create file name from start date
    file_name = f"{data_path}{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    # create string for glob to check if file exists for start date
    start_date_glob_str = f"{data_path}{datetime.now().strftime('%Y%m%d')}*.csv"
    if once_only and glob.glob(start_date_glob_str):
        logging.info(f"File already exists for start date {start_date}, skipping download.")
        existing_files = glob.glob(start_date_glob_str)
        file_name = existing_files[0]  # Get the first matching file
        return return_code, return_message, file_name, None

        
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
        try:
            os.makedirs(data_path)
        except Exception as e:
            logging.error(f"Error creating directory: {e}")
            return_code = -1
            return_message = f"Error creating directory: {e}"
            return return_code, return_message, None, None
    # get start date only string for file name for once only check

    # if file exists then delete it
    if os.path.exists(file_name):
        logging.info(f"Deleting existing file: {file_name}")
        try:
            os.remove(file_name)
        except Exception as e:
            logging.error(f"Error deleting file: {e}")
            return_code = -1
            return_message = f"Error deleting file: {e}"
            return return_code, return_message, None, None
    logging.info(f"Start Date: {start_date}, End Date: {end_date}")
    try:
        df_data = yf.download(tickers, interval=interval, group_by='ticker',start=start_date, end=end_date,auto_adjust=True)
    except Exception as e:
        logging.error(f"Error downloading data: {e}")
        return_code = -1
        return_message = f"Error downloading data: {e}"
        return return_code, return_message, None, None
    # Save the data to a CSV file
    df_data.to_csv(file_name)
    return return_code, return_message, file_name,df_data
   



   

#------------------------------------------------------------------------------
# Function to get the latest file from a directory
#------------------------------------------------------------------------------

def get_latest_file(data_path="./data/")-> tuple[int, str, str]:
    """
    Returns the path to the latest data file in the specified directory.

    Args:
        data_path (str): The path to the directory containing the data files.

        tuple: (return_code, return_message, latest_file) where latest_file is the path to the latest data file, or None if no files are found.
        str: The path to the latest data file, or None if no files are found.
    """
    return_code = 0
    return_message = "Success"
    latest_file = None

    logging.info(f"Getting the latest file from {data_path}")
    # File pattern
    file_pattern = "20[0-9][0-9][0-1][0-9][0-3][0-9]_[0-9][0-9][0-9][0-9][0-9][0-9].csv"
    # Add path to file pattern
    file_pattern = os.path.join(data_path, file_pattern)
    # glob searches directories for files based on a pattern
    try:
        list_of_files = glob.glob(file_pattern)
    except Exception as e:
        logging.error(f"Error occurred while searching for files: {e}")
        return_code=-1
        return_message=f"Error occurred while searching for files: {e}"
        return return_code, return_message, None
    if not list_of_files:
        logging.warning(f"No files found in {data_path} matching pattern {file_pattern}")
        return_code = -1
        return_message = f"No files found in {data_path} matching pattern {file_pattern}"
        return return_code, return_message, None
    # find the latest file based on creation time
    #    max parameters - list and function which gets "value" associated with each item in the list
    #    this gets the "youngest" file based on creation time 
    #    not necessarily the latest date in the file name - design decision 
    #    the premise is that the latest file created is the one we want to use
    latest_file = max(list_of_files, key=os.path.getctime)
    logging.info(f"Latest file: {latest_file}")
    return return_code, return_message, latest_file

#-------------------------------------------------------------------------------
# Function to get PNG filename from CSV filename
#-------------------------------------------------------------------------------
def get_PNG_filename_from_CSV_filename(csv_filename,plot_path="./plots/")-> tuple[int, str, str]:
    """
    Function to get PNG filename from CSV filename

    Parameters:
    csv_filename (str): Name of the CSV file
    plot_path (str): Path to save the PNG file. Default is "./plots/".

    Returns:
       png_filename (str): Name of the PNG file
    """
    return_code = 0
    return_message = "Success"
    # extract base name from csv_filename
    base_name = os.path.basename(csv_filename)
    # remove .csv extension
    base_name = os.path.splitext(base_name)[0]
    # create png filename
    png_filename = f"{plot_path}{base_name}.png"
    return return_code, return_message, png_filename
#------------------------------------------------------------------------------
# Function to load file into dataframe
#------------------------------------------------------------------------------
def load_file_into_dataframe(file)-> tuple[int, str, pd.DataFrame]:
    """load_file_into_dataframe

    Args:
        file (str): The path to the data file.

    Returns:
        tuple: (return_code, return_message, df) where:
            return_code (int): 0 for success, -1 for failure
            return_message (str): Success or error message
            df (pd.DataFrame): The data as a pandas DataFrame with multi-level columns
    """
    return_code = 0
    return_message = "Success"
    df = None
    
    # Check if file name is provided
    if file is None:
        logging.error("No file provided to load into dataframe.")
        return_code = -1
        return_message = "No file provided to load into dataframe."
        return return_code, return_message, None
    
    # Check if file exists
    if not os.path.exists(file):
        logging.error(f"File does not exist: {file}")
        return_code = -1
        return_message = f"File does not exist: {file}"
        return return_code, return_message, None
    
    # Load the CSV file into a DataFrame with multi-level columns
    try:
        df = pd.read_csv(file, header=[0,1], index_col=0, parse_dates=True)
        logging.info(f"Successfully loaded data from {file}. Shape: {df.shape}")
    except Exception as e:
        logging.error(f"Error loading file {file}: {e}")
        return_code = -1
        return_message = f"Error loading file {file}: {e}"
        return return_code, return_message, None
    
    return return_code, return_message, df


#------------------------------------------------------------------------------
# Function to plot data
#------------------------------------------------------------------------------
def plot_data(show_plot=False,bpi=300)-> tuple[int, str, str]:
    """plot_data

    Args:
        df (pd.DataFrame): The data as a pandas DataFrame.
        png_file_path (str): The path to save the plot image.
        bpi (int): The resolution (bits per inch) for the saved plot image.

    Returns:
        None
    """
    
    return_code = 0
    return_message = "Success"
    png_file_name = None
    # define date format string YYYY-MM-DD HH
    # date_format_str = "%m-%d %H" # Removed
    # create date formatter
    # date_formatter = plt.matplotlib.dates.DateFormatter(date_format_str) # Removed
    logging.info("Starting data plotting...")
    # Get the latest file
    return_code, return_message, png_file_name = get_latest_file()
    if return_code != 0:
        logging.error(f"Latest file retrieval failed - Return Code: {return_code}, Message: {return_message}")
        return return_code, return_message, png_file_name
    # load data into dataframe
    return_code, return_message, df_raw = load_file_into_dataframe(png_file_name)
    if return_code != 0:
        logging.error(f"File load failed - Return Code: {return_code}, Message: {return_message}, File: {png_file_name}")
        return return_code, return_message, png_file_name
    # convert csv file name to png file name
    return_code, return_message, png_file_name = get_PNG_filename_from_CSV_filename(png_file_name, "./plots/")
    if return_code != 0:
        logging.error(f"PNG file path retrieval failed - Return Code: {return_code}, Message: {return_message}, File: {png_file_path}")
        return return_code, return_message, png_file_name
    # Create plots directory if it doesn't exist
    # extract path from png_file_name
    png_path = Path.Path(png_file_name).parent 
    try:
        os.makedirs(png_path, exist_ok=True)
    except Exception as e:
        logging.error(f"Failed to create directory {png_path} - {e}")
        return 1, f"Failed to create directory {png_path}", None
    # copy dataframe to avoid modifying original
    df = df_raw.copy()
    # Convert index to EST timezone and extract date - NASDAQ data is in EST
    df['Datetime_EST'] = df.index.tz_convert('US/Eastern')
    # Extract date from datetime
    df['Date'] = df['Datetime_EST'].dt.date
    # Get start and end dates for title
    start_date = df['Date'].min()
    end_date = df['Date'].max()
    logging.info(f"Data covers from {start_date} to {end_date}")
    # ---- Plotting ----
    fig, ax = plt.subplots()
    fig.suptitle("FAANG Stock Reports", fontsize=16)
    # Define tickers globally or pass as parameter
    if df is None or png_path is None:
        logging.error("DataFrame or PNG file path is None.")
        return
    #print(date_list)
    fig.set_size_inches(14, 8)
    for ticker in tickers:
        ax.plot(df['Datetime_EST'], df[(ticker, 'Close')], label=ticker, linestyle='-', marker='o')


    # set date ticks to 90-degree rotation for readability
    # plt.xticks(rotation=90) # Removed - Handled by autofmt_xdate
    ax.grid(True, which='both', linestyle='--', linewidth=0.5) # Updated grid
    ax.set_xlabel(' Trading Date and Time ', fontsize=12)
    ax.set_ylabel('Close Price in $', fontsize=12)
    ax.set_title(f'FAANG Stock Closing Price  From {start_date} to {end_date}', fontsize=14)
    leg = ax.legend(loc='upper left', fontsize=10, bbox_to_anchor=(1, 1), borderaxespad=0.)
    leg.get_title().set_fontsize(11)
    leg.set_title('Tickers')
    
    # Split date time into major and minor ticks
    
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    # Minor Ticks: Hours (HHh)
    # Show hours 0, 6, 12, 18. Adjust 'byhour' as needed.
    ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=[0, 6, 12, 18]))
    ax.xaxis.set_minor_formatter(mdates.DateFormatter('%Hh'))
    
    # Rotate major labels for readability
    fig.autofmt_xdate(which='major', rotation=90)
    fig.autofmt_xdate(which='minor', rotation=90)
    

    plt.tight_layout()
    plt.savefig(png_file_name, dpi=bpi)
    if show_plot:
        plt.show()

    logging.info(f"Plot saved to {png_file_name}")
    plt.close(fig)
    return return_code, return_message, png_file_name
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
    #==============================================================================
    # EXTRACT (1)
    #==============================================================================
    return_code, return_message, file_name , df_data = get_data(tickers, start_date=start_date, end_date=end_date, interval=interval, data_path=data_path)
    if return_code != 0:
        logging.error(f"Data extraction failed: { return_message , return_code}")
        exit(1)
    logging.info(f"Data extraction complete. Data saved to {file_name}")
    
    
    logging.info("Extract Complete")
    logging.info("Starting to Load File")
    #==============================================================================
    # PLOT (1)
    #==============================================================================
    return_code, return_message, png_file_path = plot_data()
    if return_code != 0:
        logging.error(f"Plotting failed: { return_message , return_code}")
        exit(1)
    logging.info(f"Plotting complete. Plot saved to {png_file_path}")
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