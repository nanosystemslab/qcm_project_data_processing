#! /usr/bin/env python3

import argparse
import glob
import logging
import os
import pathlib
import re
import sys
import types

import dateutil.parser as dup
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import pandas.plotting
import seaborn as sns
from scipy.integrate import trapz
from pathlib import Path
import io




__version__ = "0.0.1"

"""

"""
def read_and_filter_file(filepath):
    filtered_lines = []
    with open(filepath, 'r') as file:
        for line in file:
            if "None" not in line:
                filtered_lines.append(line.strip())

    return filtered_lines

def parse_lines_to_df(lines):
    # Define an empty list to store parsed data
    data = []

    # Iterate over each filtered line
    for line in lines:

        if line.startswith("A"):  # Assuming you're only interested in lines starting with "A"
            parts = line.split()
            try:
                # Parse line parts into structured data. Adjust indices as needed.
                record = {
                    "Column1": parts[1],
                    "Column2": parts[2],
                    "Column3": parts[3],
                    "Column4": parts[4],
                }

                data.append(record)
            except IndexError:
                print(f"Error parsing line: {line}. Skipping...")
            except Exception as e:
                print(f"An unexpected error occurred: {e}. Skipping...")

        # Add more conditions here if there are other line formats you're interested in
    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(data)

    # Remove leading zeros and '+' characters from each column
    for column in df.columns:
        df[column] = df[column].str.replace('+', '', regex=False)
        df[column] = df[column].str.lstrip('0')

    # Replace empty strings with NaN
    df.replace('', np.nan, inplace=True)

    # Convert DataFrame to float
    df = df.astype(float)
    return df


def load_file_qcm(filepath):
    log = logging.getLogger(__name__)
    log.debug("in")

    df = None
    print(filepath)

    if ".txt" in filepath:
        n_col = ["realTime", "tubeTemp", "chamberTemp"]
        df = pd.read_csv(filepath, skiprows=0, header=None, names=n_col)
        df = df.dropna()
        df["realTime"] = pd.to_datetime(df['realTime'], format='%H:%M:%S')
        df['elapsedTime'] = df['realTime'] - df['realTime'].iloc[0]
        df['elapsedTime'] = df['elapsedTime'].dt.total_seconds()

    if "pressure.txt" in filepath:
        filtered_lines = read_and_filter_file(filepath)
        df = parse_lines_to_df(filtered_lines)

    if ".log" in filepath:
        try:
            def is_valid_line(line):
                parts = line.strip().split(',')
                return len(parts) == 5 and all(part.strip().replace('.', '', 1).isdigit() or part.strip().lstrip('-').replace('.', '', 1).isdigit() for part in parts[:-1])

            valid_lines = []
            with open(filepath, 'r') as file:
                for line in file:
                    if is_valid_line(line):
                        valid_lines.append(line)

            # Create a DataFrame from the valid lines
            df = pd.read_csv(io.StringIO(''.join(valid_lines)), header=None, skiprows=2)
            df = df.iloc[:, :-1]  # Drop the last column
            df = df.dropna()
            df.columns = ["Time", "MassRate", "Mass", "Frequency"]
            df = df.dropna()
            df = df.astype(float)
            print(df)
        except pd.errors.ParserError as e:
            print(f"ParserError: {e}")
            # Read specific lines to diagnose
            with open(filepath, 'r') as file:
                lines = file.readlines()
                for i, line in enumerate(lines):
                    print(f"Line {i+1}: {line.strip()}")
            # Re-raise the error to halt the program for further investigation
            raise

    log.debug("out")
    return df


def load_file_multi(data_paths=None):
    log = logging.getLogger(__name__)
    log.debug("in")

    trans_df = []
    for filename in data_paths:
        print(filename)
        df = load_file_TT(filename)
        test_trial = df.meta.trial
        test_run_num = df.meta.test_run
        trans_df.append(df)

    log.debug("out")
    return trans_df, test_trial


def plot_data_f6_t():
    log = logging.getLogger(__name__)
    log.debug("in")

    plot_param_options = {
        "realTime":{'label':"Time", 'hi':300, 'lo':0, 'lbl':"Time (sec)"},
        "elapsedTime":{'label':"Time", 'hi':300, 'lo':0, 'lbl':"Time (sec)"},
        "tubeTemp":{'label':"Force", 'hi':400, 'lo':0, 'lbl':"Force (N)"},
        "chamberTemp":{'label':"Stroke", 'hi':4, 'lo':0, 'lbl':"Stroke (mm)"},
        "Time":{'label':"Stress", 'hi':50, 'lo':0, 'lbl':"Stress (MPa)"},
        "MassRate":{'label':"Strain", 'hi':0.15, 'lo':0, 'lbl':"Strain (mm/mm)"},
        "Mass":{'label':"Strain", 'hi':0.15, 'lo':0, 'lbl':"Strain (mm/mm)"},
        "Frequency":{'label':"Strain", 'hi':0.15, 'lo':0, 'lbl':"Strain (mm/mm)"},
    }

    df = load_file_qcm("./data/dataset3.txt")
    print(df)

    figsize = 4  # 12 has been default
    figdpi = 600
    hwratio = 4./3
    fig = plt.figure(figsize=(figsize * hwratio, figsize), dpi=figdpi)
    ax = fig.add_subplot(111)
    with sns.axes_style("darkgrid"):

        N = 10
        elapsedTime_downsampled = df["elapsedTime"][::N] - 12000
        chamberTemp_downsampled = df["chamberTemp"][::N]
        tubeTemp_downsampled = df["tubeTemp"][::N]


        ax.plot(elapsedTime_downsampled, chamberTemp_downsampled, label="Chamber", color="#43d1e8",
                linestyle='-', lw=1, marker='o', markevery=10)
        ax.plot(elapsedTime_downsampled, tubeTemp_downsampled, label="Tube", color="#ef60a3",
                linestyle='--', lw=1, marker='x', markevery=10)

        ax.set_xlim([0, 6000])
        ax.set_ylim([69.9, 70.1])

        ax.set_xlabel("Elapsed Time [s]")
        ax.set_ylabel("Temperature [°C]")
        ax.set_title("Experimental Phase: Transient Temperature",loc='left')

        ax.legend(frameon=True)

        logging.info(sys._getframe().f_code.co_name)
        plot_fn = "out/figure_6t.png"
        logging.info("write plot to: {}".format(plot_fn))
        fig.savefig(plot_fn, bbox_inches='tight')
    return

def plot_data_f6_b():
    log = logging.getLogger(__name__)
    log.debug("in")

    plot_param_options = {
        "realTime":{'label':"Time", 'hi':300, 'lo':0, 'lbl':"Time (sec)"},
        "elapsedTime":{'label':"Time", 'hi':300, 'lo':0, 'lbl':"Time (sec)"},
        "tubeTemp":{'label':"Force", 'hi':400, 'lo':0, 'lbl':"Force (N)"},
        "chamberTemp":{'label':"Stroke", 'hi':4, 'lo':0, 'lbl':"Stroke (mm)"},
        "Time":{'label':"Stress", 'hi':50, 'lo':0, 'lbl':"Stress (MPa)"},
        "MassRate":{'label':"Strain", 'hi':0.15, 'lo':0, 'lbl':"Strain (mm/mm)"},
        "Mass":{'label':"Strain", 'hi':0.15, 'lo':0, 'lbl':"Strain (mm/mm)"},
        "Frequency":{'label':"Strain", 'hi':0.15, 'lo':0, 'lbl':"Strain (mm/mm)"},
    }

    df = load_file_qcm("./data/dataset3.txt")
    df = df[df['elapsedTime'] > 12000]
    print(df)

    figsize = 4  # 12 has been default
    figdpi = 600
    hwratio = 4./3
    fig = plt.figure(figsize=(figsize * hwratio, figsize), dpi=figdpi)
    ax = fig.add_subplot(111)
    with sns.axes_style("darkgrid"):

        #sns.kdeplot(np.array( df["chamberTemp"]), bw=0.5, label="Chamber", fill=True, fill_kwds={'hatch': '///'})
        kde = sns.kdeplot(np.array(df["chamberTemp"]), bw_method=0.5, label="Chamber", color="#43d1e8",fill=True)
        for area in kde.collections:
            area.set_hatch('///')

        sns.kdeplot(np.array( df["tubeTemp"]), bw=0.5, label="Tube", color="#ef60a3",fill=True,alpha=0.5)
        ax.set_ylabel("Number of Occurrences")
        ax.set_xlabel("Temperature [°C]")
        ax.set_title("Experimental Phase: Transient Temperature",loc='left')
        ax.set_xlim([69.9, 70.1])
        # Function to format x-axis labels
        def format_func(value, tick_number):
            return f'{value:.2f}'

        ax.xaxis.set_major_formatter(FuncFormatter(format_func))

        ax.legend(frameon=True)

        logging.info(sys._getframe().f_code.co_name)
        plot_fn = "out/figure_6b.png"
        logging.info("write plot to: {}".format(plot_fn))
        fig.savefig(plot_fn, bbox_inches='tight')
    return



def plot_data_f7():
    log = logging.getLogger(__name__)
    log.debug("in")

    plot_param_options = {
        "realTime":{'label':"Time", 'hi':300, 'lo':0, 'lbl':"Time (sec)"},
        "elapsedTime":{'label':"Time", 'hi':300, 'lo':0, 'lbl':"Time (sec)"},
        "tubeTemp":{'label':"Force", 'hi':400, 'lo':0, 'lbl':"Force (N)"},
        "chamberTemp":{'label':"Stroke", 'hi':4, 'lo':0, 'lbl':"Stroke (mm)"},
        "Time":{'label':"Stress", 'hi':50, 'lo':0, 'lbl':"Stress (MPa)"},
        "MassRate":{'label':"Strain", 'hi':0.15, 'lo':0, 'lbl':"Strain (mm/mm)"},
        "Mass":{'label':"Strain", 'hi':0.15, 'lo':0, 'lbl':"Strain (mm/mm)"},
        "Frequency":{'label':"Strain", 'hi':0.15, 'lo':0, 'lbl':"Strain (mm/mm)"},
    }

    df = load_file_qcm("./data/dataset2-pressure.txt")
    df = df[df.index > 100]
    print(df)

    figsize = 4  # 12 has been default
    figdpi = 600
    hwratio = 4./3
    fig = plt.figure(figsize=(figsize * hwratio, figsize), dpi=figdpi)
    ax = fig.add_subplot(111)
    x = df.index
    y = df["Column4"]


    window_size = 10
    y_filtered = remove_spikes(y, window_size)


    with sns.axes_style("darkgrid"):

        ax.plot(x, y_filtered, color="#67b346", label ="Mass Flow")

        ax.set_ylabel("Pressure [psia]")
        ax.set_xlabel("Time [sec]")
        ax.set_title("Experimental Phase: Transient Mass Flow",loc='left')
        ax.legend(frameon=True)

        logging.info(sys._getframe().f_code.co_name)
        plot_fn = "out/figure_7.png"
        logging.info("write plot to: {}".format(plot_fn))
        fig.savefig(plot_fn, bbox_inches='tight')
    return


def plot_data_f8():
    log = logging.getLogger(__name__)
    log.debug("in")

    plot_param_options = {
        "realTime":{'label':"Time", 'hi':300, 'lo':0, 'lbl':"Time (sec)"},
        "elapsedTime":{'label':"Time", 'hi':300, 'lo':0, 'lbl':"Time (sec)"},
        "tubeTemp":{'label':"Force", 'hi':400, 'lo':0, 'lbl':"Force (N)"},
        "chamberTemp":{'label':"Stroke", 'hi':4, 'lo':0, 'lbl':"Stroke (mm)"},
        "Time":{'label':"Stress", 'hi':50, 'lo':0, 'lbl':"Stress (MPa)"},
        "MassRate":{'label':"Strain", 'hi':0.15, 'lo':0, 'lbl':"Strain (mm/mm)"},
        "Mass":{'label':"Strain", 'hi':0.15, 'lo':0, 'lbl':"Strain (mm/mm)"},
        "Frequency":{'label':"Strain", 'hi':0.15, 'lo':0, 'lbl':"Strain (mm/mm)"},
    }

    df = load_file_qcm("./data/dataset2-pressure.txt")
    df = df[df.index > 100]
    print(df)

    figsize = 4  # 12 has been default
    figdpi = 600
    hwratio = 4./3
    fig = plt.figure(figsize=(figsize * hwratio, figsize), dpi=figdpi)
    ax = fig.add_subplot(111)
    with sns.axes_style("darkgrid"):

        ax.plot(df.index, df["Column1"],
            color="#ff7f32", label ="Pressure")

        ax.set_ylabel("Pressure [psia]")
        ax.set_xlabel("Time [sec]")
        ax.set_title("Experimental Phase: Transient Pressure",loc='left')
        ax.legend(frameon=True)

        logging.info(sys._getframe().f_code.co_name)
        plot_fn = "out/figure_8.png"
        logging.info("write plot to: {}".format(plot_fn))
        fig.savefig(plot_fn, bbox_inches='tight')
    return

def remove_spikes(data, window_size):
    # Apply median filter to smooth out sudden jumps
    smoothed_data = np.copy(data)
    for i in range(len(data)):
        lower_bound = max(0, i - window_size // 2)
        upper_bound = min(len(data), i + window_size // 2 + 1)
        smoothed_data[i] = np.median(data[lower_bound:upper_bound])
    return smoothed_data


def plot_data_f10(time_unit='seconds'):
    log = logging.getLogger(__name__)
    log.debug("in")

    plot_param_options = {
        "realTime": {'label': "Time", 'hi': 300, 'lo': 0, 'lbl': "Time (sec)"},
        "elapsedTime": {'label': "Time", 'hi': 300, 'lo': 0, 'lbl': "Time (sec)"},
        "tubeTemp": {'label': "Force", 'hi': 400, 'lo': 0, 'lbl': "Force (N)"},
        "chamberTemp": {'label': "Stroke", 'hi': 4, 'lo': 0, 'lbl': "Stroke (mm)"},
        "Time": {'label': "Stress", 'hi': 50, 'lo': 0, 'lbl': "Stress (MPa)"},
        "MassRate": {'label': "Strain", 'hi': 0.15, 'lo': 0, 'lbl': "Strain (mm/mm)"},
        "Mass": {'label': "Strain", 'hi': 0.15, 'lo': 0, 'lbl': "Strain (mm/mm)"},
        "Frequency": {'label': "Strain", 'hi': 0.15, 'lo': 0, 'lbl': "Strain (mm/mm)"},
    }

    df1 = load_file_qcm("./data/stmData/04-23-24/STM2_62_042324.log")
    df2 = load_file_qcm("./data/stmData/05-10-24/STM2_66_051024.log")
    df3 = load_file_qcm("./data/stmData/05-11-24/STM2_66_051124.log")
    df4 = load_file_qcm("./data/stmData/05-01-24/STM2_63_050124.log")
    df5 = load_file_qcm("./data/stmData/05-02-24/STM2_62_050224.log")
    dfs = [df1, df2, df3, df4, df5]
    df_paper = load_file_qcm("./data/stmData/05-31/STM2_616-2.log")
    df_paper = df_paper[(df_paper["Time"] >= 1000)]
    df_paper["Time"] = df_paper["Time"] - 1000

    # Convert time based on the chosen time unit
    if time_unit == 'hours':
        df_paper["Time"] = df_paper["Time"] / 3600

    markers = ['o', 's', '^', 'D', 'v', 'P', '*']  # List of unique markers
    figsize = 4  # 12 has been default
    figdpi = 600
    hwratio = 4./3
    fig = plt.figure(figsize=(figsize * hwratio, figsize), dpi=figdpi)
    ax = fig.add_subplot(111)

    # Downsample factor
    n = 75

    with sns.axes_style("darkgrid"):
        i = 1

        all_x = []
        all_y = []


        for j, df in enumerate(dfs):
            df_truncated = df[(df["Time"] >= 7200) & (df["Time"] <= 18000)]
            df_truncated["Time"] = df_truncated["Time"] - 7200
            # Convert time based on the chosen time unit
            if time_unit == 'hours':
                df_truncated["Time"] = df_truncated["Time"] / 3600
            x = df_truncated["Time"][::n]
            y = df_truncated["Frequency"][::n] - df_truncated["Frequency"].max()
            #ax.plot(x,y , label=f"Test {i}", linewidth=2, marker=markers[(j + 1) % len(markers)])
            all_x.append(x)
            all_y.append(y)
            max_t = df_truncated["Time"].max()
            i += 1

        # Concatenate all x and y data
        concat_x = pd.concat(all_x, axis=1).reset_index(drop=True)
        concat_y = pd.concat(all_y, axis=1).reset_index(drop=True)
        concat_x['average_x'] = concat_x.mean(axis=1)
        concat_y['average_y'] = concat_y.mean(axis=1)
        concat_y['std_y'] = concat_y.std(axis=1)


        # 67b346
        # ffcc52
        # f4364c
        # ef60a3
        # ff7f32
        # 024731
        # 43d1e8
        set_color = "#f32e45"

        # Ensure that the indices of concat_y can be used for plotting (you might need to reset or define these if not already set)
        ax.plot(concat_x['average_x'], concat_y['average_y'], label='Average Frequency', color=set_color)

        # Fill between the average and ±1 standard deviation
        ax.fill_between(concat_x['average_x'],
                        concat_y['average_y'] - concat_y['std_y'],
                        concat_y['average_y'] + concat_y['std_y'],
                        color=set_color, alpha=0.2, label='Frequency STD')

        # Plot the "Pham et al." data, assuming df_paper and 'markers' are defined
        pham_plot = ax.plot(df_paper["Time"][::n], df_paper["Frequency"][::n] - df_paper["Frequency"].max(),
                            label="Pham et al.", linestyle='--', linewidth=2, marker=markers[0], color="#aa2030")[0]

        ax.set_ylabel("$\Delta$ Frequency [Hz]")
        ax.set_xlabel(f"Time [{time_unit}]")
        ax.set_title("Experimental QCM Frequency Data", loc='left')
        if time_unit == 'hours':
            ax.set_xlim([0, 8000 / 3600])  # Convert the x-axis limit from seconds to hours
        else:
            ax.set_xlim([0, max_t])


        # Create a combined line and patch element for the legend
        legend_elements = [
            Line2D([0], [0], color=set_color, lw=2, label='Avg. Frequency ± stdev'),
            Patch(facecolor=set_color, edgecolor='none', alpha=0.2)  # Represents the fill
        ]

        # Create a legend by explicitly including the plot of "Pham et al."
        ax.legend(handles=[(legend_elements[0], legend_elements[1]), pham_plot], labels=['Avg. Frequency ± Std. Dev.', 'Pham et al.'])

        logging.info(sys._getframe().f_code.co_name)
        plot_fn = "out/figure_10.png"
        logging.info("write plot to: {}".format(plot_fn))
        fig.savefig(plot_fn, bbox_inches='tight')
    return

def plot_data_f11():
    log = logging.getLogger(__name__)
    log.debug("in")

    data = {
    ('70C', 0.1): [3.94, 3.28, 2.2],
    ('70C', 0.2): [5.14, 4.08, 3.6],
    ('70C', 0.3): [5.56, 7.28, 5.81],
    ('70C', 0.4): [11.2, 8.62, 10],
    ('55C', 0.1): [4.67, 4.8, 4.9],
    ('55C', 0.2): [6.3, 7.49, 6.05],
    ('55C', 0.3): [7.25, 10.18, 10.67],
    ('55C', 0.4): [9.1, 11.2, 12.08],
    }

    # Creating the DataFrame
    df = pd.DataFrame(data, index=['delta_f 1', 'delta_f 2', 'delta_f 3'])
    df.loc['average'] = df.mean()
    df.loc['std'] = df.std()/0.0815
    # Cf	0.0815	Hz.ng-1.cm-2
    df.loc['delta_m'] = df.loc['average'] / 0.0815


    figsize = 4  # 12 has been default
    figdpi = 600
    hwratio = 4./3
    fig = plt.figure(figsize=(figsize * hwratio, figsize), dpi=figdpi)
    ax = fig.add_subplot(111)
    with sns.axes_style("darkgrid"):

        # Plotting the 70C line with standard deviation as error bars
        ax.errorbar(df.columns.levels[1], df.loc['delta_m', '70C'],
                    yerr=df.loc['std', '70C'], label="$70^\circ C$", color="#43d1e8",
                    linestyle='-', lw=1.5, marker='o', capsize=5, ecolor='black')

        # Plotting the 55C line with standard deviation as error bars
        ax.errorbar(df.columns.levels[1], df.loc['delta_m', '55C'],
                    yerr=df.loc['std', '55C'], label="$55^\circ C$", color="#ef60a3",
                    linestyle='-', lw=1.5, marker='d', capsize=5, ecolor='black')


        ax.axvline(x=0.1, ymin=0, ymax=100, color="#8CC739", linestyle='--')
        ax.axvline(x=0.3, ymin=0, ymax=100, color="#8CC739", linestyle='--')


        min_y_55C = np.min(df.loc['delta_m', '55C'])
        print(min_y_55C)

        # Assuming x-values are stored in df.columns.levels[1]
        x_values = df.columns.levels[1]

        # Find the indices of x-values that are closest to 0.2 and 0.3
        x_02_idx = (np.abs(x_values - 0.2)).argmin()
        x_03_idx = (np.abs(x_values - 0.3)).argmin()
        x_values_at_02_03 = [x_values[x_02_idx], x_values[x_03_idx]]

        # Extracting the y-values for 70C at x closest to 0.2 and 0.3
        y_values_at_02_03 = np.array([df.loc['delta_m', '70C'].iloc[x_02_idx], df.loc['delta_m', '70C'].iloc[x_03_idx]])

        # Displaying the array
        x_intersection = x_values_at_02_03[0] + (min_y_55C - y_values_at_02_03[0]) * (x_values_at_02_03[1] - x_values_at_02_03[0]) / (y_values_at_02_03[1] - y_values_at_02_03[0])

        ax.hlines(y=min_y_55C, xmin=0.1, xmax=x_intersection, color="#8CC739", linestyle='--')

        # Define the color
        color = "#8CC739"
        x_coords = [x_intersection, 0.1]
        y_coords = [min_y_55C, min_y_55C]

        # Plot the points using scatter with 'X' markers
        ax.scatter(x_coords, y_coords, color=color, s=64, edgecolor='black', linewidth=1.5, marker='X', label='C-C Eq. pt.')

        ax.vlines(x=x_intersection, ymin=0, ymax=min_y_55C+20, color="#8CC739", linestyle='--', linewidth=1.5)
        ax.set_ylim(10, None)

        # Plot the horizontal line with arrows pointing outward
        ax.annotate(
            '', xy=(0.1, 120), xytext=(0.3, 120),
            arrowprops=dict(arrowstyle='<|-|>', color=color, linewidth=1.5)
        )

        # Add the label "Monolayer" above the line
        ax.text(0.2, 123, 'Monolayer', ha='center', va='bottom', fontsize=12, color=color)

        # Add x_intersection to the x-axis as a tick
        current_ticks = ax.get_xticks()
        formatted_x_intersection = round(x_intersection, 3)  # Format to two significant figures
        ax.set_xticks(np.append(current_ticks, formatted_x_intersection))

        # Adjust tick labels spacing by rotating them slightly
        ax.tick_params(axis='x', rotation=45)

        # Set the color of the new tick
        tick_labels = ax.get_xticklabels()
        tick_labels[-1].set_color("#8CC739")
        ax.set_xticklabels(tick_labels)

        ax.set_ylabel("$\Delta m$ $(ng/cm^2)$")
        ax.set_xlabel("Partial Pressure ($p/p_0$)")
        ax.set_title("Isosteric Heat of Adsorption from Clausius - Clapeyron Equation",loc='left')

        ax.legend(frameon=True)

        logging.info(sys._getframe().f_code.co_name)
        plot_fn = "out/figure_11.png"
        logging.info("write plot to: {}".format(plot_fn))
        fig.savefig(plot_fn, bbox_inches='tight')
    return


def setup_logging(verbosity):
    log_fmt = ("%(levelname)s - %(module)s - "
               "%(funcName)s @%(lineno)d: %(message)s")
    # addl keys: asctime, module, name
    logging.basicConfig(filename=None,
                        format=log_fmt,
                        level=logging.getLevelName(verbosity))

    return

def parse_command_line():
    parser = argparse.ArgumentParser(description="Analyse sensor data")
    parser.add_argument("-V", "--version", "--VERSION", action="version",
                        version="%(prog)s {}".format(__version__))
    parser.add_argument("-v", "--verbose", action="count", default=0,
                        dest="verbosity", help="verbose output")
    # -h/--help is auto-added
    parser.add_argument("-d", "--dir", dest="dirs",
                        # action="store",
                        nargs='+',
                        default=None, required=False, help="directories with data files")
    parser.add_argument("-i", "--in", dest="input",
                        # action="store",
                        nargs='+',
                        default=None, required=False, help="path to input")
    ret = vars(parser.parse_args())
    ret["verbosity"] = max(0, 30 - 10 * ret["verbosity"])

    return ret

def main():
    cmd_args = parse_command_line()
    setup_logging(cmd_args['verbosity'])

    plot_data_f6_t()
    plot_data_f6_b()
    plot_data_f7()
    plot_data_f8()
    plot_data_f10()
    plot_data_f11()


if "__main__" == __name__:
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("exited")



