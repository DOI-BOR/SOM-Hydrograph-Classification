import numpy as np
import pandas as pd
import glob, os, copy
from minisom import MiniSom
import matplotlib.pyplot as plt
import sklearn
from sklearn.cluster import MeanShift
from som_functions import *


if __name__ == "__main__":

    ### User inputs ##
    # Name of the folder that timeseries files are contained within. Data should be in ft^3/s.
    input_path = "OUTPUT"

    # Number of days for each simulation
    number_of_window_days = 15

    # Number of entries per day
    sample_freq = 24

    # If user wants to generate metric cluster plots and SOM cell hydrograph plots, set plots to True below
    # Generating plots increases runtime
    plots = True

    # A correction factor for clustering bandwidth - user may need to tweak
    correction = 1.25

    # y-axis min and max values for fixed range cluster plots
    # User will most likely need to adjust based on dataset if fixed range cluster plots are desired
    min_y = 0
    max_y = 7000

    ################################################################################################################################################################################
    ### Extracting SEFM data ###

    # Makes an array of every hour in the simulation from 0 to the final hour
    d_indices = np.arange(0, number_of_window_days * sample_freq, 1)

    # Initializes a dataframe to store basin hydrographs in
    hydros = pd.DataFrame(columns=d_indices)

    print("Importing data...")

    # Uses glob package to go through the input file and read in all .plt files
    i = 0
    for file in glob.glob(str(input_path) + '/' + '*.PLT'):
        # Reads hydrograph data from file (each file is one hydrograph)
        data = np.genfromtxt(file, delimiter=' ', skip_header=2)

        # Adds hydrograph as one row in a dataframe containing all hydrographs from the file
        hydros.loc[i] = data
        i += 1

    print("Done importing data. Number of hydrographs:", len(hydros))

    # Creates a sample frequency column to the df that tells how often samples are taken each day
    freq_column = []
    for i in range (len(hydros)):
        freq_column.append(sample_freq)

    # Adds the column to the dataframe with the hydrographs
    hydros.insert(0, "Samples Per Day", freq_column)

    ### Creates folders for script file outputs ###
    # Makes an overall path to hold all results
    results_path = os.path.abspath("results")

    # Creates paths for cluster plots, distribution plots, and metric spreadsheets in outputs
    clusters_path = os.path.join(results_path, "Cluster_plots/")
    fixed_clusters_path = os.path.join(results_path, "Fixed_Range_Cluster_Plots/")
    distributions_path = os.path.join(results_path, "Distributions/")
    metrics_path = os.path.join(results_path, 'Cluster_metrics/')

    # If any of the folders above do not exist, they are created
    if not os.path.isdir(results_path):
        os.makedirs(results_path)

    if not os.path.isdir(clusters_path):
        os.makedirs(clusters_path)

    if not os.path.isdir(fixed_clusters_path):
        os.makedirs(fixed_clusters_path)

    if not os.path.isdir(metrics_path):
        os.makedirs(metrics_path)

    ### Calculating Largest Peak Value and Number of Peaks per Hydrograph Prior to the SOM ###
    # creates a dataset without the frequency column for hydrograph analysis
    # allows the script file to be restarted from this cell after adjustments are made instead of re-reading all input data
    data = hydros.iloc[:, 1:len(hydros.columns) - 1]

    # Makes  a list of 0 through the number of hours in each window
    time_hours = range(data.shape[1])

    # Calculates overall mean of all hydrographs by taking mean of all hydrograph means
    overall_mean = np.mean(np.mean(data))

    # Makes input data an array so that it can be iterated through to perform calcs on each hydrograph
    som_input = data.to_numpy()

    # Finds the value of the largest peak and the number of peaks in each hydrograph
    peaks = []
    intersections = []

    # iterates through every hydrograph in the input data
    for timeseries in som_input:
        # Calls the peakVal function
        peak = calculate_peak_value(timeseries)

        # Calculate the individual hydrograph's mean
        timeseries_mean = np.mean(timeseries)

        # Scales the peak value by the hydrograph mean and the overall mean of all hydrographs and adds to a list of peaks
        scaled_peak = peak / timeseries_mean * overall_mean
        peaks.append(scaled_peak)

        # Adds number of intersections with a percentile line (peaks) to a list by calling numPeaks function
        intersections.append(count_number_of_peaks(timeseries))

    # Finds the maximum number of peaks out of all hydrographs
    max_peaks = max(intersections)

    # Scales the numbers of peaks list by the maximum number of peaks and overall hydrograph mean
    scaled_intersects = intersections / max_peaks * overall_mean

    # Adds columns containing number of peaks and largest peak values (one value for each parameter for each hydrograph) to the input data
    data["Number of Peaks"] = scaled_intersects
    data["Peaks"] = peaks
    som_input = data.to_numpy()

    ### Creates and trains a 25 x 25 som on the input data array ###
    # Creates and trains a 25 x 25 som on the input data array. Sets a random seed so that results are reproducable.
    som = MiniSom(25, 25, som_input.shape[1], random_seed=1)

    # Uses 2500 training iterations, which was determined to be sufficient for reducing difference between iterations
    som.train(som_input, 2500)

    # win_map shows which hydrographs from input data are contained in each bin of the trained SOM
    win_map = som.win_map(som_input, return_indices = True)

    # Makes a copy of win_map to reference later after changing cell names from original win_map
    win_map_copy = copy.copy(win_map)
    print("Number of SOM cells used:", len(win_map))

    # After fitting the SOM, drops the peaks and intersections from the data so that the hydrograph plots aren't skewed
    data = data.drop("Peaks", axis=1)
    data = data.drop("Number of Peaks", axis=1)
    som_input = data.to_numpy()

    # Preserves current key names in a list
    som_cell_indices = list(win_map.keys())

    # Replaces every old key name with numbers from 0 to the number of SOM cells used
    for i in range(len(win_map)):
        old = som_cell_indices[i]
        win_map[i] = win_map.pop(old)

    # Makes a list of entries from the win_map dictionary
    som_assigned_timeseries = list(win_map.items())

    # Sorts the list of entries by how many hydroraphs are contained in each bin, from highest to lowest number
    som_assigned_timeseries.sort(key=lambda x: len(x[1]), reverse=True)

    ### Plotting hydrographs from each SOM cell ###
    # Makes plots of the all hydrographs contained in each bin if user has set to generate plots
    if plots:
        plot_cells(results_path, som_assigned_timeseries, som_input, time_hours, som_cell_indices)

    ### Plots hydrographs from each bin along with the overall weight vector of the bin ###
    # Gets weights and distances for each cell in the SOM
    weights = som.get_weights()
    distances = som.distance_map()

    # Makes lists to be filled with volumes, relevant distance (distances from cells that contain hydrographs), intersections and largest peak outputs for each cell
    volumes, relevant_distances, peaks, intersections = calculate_cell_values(results_path, som_assigned_timeseries, som_input, som_cell_indices, weights, time_hours, distances,
                                                                              plots)

    ### Mean-Shift Clustering ###
    # Perform the mean shift clustering
    label, volumes_scaled, flat_distances_scaled = calculate_mean_shift_clustering(volumes, relevant_distances, peaks, intersections, correction)

    # Finds the number of clusters
    unique_labels = np.unique(label)
    print("Number of Clusters: ", len(unique_labels))

    # Adds coordinates for SOM cells containing hydrographs to the dataframe
    cellList = []
    for cell, value in som_assigned_timeseries:
        cellList.append(som_cell_indices[cell])

    # Makes a data frame with the input distance data, the cluster assigned to each data point, and the cell index for each point
    df = pd.DataFrame(flat_distances_scaled)
    df["Volume"] = volumes_scaled
    df["Cluster"] = label
    df["Cell Num"] = range(0,len(relevant_distances.flatten()))
    df["Cell Index"] = cellList

    ############################################# Plotting the clusters and generating results ###################################################
    output_summary_spreadsheet(time_hours, plots, distributions_path, unique_labels, df, win_map_copy, weights, number_of_window_days, sample_freq, som_input, min_y, max_y,
                               clusters_path, fixed_clusters_path, metrics_path)

    ### Print that the analysis is complete ###
    print("Done")