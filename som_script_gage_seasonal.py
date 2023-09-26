import multiprocessing
from itertools import repeat
import numpy as np
import pandas as pd
import os, math, copy
from minisom import MiniSom
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift, AgglomerativeClustering, estimate_bandwidth
from sklearn.neighbors import kneighbors_graph
from som_functions import *
from datetime import date, datetime, timedelta


def _create_subgroup_plot(som_input, time_hours, timeseries_subgroups, output_path, type, units, entry_group):
    """
    Creates a subgroup plot partitioned out from the main clusters. This allows the plots to be generated in parallel

    Parameters
    ----------
    som_input: ndarray
        Input to the SOM classification
    time_hours: ndarray
        Time vector to plot the timeseriess
    timeseries_subgroups: ndarray
        Gives the subgroup label for each timeseries
    output_path: str
        Output path
    type: str
        Type of the variable in the timeseries
    units: str
        Units of the variable in the timeseries
    entry_group: int
        Number of the subgroup being classified

    Returns
    -------
    None. Plots are written to the disk

    """

    da_group = som_input[np.argwhere(timeseries_subgroups == entry_group).flatten(), :]
    for entry_series in range(0, da_group.shape[0], 1):
        plt.plot(np.array(time_hours), da_group[entry_series, :])

    plt.xlabel('Time (hours)')
    plt.ylabel(type.capitalize() + " (" + units + ")")
    plt.grid(which='both', alpha=0.7, linestyle='--')
    plt.savefig(os.path.join(output_path, 'subgroup_' + str(entry_group) + '.png'), dpi=600)
    plt.close()


if __name__ == "__main__":

    ### User inputs ###
    # First line of CSV should be column names. First 2 columns of data should be datetime and streamflow values in ft^3/s.
    input_filename = r"16-0549_HourlyPrecip.csv"
    input_type = 'precipitation'
    input_units = "in"

    # Specify the number of units(days) in each sliding window
    number_of_window_days = 3

    # Specify the number of samples taken each days
    sample_freq = 24

    # If user wants to generate metric cluster plots and SOM cell timeseries plots, set plots to True below. Generating plots increases runtime
    plots = False

    # A correction factor for clustering bandwidth - user may need to tweak
    shape_correction = 1.25

    # The y-axis limit for fixed range cluster plots. User will most likely need to adjust based on dataset if fixed range cluster plots are desired[
    min_y = 0
    max_y = 7000

    # Indicates if zero valued series should be removed
    remove_zeros = True

    ################################################################################################################################################################################
    ### Open a processing pool ###
    compute_pool = multiprocessing.Pool()

    ### Preprocessing ###
    # Read the input data. This may need to be adjusted depending on the format of your data
    input_data = pd.read_csv(input_filename, index_col=[0], parse_dates=True)

    # If the dataset has negative values, those are changed to NaN's
    input_data[input_data < 0] = np.NAN

    # Check the daterange of the data
    oa_date_range = pd.date_range(input_data.index[0], input_data.index[-1], freq=str(sample_freq) + 'H')
    assert input_data.shape[0] == oa_date_range.shape[0], "There are missing timesteps in the sample file. Please correct and rerun."

    # Changes sampling frequency if it isn't already hourly
    input_data = resample_timeseries(input_data, sample_freq)
    sample_freq = 24

    # User must enter desired number of days per window as well as sampling frequency of data above
    window_hours = number_of_window_days * sample_freq

    # Makes the data column into an array to be used with the sliding windows function below
    stream_flows = input_data.values.flatten()

    # Divides data into sliding windows, each containing the desired number of days per window
    stream_flows = np.lib.stride_tricks.sliding_window_view(stream_flows, window_hours)[::sample_freq, :]

    # Creates column names for a correctly sized dataframe
    columns = []
    for i in range(number_of_window_days * sample_freq):
        # Column names are 0 to n
        columns.append(i)

    # Puts streamflow data into the data frame using column names above
    working_data = pd.DataFrame(stream_flows, columns=columns)

    # Makes two arrays from the datetime objects and doesn't include the time variable in one so that the unique function can be used below
    date_times = np.asarray(input_data.index)

    # Sorts through dates and adds unique dates to the start dates array
    unique_dates, indices = np.unique(input_data.index.date, return_index=True)
    start_dates = date_times[indices]

    # Adjusts list of start dates/times to account for the 6 days in the last 7-day window that will never be start dates
    # This makes the start_datetimes list have the same number of entries as the number of windows from the data
    difference = len(start_dates) - len(stream_flows)
    start_dates = np.delete(start_dates, slice(-difference, len(start_dates)))

    # Adds a column of start dates/times to the dataset
    working_data['Date'] = start_dates
    
    # Removes rows with NaNs
    working_data.drop(np.unique(np.argwhere(np.isnan(working_data).values[:, :-1])), inplace=True)

    # Remove rows with all zeros
    if remove_zeros:
        # Calculate a list of zero entries
        zero_indices = np.array([np.all(working_data.iloc[x, :-1] == 0) for x in range(0, working_data.shape[0], 1)]).astype(bool)

        # Remove the entries from the dataset
        working_data = working_data.iloc[~zero_indices, :]

    # Update start dates after the row drop
    start_dates = working_data['Date'].values

    print("Number of timeseries:", len(working_data))

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

    ### Calculating Largest Peak Value and Number of Peaks per timeseries Prior to the SOM ###
    # allows the script file to be restarted from this cell after adjustments are made instead of re-reading all input data
    som_input = working_data.iloc[:, 0:len(working_data.columns) - 1]

    # Makes a list of 0 through the number of hours in each window
    time_hours = range(som_input.shape[1])

    # Calculates overall mean of all timeseries by taking mean of all timeseries means
    overall_mean = np.mean(np.mean(som_input))

    # Makes input data an array so that it can be iterated through to perform calcs on each timeseries
    som_input = som_input.to_numpy()

    # Finds the value of the largest peak and the number of peaks in each timeseries
    peaks = []
    intersections = []

    # iterates through every timeseries in the input data
    for timeseries in som_input:
        # Calls the peakVal function
        peak = calculate_peak_value(timeseries)

        # Calculate the individual timeseries's mean
        timeseries_mean = np.mean(timeseries)

        if timeseries_mean == 0:
            peaks.append(0)

        else:
            # Scales the peak value by the timeseries mean and the overall mean of all timeseries and adds to a list of peaks
            scaled_peak = peak / timeseries_mean * overall_mean
            peaks.append(scaled_peak)

        # Adds number of intersections with a percentile line (peaks) to a list by calling numPeaks function
        intersections.append(count_number_of_peaks(timeseries))

    # Finds the maximum number of peaks out of all timeseries
    max_peaks = max(intersections)

    # Scales the numbers of peaks list by the maximum number of peaks and overall timeseries mean
    scaled_intersects = intersections / max_peaks * overall_mean

    # Adds columns containing number of peaks and largest peak values (one value for each parameter for each timeseries) to the input data
    working_data["Number of Peaks"] = scaled_intersects
    working_data["Peaks"] = peaks

    # Timestamp is not included. Remove the column
    working_data.drop('Date', axis=1, inplace=True)

    # Convert the datagrame to an array
    som_input = working_data.to_numpy()

    ### Creates and trains a 25 x 25 som on the input data array ###
    # Creates and trains a 25 x 25 som on the input data array. Sets a random seed so that results are reproducable.
    som = MiniSom(25, 25, som_input.shape[1], random_seed=1)

    # Uses 2500 training iterations, which was determined to be sufficient for reducing difference between iterations
    som.train(som_input, 5000)

    # win_map shows which timeseriess from input data are contained in each bin of the trained SOM
    win_map = som.win_map(som_input, return_indices=True)

    # Makes a copy of win_map to reference later after changing cell names from original win_map
    win_map_copy = copy.copy(win_map)
    print("Number of SOM cells used:", len(win_map))

    # After fitting the SOM, drops the peaks and intersections from the data so that the timeseries plots aren't skewed
    working_data.drop("Peaks", axis=1, inplace=True)
    working_data.drop("Number of Peaks", axis=1, inplace=True)
    som_input = working_data.to_numpy()

    # Preserves current key names in a list
    som_cell_indices = list(win_map.keys())

    # Replaces every old key name with numbers from 0 to the number of SOM cells used
    for i in range(len(win_map)):
        old = som_cell_indices[i]
        win_map[i] = win_map.pop(old)

    # Makes a list of entries from the win_map dictionary
    som_assigned_timeseries = list(win_map.items())

    # Sorts the list of entries by how many timeseries are contained in each bin, from highest to lowest number
    som_assigned_timeseries.sort(key=lambda x: len(x[1]), reverse=True)

    ### Plotting timeseriess from each SOM cell ###
    # Makes plots of the all timeseriess contained in each bin if user has set to generate plots
    if plots:
        plot_cells(results_path, som_assigned_timeseries, som_input, time_hours, som_cell_indices, input_type, input_units)
    
    ### Plots timeseriess from each bin along with the overall weight vector of the bin ###
    # Gets weights and distances for each cell in the SOM
    weights = som.get_weights()
    distances = som.distance_map()

    weights = weights[:, :, :-2]

    # Makes lists to be filled with volumes, relevant distance (distances from cells that contain timeseriess), intersections and largest peak outputs for each cell
    volumes, relevant_distances, peaks, intersections = calculate_cell_values(results_path, som_assigned_timeseries, som_input, som_cell_indices, weights, time_hours, distances,
                                                                              plots)
 
    ### Mean-Shift Clustering ###
    # Perform the mean shift clustering
    label, volumes_scaled, flat_distances_scaled = calculate_mean_shift_clustering(volumes, relevant_distances, peaks, intersections, shape_correction)

    # Finds the number of clusters
    unique_labels = np.unique(label)
    print("Number of Clusters: ", len(unique_labels))

    # Adds coordinates for SOM cells containing timeseriess to the dataframe
    cellList = []
    for cell, value in som_assigned_timeseries:
        cellList.append(som_cell_indices[cell])

    ### Subcluster based on the dates hydrographs for each cluster ###
    # Define an array to hold the subgroup number
    timeseries_subgroups = np.zeros(len(start_dates))

    # Define a counter to keep track of the subgroups
    offset_counter = 0

    # Loop and subpartition each cluster
    for entry_cluster in range(0, unique_labels.shape[0], 1):
        # Pring the cluster to know the workflow is alive
        print(entry_cluster)

        # Get the SOM cells assigned to the cluster
        assigned_cells = np.argwhere(label == entry_cluster).flatten()

        # Get timeseries within the SOM cells
        clustered_timeseries = np.hstack([som_assigned_timeseries[x][1] for x in assigned_cells])

        # Get the start dates associated with the timeseries
        timeseries_starts = pd.to_datetime([start_dates[clustered_timeseries[x]] for x in range(0, len(clustered_timeseries), 1)])

        # Find the leap year indices and adjust day to 28 to eliminate leap year errors
        ia_indices = np.logical_and(timeseries_starts.day == 29, timeseries_starts.month == 2)
        ia_indices = np.argwhere(ia_indices).flatten()

        # Move leap year dates to February 28th to prevent math issues in the subsequent time distancing operation
        timeseries_starts = timeseries_starts.to_pydatetime()
        for entry_date in range(0, ia_indices.shape[0], 1):
            timeseries_starts[ia_indices[entry_date]] = datetime(year=timeseries_starts[ia_indices[entry_date]].year, month=2, day=28,
                                                                 hour=timeseries_starts[ia_indices[entry_date]].hour, minute=timeseries_starts[ia_indices[entry_date]].minute)

        # Calculate the Julian date
        timeseries_starts_julian = pd.to_datetime(timeseries_starts).dayofyear

        # Setup the cyclic basis
        timeseries_starts_cyclic = np.vstack((np.sin(2 * np.pi * timeseries_starts_julian / 365), np.cos(2 * np.pi * timeseries_starts_julian / 365)))
        timeseries_starts_cyclic = np.transpose(timeseries_starts_cyclic)

        # Perform the clustering
        if timeseries_starts_cyclic.shape[0] > 1:
            # Sufficient samples exist. Create the connectivity graph
            if timeseries_starts_cyclic.shape[0] < 21:
                knn_graph = kneighbors_graph(timeseries_starts_cyclic, timeseries_starts_cyclic.shape[0] - 1, include_self=False, metric='manhattan')
            else:
                knn_graph = kneighbors_graph(timeseries_starts_cyclic, 20, include_self=False, metric='manhattan')

            # Create the labels
            clustering = AgglomerativeClustering(distance_threshold=2 * np.pi / 4, n_clusters=None,  metric='manhattan', linkage='average', connectivity=knn_graph)
            sublabel = clustering.fit_predict(timeseries_starts_cyclic)

        else:
            # Only one sample exists. Assign it to its own cluster
            sublabel = np.array([0])

        # Assign the sublabel to the hydrographs
        timeseries_subgroups[clustered_timeseries] = sublabel + offset_counter

        # Increment the offset to produce a unigue group number for the next set of hydrographs
        offset_counter += np.max(sublabel) + 1


    ### Output the secondary clustering information ###
    # Output plots
    if plots:
        subgroup_path = os.path.join(results_path, "subgroups")
        if not os.path.exists(subgroup_path):
            os.makedirs(subgroup_path)

        list(compute_pool.starmap(_create_subgroup_plot, zip(repeat(som_input, offset_counter), repeat(time_hours, offset_counter), repeat(timeseries_subgroups, offset_counter),
                                                             repeat(subgroup_path, offset_counter), repeat(input_type, offset_counter), repeat(input_units, offset_counter),
                                                             np.arange(0, offset_counter, 1))))

    ### Output the primary cluster information ###
    # Makes a data frame with the input distance data, the cluster assigned to each data point, and the cell index for each point
    output_data = pd.DataFrame(flat_distances_scaled)
    output_data["Volume"] = volumes_scaled
    output_data["Cluster"] = label
    output_data["Cell Num"] = range(0, len(relevant_distances.flatten()))
    output_data["Cell Index"] = cellList

    ### Plotting the clusters and generating results ###
    output_summary_spreadsheet(time_hours, plots, distributions_path, unique_labels, output_data, win_map_copy, weights, number_of_window_days, sample_freq, som_input,
                               min_y, max_y, clusters_path, fixed_clusters_path, metrics_path, results_path, input_type, input_units, start_dates, timeseries_subgroups)

    ### Print that the analysis is complete ###
    # Close the compute pool
    compute_pool.close()
    print("Done")
