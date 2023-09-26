import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import Pool

import sklearn
from sklearn.cluster import MeanShift


def resample_timeseries(data, sample_freq):
    """
    Converts data that is not on an hourly timestep to an hourly timestep

    Parameters
    ----------
    data: DataFrame
        Input data to be resampled
    sample_freq: int
        Number of daily samples

    Returns
    -------

    """

    if sample_freq > 24:
        # Convert from subhourly data to hourly data
        resampled_data = data.resample('H').mean()

    else:
        # Convert from above hourly data to hourly. This assumes a forward fill of the data.
        resampled_data = data.resample('H').ffill()

    # Return to the calling function
    return resampled_data


def calculate_peak_value(timeseries):
    """
    Finds and returns the value of the largest peak (defined as the value with the greatest absolute difference from the mean of the array)
    in the input array
    
    Parameters
    ----------
    timeseries : np.array
        An array containing timeseries values
    
    Returns
    -------
    peak : float
        The greatest difference between any value in the array and the mean of the array
    
    """

    # Sets greatest peak value at zero to start
    peak = 0

    # Calculates each timeseries mean
    h_mean = np.mean(timeseries, axis = 0)

    # Loop over the timeseries values looking for the largest peak value
    for entry in timeseries:
        # Calculates difference between every value in the timeseries and the individual timeseries mean
        diff = abs(entry - h_mean)

        # If the difference is greater than the current peak value, then the largest peak value is set as the difference
        if diff > peak:
            peak = diff    

    # Return to the calling function
    return peak


def count_number_of_peaks(timeseries, percentile=85):
    """
    Returns the number of peaks in an array by calculating how many times the array intersects with a percentile line
    
    Parameters
    ----------
    timeseries : np.array
        An array containing timeseries values 
    percentile : int, optional
        Sets the percentile to use when calculating the number of peaks, default is the 85th percentile
    
    Returns
    -------
    count : int
        The number of times that values in the input array intersect with the set percentile (used as number of peaks)
    
    """
    
    # Calculates the 85th percentile of each timeseries
    # A peak is defined as being greater than the 85th percentile of the timeseries
    percentile_val = np.percentile(timeseries, percentile)
    
    # Makes a list filled with the percentile value so it can be compared to all values in the timeseries
    percentile_line = np.empty(len(timeseries))
    percentile_line.fill(percentile_val)
    
    # Calculates how many times each timeseries intersects with the percentile line
    diff = timeseries - percentile_line
    
    # Sets a condition that is true when the difference between timeseries value and percentile line is greater than 0
    positive = diff > 0
    
    # Keeps track of when the difference goes from being positive to negative, which indicates an intersection
    count = np.logical_xor(positive[1:], positive[:-1]).sum()

    # Return to the calling function
    return count


def plot_cells(results_path, som_assigned_timeseries, som_input, time_hours, cell_indices, type='streamflow', units="$ft^3$/s"):
    """
    Plots the timeseries assigned to the cell without the trained weight vector

    Parameters
    ----------
    results_path: str
        Path to the output results folder
    som_assigned_timeseries: list
        Timeseries assigned to the cell
    som_input: ndarray
        Contains the input data used to train the SOM. Each row is an additional timeseries
    time_hours: object
        Iterable for the duration of the timeseries
    cell_indices: list
       Contains the cell indices to reference the location of the cell in the SOM grid
    type: str
        Type of the variable in the timeseries
    units: str
        Units of the variable in the timeseries

    Returns
    -------
    None. Plots of the cell information are written to disk.

    """

    # Creates a folder for plots of all timeseries in each SOM cell without SOM weights shown
    hydro_plots = os.path.join(results_path, 'SOM_Cell_Plots/')

    # If folder for timeseries plots doesn't exist, creates a folder
    if not os.path.isdir(hydro_plots):
        os.makedirs(hydro_plots)

    # Iterates through the dictionary contianing each SOM cell and the associated timeseries indices
    for cell, value in som_assigned_timeseries:

        # For each timeseries number, the corresponding row/timeseries from the input data is referenced
        hydroData = som_input[value]

        # Plots the timeseries as a function of time
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = "Times New Roman"
        plt.plot(time_hours, hydroData.T)
        plt.xlabel("Hours in Window")
        plt.ylabel(type.capitalize() + " (" + units + ")")
        plt.xlim(left=0)  # Starts the x axis values at 0
        plt.grid()

        # Names each plot after the SOM cell it represents
        plt.title(cell_indices[cell])

        # Saves each plot as a png image in the Som_Cell_Plots folder
        file_name = str(cell_indices[cell]) + ".png"
        plt.savefig(hydro_plots + file_name, dpi=300)
        plt.close()


def plot_cells_with_weight(output_path, time_hours, som_cell_data, weight, som_cell_index, cell, type='streamflow', units="$ft^3$/s"):
    """
    Plots the timeseries assigned to the cell with the trained weight vector

    Parameters
    ----------
    output_path: str
        Folder into which teh files are saved
    time_hours: object
        Iterable for the duration of the timeseries
    som_cell_data: ndarray
        Contains the timeseries assigned to the SOM cell
    weight: ndarray
        Weight vector of the SOM cell
    som_cell_index: list
        Index of the cell being plotted
    cell: int
        Number of the cell being plotted
    type: str
        Type of the variable in the timeseries
    units: str
        Units of the variable in the timeseries

    Returns
    -------
    None. File is written to the disk

    """

    # If user is generating plots and plot folder does not exist, creates a folder to hold plots
    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    # Plots the timeseries as a function of time from the beginning of the window
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = "Times New Roman"
    plt.plot(time_hours, som_cell_data.T)
    plt.plot(time_hours, weight.T, linestyle="", marker="o")

    # Uncomment line below to displays the percentile line from the cell weight as a black dashed line
    plt.xlabel("Hours in Window")
    plt.ylabel(type.capitalize() + " (" + units + ")")
    plt.xlim(left=0)  # Starts the x axis values at 0

    # Names each plot after the bin it represents
    plt.title(som_cell_index[cell])
    plt.grid()
    fileName = str(som_cell_index[cell]) + ".png"
    plt.savefig(output_path + fileName, dpi=300)
    plt.close()


def calculate_cell_values(results_path, som_assigned_timeseries, som_input, som_cell_indices, weights, time_hours, som_distances, plots, type='streamflow', units="$ft^3$/s"):
    """
    Calculates summary values from the SOM cells to use as inputs to later classification steps

    Parameters
    ----------
    results_path: str
        Path to the output folder
    som_assigned_timeseries: list
        Timeseries assigned to the cell
    som_input: ndarray
        Array containing the SOM input data
    som_cell_indices: list
        Indices corresponding to each SOM cell
    weights: narray
        Trained eight that represents each SOM cell
    time_hours: iterable
        Length of the duration of the timeseries, in hours
    som_distances: ndarray
        Distance map from the SOM training
    plots: bool
        Indicates if plots should be created
    type: str
        Type of the variable in the timeseries
    units: str
        Units of the variable in the timeseries

    Returns
    -------
    volumes: ndarray
        Area under the timeseries
    relevant_distances: ndarray
        Distances of cells that contain timeseries
    peaks: ndarray
        Number of peaks in the weight vector
    intersections: ndarray
        Number of intersections in the weight vector

    """

    volumes = []
    relevant_distances = []
    peaks = []
    intersections = []

    # Creates a pat for plots of all timeseries in each SOM cell with SOM weights shown
    weight_plots = os.path.join(results_path, 'SOM_Cell_Plots_Weight/')

    for cell, value in som_assigned_timeseries:
        # For each timeseries number, the corresponding number window from the input data is plotted
        hydroData = som_input[value]

        # The cell number is read in as coordinates in the weight vector, and the weight vector is referenced accordingly
        m, n = som_cell_indices[cell]

        # The last two elements of the weight vector are for the number of interesections and peak value parameters, and are not plotted
        weight = weights[m, n, :]

        # Calculates the volume under the weight curve for each SOM cell and adds it to a list
        volumes.append(np.trapz(weight, time_hours))

        # Makes a list of distances for cells that have associated timeseries (instead of including distances for all cells on the map)
        relevant_distances.append(som_distances[m][n])

        # Calls the peakVal function for each weight
        peaks.append(calculate_peak_value(weight))

        # Adds number of intersections with a percentile line (peaks) for each weight to a list by calling numPeaks function
        intersections.append(count_number_of_peaks(weight))

        if plots:
            plot_cells_with_weight(weight_plots, time_hours, hydroData, weight, som_cell_indices, cell, type, units)

    # Promote the lists to an array
    relevant_distances = np.array(relevant_distances)
    volumes = np.array(volumes)
    peaks = np.array(peaks)
    intersections = np.array(intersections)

    # Return to the calling function
    return volumes, relevant_distances, peaks, intersections


def calculate_mean_shift_clustering(relevant_distances, volumes, peaks, intersections, correction):
    """
    Performs mean shift clustering on the SOM output

    Parameters
    ----------
    volumes: ndarray
        Area under the timeseries
    relevant_distances: ndarray
        Distances of cells that contain timeseries
    peaks: ndarray
        Number of peaks in the weight vector
    intersections: ndarray
        Number of intersections in the weight vector
    correction: float
        Correction factor for the mean shift bandwidth

    Returns
    -------
    label: list
        Cluster assignment for each SOM cell
    volumes_scaled: ndarray
        Scaled volume vector used for mean shift clustering
    flat_distances_scaled:
        Scaled distance vector used for mean shift clustering

    """
    # todo: add more detail about the workflow

    # Stacks rows of data so that they are all in one array instead of array of arrays
    flat = relevant_distances.flatten()
    flat_volumes = volumes.flatten()
    flat_peaks = peaks.flatten()
    flat_intersections = intersections.flatten()

    # Reshapes the arrays so that they are in the correct form for the fit_predict function
    flat_distances = flat.reshape(len(relevant_distances),1)
    flat_volumes = flat_volumes.reshape(len(volumes),1)
    flat_peaks = flat_peaks.reshape(len(peaks), 1)
    flat_intersections = flat_intersections.reshape(len(intersections), 1)

    scaler = sklearn.preprocessing.MinMaxScaler()
    volumes_scaled = scaler.fit_transform(flat_volumes)
    distances_scaled = scaler.fit_transform(flat_distances)
    peaks_scaled = scaler.fit_transform(flat_peaks)
    intersect_scaled = scaler.fit_transform(flat_intersections)

    # Re-flattens the scaled volumes so that they can be combined with distances in an array
    flat_volumes_scaled = volumes_scaled.flatten()
    flat_distances_scaled = distances_scaled.flatten()
    flat_peaks_scaled = peaks_scaled.flatten()
    flat_intersect_scaled = intersect_scaled.flatten()

    # Makes a 5D input for the clustering model (distance, volume, peak value, number of peaks)
    dist_vol_peaks_intersect = np.array([flat_distances_scaled, flat_volumes_scaled, flat_peaks_scaled, flat_intersect_scaled])

    # Estimates the best bandwidth parameter to use in mean shift clustering model
    bw = sklearn.cluster.estimate_bandwidth(dist_vol_peaks_intersect.T)

    # Divides bandwidth by a correction factor to adjust number of clusters
    bw = bw / correction

    # Initializes a mean shift clustering model with the bandwidth from the previous cell
    clustering = MeanShift(bandwidth=bw)

    # Assigns each SOM cell to a cluster based on distance, volume, peak value, and number of peaks
    label = clustering.fit_predict(dist_vol_peaks_intersect.T)

    # Return to the calling function
    return label, volumes_scaled, flat_distances_scaled


def _extract_weight(c, weights):
    # todo: doc string
    m, n = c
    weight = weights[m, n, :]

    return weight


def output_summary_spreadsheet(time_hours, plots, dists, unique_labels, data, win_map_copy, weights, number_of_window_days, sample_freq, som_input, min_y, max_y,
                               clusters_path, fixed_cluster_path, metrics_path, results_path, type='streamflow', units="$ft^3$/s", start_dates=None, subgroup=None, ):
    """
    Creates a summary spreadsheet and plots of the combined SOM and mean shift clustering

    Parameters
    ----------
    time_hours: iterable
        Length of the duration of the timeseries, in hours
    plots: bool
        Indices if plots should be created
    dists: str
        Path to the distribution folder
    unique_labels: ndarray
        Contains a number assigned to each unique cluster
    data: DataFrame
        Contains the input data
    win_map_copy: list
        Contains the cells assigned to each SOM cell
    weights: ndarray
        Contains the weight vector of each SOM cell
    number_of_window_days: int
        Number of days in the timeseries window
    sample_freq: int
        Number of samples per day
    som_input: ndarray
        Contains the input data used to train the SOM. Each row is an additional timeseries
    min_y: float
        Minimum of the y axis range for generating fixed axis plots
    max_y: float
        Maximum of the y axis range for generating fixed axis plots
    clusters_path: str
        Path to the dynamic y range cluster plots
    fixed_cluster_path: str
        Path to the fixed y range cluster plots
    metrics_path: str
        Path to the metric plots
    results_path: str
        Path to the top level results folder
    type: str
        Type of the variable in the timeseries
    units: str
        Units of the variable in the timeseries
    start_dates: ndarray
        Date of each vector
    subgroup: narray
        List of subgroups to which each timeseriesis assigned

    Returns
    -------
    None. Plots and spreadsheet are written to disk.

    """
    # todo: break items out into additional functions. Each of the loops can be parallelized

    # Dictates which excel file the results will be written to
    writer = pd.ExcelWriter(os.path.join(results_path, 'results.xlsx'))

    # Creates lists to be filled with a table for each cluster
    key_dfs = []
    som_cluster_dfs = []
    hydro_cluster_dfs = []

    # Stand up a compute pool
    # compute_pool = Pool()

    # iterates through each cluster number
    for i in unique_labels:
        print("Working on cluster ", i)

        # Finds the indices in the dataframe with one cluster label at a time
        indices = data.index[data['Cluster'] == i].tolist()

        # Keeps track of which cells  (referenced by a tuple) are in each cluster
        areas = []
        min_diff = []
        max_diff = []
        ave_slopes = []

        # Goes through the bin numbers in each cluster and adds all timeseries from each bin to a list
        cell_index = [c for c in data["Cell Index"][indices] if c in win_map_copy.keys()]
        cluster_hydros = [win_map_copy.get(c) for c in cell_index]
        weightList = [_extract_weight(c, weights) for c in cell_index]
        areaList = [np.trapz(x, time_hours) for x in weightList]

        # Stops current iteration and starts next cluster if list of weights is empty
        if len(weightList) == 0:
            continue

        # Converts list to a dataframe with every cell as one row so that an average weight row can be calculated
        weightdf = pd.DataFrame(weightList)

        # Adds the average weight row to the end of the dataframe
        weightdf.loc[len(weightdf.index)] = weightdf.mean(axis=0)

        # Pulls the average weight row from the dataframe and converts to an array so that it can be transposed
        weightarray = np.array(weightdf.loc[len(weightdf.index) - 1])

        # Calculates the area under the cluster's average weight curve
        ave_area = np.trapz(weightarray, time_hours)
        areas.append(areaList)

        ######################################### Calculates metrics for each cluster ############################################
        # Generates column names for a correctly sized dataframe to hold cluster metrics
        columns = np.arange(0, number_of_window_days * sample_freq, 1)

        # Initializes a dataframe to hold info about SOM weights from each cluster
        som_cluster_df = pd.DataFrame(columns=columns)

        # Iterates through every cell in the cluster and calculates metrics based on weight
        j = 0
        for w in weightList:
            # Calculates cluster metrics to be output in a spreadsheet for each cluster
            # Difference between cell weight and average weight
            diff = w - weightarray

            # Maximum difference between each cell weight and ave weight
            max_diff.append(max(abs(diff)))

            # Minimum difference between each cell weight and ave weight
            min_diff.append(min(abs(diff)))

            # Calculates the best fit slope for each weight curve
            m, b = np.polyfit(time_hours, w, 1)

            # Adds the slope to a list of slopes with one slope for each weight curve
            ave_slopes.append(m)
            m_ave, b_ave = np.polyfit(time_hours, weightarray, 1)

            # Fills the weight vector for each SOM in the cluster into a dataframe
            som_cluster_df.loc[j] = w
            j += 1

        # Gets rid of timeseries that contain missing values
        clean = [ele for ele in cluster_hydros if ele != None]

        # Keeps track of the number of timeseries in each SOM cell
        num_timeseries = [len(value) for value in clean]

        ##################################### Plots timeseries and weight for each cluster ######################################
        # Initializes a list to contain timeseriesdata for the cluster
        full_hydros = []
        cell_mean = []

        # Iterates through the list of timeseries and pulls data for each one to plot it
        for value in clean:
            timeseries = som_input[value]

            # Adds each timeseriest o a list
            full_hydros.append(timeseries)

            # calculates mean of each SOM cell
            cell_mean.append(np.mean(timeseries))

            # Plots the timeseries as a function of time from the beginning of the window
            plt.rcParams["font.family"] = "serif"
            plt.rcParams["font.serif"] = "Times New Roman"
            plt.plot(time_hours, timeseries.T)
            plt.plot(time_hours, weightarray.T, linewidth=5)

            # plt.plot(time_hours, weightarray.T, linestyle = "", marker = "o")
            plt.xlabel("Hours in Window")
            plt.ylabel(type.capitalize() + " (" + units + ")")
            plt.xlim(left=0)  # Starts the x axis values at 0

            # Names each plot after the bin it represents
            plt.title(i)

            # Saves plots to pngs in the home directory
            fileName = str(i) + ".png"
            plt.savefig(clusters_path + fileName, dpi=300)

        plt.close()

        # Makes fixed range plots for the example
        for value in clean:
            plt.rcParams["font.family"] = "serif"
            plt.rcParams["font.serif"] = "Times New Roman"
            timeseries = som_input[value]

            # Plots the timeseries as a function of time from the beginning of the window
            plt.plot(time_hours, timeseries.T)
            plt.plot(time_hours, weightarray.T, linestyle="", marker="o")
            plt.xlabel("Hours in Window")
            plt.ylabel(type.capitalize() + " (" + units + ")")
            plt.xlim(left=0)  # Starts the x axis values at 0
            plt.ylim(bottom=min_y, top=max_y)
            # Names each plot after the bin it represents
            plt.title(i)

            # Saves plots to pngs in the home directory
            fileName = str(i) + " fixed.png"
            plt.savefig(fixed_cluster_path + fileName, dpi=300)

        plt.close()

        ################################# Calculates and plots distributions for each cluster #####################################
        # Creates lists to be filled with distributions for each cluster
        hydro_mean = []
        hydro_vol = []
        peak_vals = []
        intersections = []
        cell_intersects = []

        # Calculates overall mean for all timeseries in cluster
        overall_mean = np.mean(cell_mean)

        # Iterates through every cell in the cluster and every timeseries with each cell
        for cell in full_hydros:
            single_cell_intersects = []
            for h in cell:
                # Calculates timeseries mean
                hydro_mean.append(np.mean(h))

                # Calculates scaled timeseriesvolume
                vol = (np.trapz(h, time_hours))
                hydro_vol.append(vol / np.mean(h) * overall_mean)

                # Calculate the mean of the hydrograph
                timeseries_mean = np.mean(h)

                # Calls the peak val function to calculate the largest peak in each timeseries
                peak = calculate_peak_value(h)
                scaled_peak = peak / timeseries_mean * overall_mean
                peak_vals.append(scaled_peak)

                # Calls the numPeaks function to calculate the number of peaks in each timeseries
                intersections.append(count_number_of_peaks(h))
                single_cell_intersects.append(count_number_of_peaks(h))

            cell_intersects.append(sum(single_cell_intersects))

        # Determines the maximum number of peaks in any of the timeseries and uses it for scaling
        max_peaks = max(intersections)
        if max_peaks == 0:
            peakNum = 0

        else:
            # Scales the intersection count by the individual and overall timeseries means and adds to a list of intersections
            peakNum = intersections / max_peaks * overall_mean
            peakNum = list(peakNum)

        if plots == True:
            if not os.path.isdir(dists):
                os.makedirs(dists)

            # Outputs histograms for distribution parameters from every SOM Cell and saves to a file within the results folder
            # List of parameters to be plotted
            params = [peak_vals, peakNum, hydro_mean, hydro_vol]

            # List of labels for each histogram's x axis
            xlabs = ["Largest Peak Value", "Number of Peaks", "Individual Timeseries Means", "Timeseries Volume"]

            # List of names for each distributions image to be saved as
            pngNames = [" PeakVal.png", " NumPeaks.png", " Mean.png", " Volume.png"]
            n = 0

            # Iterates through every parameter and plots the parameter for every timeseries in the cluster as a histogram
            for distribution in params:
                plt.rcParams["font.family"] = "serif"
                plt.rcParams["font.serif"] = "Times New Roman"
                plt.hist(distribution, 300)
                plt.title("Cluster " + str(i))
                plt.xlabel(xlabs[n])
                plt.ylabel("Frequency")
                plt.xlim(left=0)  # Starts the x axis values at 0
                plt.grid()
                plt.savefig(dists + str(i) + pngNames[n], dpi=300)
                plt.close()
                n += 1

        ############################# Generates and formats result tables to be output ########################################
        # Generates a list of cluster labels according to current iteration number to be added to df
        cluster_labels = [i for k in range(len(areaList))]

        # Makes a df for each cluster containing the following metrics
        clusterDf = pd.DataFrame({"Cell Index": cell_index,
                                  "Cluster": cluster_labels,
                                  "Timeseries Volume": areaList,
                                  "Min Difference": min_diff,
                                  "Max Difference": max_diff,
                                  "Average Slope": ave_slopes,
                                  "Num. Intersections": cell_intersects})
        clusterDf.loc[len(weightdf.index)] = ["average", i, ave_area, 0, 0, m_ave, 0]

        # Writes each cluster to a csv file
        filename = str(i) + " metricData.csv"
        clusterDf.to_csv(metrics_path + filename)

        # Adds columns to the SOM Cluster df listing the cell index and number of timeseries contained in each cluster
        som_cluster_df.insert(0, "Cell Index", cell_index)
        som_cluster_df.insert(1, "Timeseries in cell", num_timeseries)
        som_cluster_dfs.append(som_cluster_df)

        # Makes a dataframe out of the timeseries from each SOM cell
        dfs = []
        for h in full_hydros:
            dfs.append(pd.DataFrame(h))

        # Concatenates all the SOM cell timeseries dataframes into one large dataframe of timeseries for the cluster
        hydro_cluster_df = pd.concat(dfs)

        # Makes a column of zeros in the timeseries df as a placeholder for timeseriesnumbers
        hydro_cluster_df.insert(0, "Timeseries Number", np.zeros(len(hydro_cluster_df)))
        if start_dates is not None:
            hydro_cluster_df.insert(1, 'Date', '')

        if subgroup is not None:
            hydro_cluster_df.insert(2, 'Subgroup', '')

        # Goes through the list of timeseries in each cell and adds them to the first column of the hydro_cluster df
        m = 0
        for value in clean:
            for n in value:
                # Set the timeseries number
                hydro_cluster_df.iloc[m, 0] = n

                # Set the timeseries date
                if start_dates is not None:
                    hydro_cluster_df.iloc[m, 1] = pd.to_datetime(start_dates[m]).strftime('%Y-%m-%d')

                if subgroup is not None:
                    hydro_cluster_df.iloc[m, 2] = subgroup[n]

                # Increment the counter
                m += 1

        # Adds the timeseries dataframe to a list of dataframes so that output can be written to an excel file at the end of the loop
        hydro_cluster_dfs.append(hydro_cluster_df)

        # Creates a list of cell indices that can be read next to the list of timeseries numbers to indicate which cell each timeseries is in
        df_cell_index = []
        df_cluster = []
        k = 0

        # Goes through the timeseries in each cell
        for n in clean:
            # Duplicates the cell index for every timeseries contained in the cell
            for a in range(len(n)):
                df_cell_index.append(cell_index[k])

                # Also makes a list containing cluster index that is the same length as the number of timeseries in the cluster
                df_cluster.append(i)

            k += 1

        # Adds timeseries number, cell index for each hydrograph, and cluster for each timeseries to a dataframe
        key_df = pd.DataFrame(columns=["Timeseries Number", "SOM Cell Index", "Cluster"])
        key_df["Timeseries Number"] = hydro_cluster_df["Timeseries Number"]

        if start_dates is not None:
            key_df["Date"] = hydro_cluster_df["Date"]

        key_df["SOM Cell Index"] = df_cell_index
        key_df["Cluster"] = df_cluster

        if subgroup is not None:
            key_df['Subcluster'] = hydro_cluster_df["Subgroup"]

        key_dfs.append(key_df)

    print("Writing results to excel file")

    # Concatenates dataframes from every cluster so that they are all in one table
    key_df = pd.concat(key_dfs)

    # Sorts the table by timeseries number so that timeseries are in order
    key_df = key_df.sort_values("Timeseries Number")

    # Writes the table to a sheet in excel
    key_df.to_excel(writer, sheet_name="Timeseries Key", index=False)

    # Writes the other dataframes to individual sheets for each cluster in the same excel file
    for number_of_window_days in range(len(som_cluster_dfs)):
        hydro_cluster_dfs[number_of_window_days].to_excel(writer, sheet_name=str(number_of_window_days) + " Timeseries", index=False)
        som_cluster_dfs[number_of_window_days].to_excel(writer, sheet_name=str(number_of_window_days) + " SOM Weights", index=False)

    writer.close()


