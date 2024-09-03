import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os, glob
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
from datetime import timedelta, datetime


def resample_timeseries(df_data, i_sample_freq):
    """
    Converts data that is not on an hourly timestep to an hourly timestep. This should only be used with continuous values. For any discontinuous
    values like, like precipitation, the series should be filled with NaNs or zeros and this should have no affect.

    Parameters
    ----------
    df_data: DataFrame
        Input data to be resampled
    i_sample_freq: int
        Number of daily samples

    Returns
    -------
    df_resampled_data: DataFrame
        Data resampled to hourly values

    """

    if i_sample_freq > 24:
        # Convert from subhourly data to hourly data
        df_resampled_data = df_data.resample('H').mean()

    else:
        # Convert from above hourly data to hourly. This assumes a forward fill of the data.
        df_resampled_data = df_data.resample('H').ffill()


    # Return to the calling function
    return df_resampled_data


def within_cluster_similarity(dm_timeseries, ia_labels):
    """
    Measures the similarity within a cluster as the mean RMSE difference between the series and the mean value of the cluster

    Parameters
    ----------
    dm_timeseries: ndarray
        Contains the clusters as the rows, columns as the vlaues
    ia_labels: ndarray
        Contains the labels associated with each of the clusters. Assumes clusters are labeled sequentially without any gaps.

    Returns
    -------
    da_similarity: ndarray
        Contains the distance metric for each clusters. Of shope (n,), where n is the number of clusters in the labels.

    """

    # Get the maximum number of clusters
    i_number_of_clusters = int(np.max(ia_labels))

    # Create a holder for the output
    da_similarity = np.zeros(i_number_of_clusters + 1)

    # Loop on each cluster
    for i_entry_cluster in range(0, i_number_of_clusters + 1, 1):
        # Get the series assigned to the cluster
        ia_cluster_indices = np.argwhere(ia_labels == i_entry_cluster).flatten()

        # Get the mean daily value across the timeseries
        da_daily_mean = np.mean(dm_timeseries[ia_cluster_indices, :], axis=0)

        # Calculate the squared difference
        da_squared_difference = dm_timeseries[ia_cluster_indices, :] - da_daily_mean[np.newaxis, :]
        da_squared_difference = da_squared_difference ** 2

        # Calculate the mean sum of the values
        d_mean_value = np.sum(da_squared_difference) / ia_cluster_indices.shape[0]

        # Set into the data holder
        da_similarity[i_entry_cluster] = d_mean_value

    # Return the similarity vector to the calling function
    return da_similarity


def between_cluster_similarity(dm_timeseries, ia_labels):
    """
    Measures the similarity between clusters as the sum of the squared difference in the means between clusters

    Parameters
    ----------
    dm_timeseries: ndarray
        Contains the clusters as the rows, columns as the vlaues
    ia_labels: ndarray
        Contains the labels associated with each of the clusters. Assumes clusters are labeled sequentially without any gaps.

    Returns
    -------
    dm_similarity: ndarray
        Contains the distance metric for each clusters. Of shope (n, n), where n is the number of clusters in the labels.

    """

    # Get the maximum number of clusters
    i_number_of_clusters = int(np.max(ia_labels))

    # Create a holder for the output
    dm_similarity = np.zeros((i_number_of_clusters + 1, i_number_of_clusters + 1))

    # Create the matrix for the centroids
    dm_centroids = np.zeros((i_number_of_clusters + 1, dm_timeseries.shape[1]))

    # Loop can calculate the centroid of each cluster
    for i_entry_cluster in range(0, i_number_of_clusters + 1, 1):
        # Get the series assigned to the cluster
        ia_cluster_indices = np.argwhere(ia_labels == i_entry_cluster).flatten()

        # Get the mean daily value across the timeseries
        dm_centroids[i_entry_cluster, :] = np.mean(dm_timeseries[ia_cluster_indices, :], axis=0)

    # Loop and calculate the distance between the centroids
    for i_entry_cluster in range(0, i_number_of_clusters + 1, 1):
        for i_entry_comparison in range(0, i_number_of_clusters + 1, 1):
            dm_similarity[i_entry_cluster, i_entry_comparison] = np.sum((dm_centroids[i_entry_cluster, :] - dm_centroids[i_entry_comparison, :]) ** 2)

    # Return the similarity matrix
    return dm_similarity


def construct_subcluster_counts(df_data, s_filename):
    """
    Plots the number of timeseries within each subcluster that start in each month

    Parameters
    ----------
    df_data: DataFrame
        Contains the dataframe with the timeseries assignments
    s_filename: str
        Name to which the plot is saved

    Returns
    -------
    None. Plot is written to disk.

    """
    
    ### Get the largest subcluster
    i_number_of_clusters = np.max(df_data['Subcluster']) + 1

    ### Get the month of each value ###
    ia_months = np.array([datetime.strptime(x, "%Y-%m-%d").month for x in df_data['Date'].values])

    ### Create the holder matrix ###
    im_counter = np.zeros((12, i_number_of_clusters))

    ### Loop and conduct the counts ###
    for i_entry_cluster in range(0, i_number_of_clusters, 1):
        for i_entry_month in range(1, 13, 1):

            # Get the indices for the subcluster and the date
            ba_valid_indices = np.logical_and(df_data['Subcluster'].values == i_entry_cluster, ia_months == i_entry_month)

            # Set the values into the matrix
            im_counter[i_entry_month - 1, i_entry_cluster] = np.sum(ba_valid_indices)


    ### Generate a plot for the subclusters ###
    o_figure, o_axis = plt.subplots(1, 1, figsize=(6, 5))
    o_raster = plt.imshow(np.flipud(im_counter), aspect='auto')

    ### Make the plot pretty ###
    # Create the labels
    plt.xlabel('Subcluster')
    plt.ylabel('Month')

    plt.yticks(np.arange(0, 12, 1) + 0.5,
               labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', "Jul", "Aug", 'Sep', 'Oct', 'Nov', 'Dec'])

    for label in o_axis.get_yticklabels(minor=True):
        label.set_verticalalignment('top')

    # Reverse the x axis
    o_axis.invert_yaxis()

    # Create the grid
    plt.grid(which='both', axis='y', color='w')

    # Create the colorbar
    plt.colorbar(o_raster, extend="max", location="top")

    # Adjust the margins
    o_figure.tight_layout()
    # o_figure.show()

    # Save the figure to the filename
    plt.savefig(s_filename + '.png', dpi=600)


if __name__ == "__main__":

    ### Define the input names ###
    # Keep data aligned between the lists, otherwise the wrong comparisons will be made
    # Define the som files to use in the comparison
    # List of paths to results file
    sl_som_files = ["results_seasonal_LTR_7day/results.xlsx", "results_seasonal_upstream_7day/results.xlsx"]

    # Define the manual files to use in the comparison
    sl_input_files = ["LTR_station.csv", "upstream.csv"]

    # Define the type of analysis
    s_type = 'streamflow'
    s_units = 'cfs'
    i_number_of_days_in_window = 7

    ##############################################################################################################################################################################
    ### Load the manual data ###
    l_manual_windows = []
    il_manual_labels = []

    for i_entry_input in range(0, len(sl_input_files), 1):
        # todo: this initial read needs to be modified based on the data being read in
        df_input_data = pd.read_csv(sl_input_files[i_entry_input], index_col=[0], parse_dates=True)

        # df_input_data.index = [df_input_data.index[x] + timedelta(hours=int(df_input_data['Hour'].iloc[x])) for x in range(0, df_input_data.shape[0], 1)]
        # df_input_data.drop('Hour', axis=1, inplace=True)

        # todo: this filling process needs to be adjusted based on the type of the input data and the dataset
        da_dates = pd.date_range(df_input_data.index[0], df_input_data.index[-1], freq='H')
        # df_input_data = df_input_data.reindex(da_dates, fill_value=0)
        # df_input_data = df_input_data.reindex(da_dates, fill_value=np.nan)

        # If the dataset has negative values, those are changed to NaN's
        df_input_data[df_input_data < 0] = np.nan

        # Resample the data
        df_input_data = resample_timeseries(df_input_data, 24)
        sample_freq = 24

        # User must enter desired number of days per window as well as sampling frequency of data above
        i_window_hours = i_number_of_days_in_window * 24

        # Divides data into sliding windows, each containing the desired number of days per window
        dm_timeseries = np.lib.stride_tricks.sliding_window_view(df_input_data.values.flatten(), i_window_hours)[::24, :]

        # Calculate a list of zero entries
        ba_zero_indices = np.array([np.all(dm_timeseries[x, :] == 0) or np.any(np.isnan(dm_timeseries[x, :])) for x in range(0, dm_timeseries.shape[0], 1)]).astype(bool)

        # Remove the entries from the dataset
        dm_timeseries = dm_timeseries[~ba_zero_indices, :]
        da_start_times = np.unique(df_input_data.index.date)[:ba_zero_indices.shape[0]][~ba_zero_indices]

        # Append into the array
        ia_labels = np.zeros(dm_timeseries.shape[0])
        ia_labels[np.logical_and(6 <= pd.DatetimeIndex(da_start_times).month, pd.DatetimeIndex(da_start_times).month <= 11)] = 1

        # Put into the holder arrays
        l_manual_windows.append(dm_timeseries)
        il_manual_labels.append(ia_labels)

    ### Load the SOM data ###
    fl_som_summary = []
    l_som_timeseries = []
    for i_entry_som in range(0, len(sl_som_files), 1):
        # Create the file path
        s_file_path = sl_som_files[i_entry_som]

        # Open the summary file to determine how many clusters there are in the analysis
        df_summary = pd.read_excel(s_file_path, sheet_name='Timeseries Key', index_col=0, parse_dates=True)

        # Calculate the number from the clusters
        i_number_of_clusters = np.max(df_summary['Cluster'])

        # Loop and import the data for each cluster
        l_clusters = []
        for i_entry_cluster in range(0, i_number_of_clusters + 1, 1):
            df_cluster = pd.read_excel(s_file_path, sheet_name=str(i_entry_cluster) + ' Timeseries', index_col=0, header=0)
            df_cluster.index = df_cluster.index.astype(int)

            l_clusters.append(df_cluster)

        # Append into the summary lists
        fl_som_summary.append(df_summary)
        l_som_timeseries.append(l_clusters)

    # Get all of the timeseries into a single list
    fl_som_timeseries = [pd.concat([y.drop(axis=1, columns=['Date', 'Subgroup']) for y in x]).sort_index() for x in l_som_timeseries]

    ### Plot the resolution of the subclustering ###
    [construct_subcluster_counts(fl_som_summary[x], 'som_' + str(os.path.dirname(sl_som_files[x]))) for x in range(0, len(fl_som_summary), 1)]

    ### Within cluster similarity ###
    # Manual
    l_manual_within_similarity = [np.mean(within_cluster_similarity(l_manual_windows[x], il_manual_labels[x])) for x in range(0, len(l_manual_windows), 1)]

    # Major clusters
    l_som_major_within_similarity = [np.mean(within_cluster_similarity(fl_som_timeseries[x].values, fl_som_summary[x]['Cluster'].values.flatten()))
                                     for x in range(0, len(fl_som_summary), 1)]

    ### Between cluster similarity ###
    # Manual
    l_manual_between_similarity = [np.mean(between_cluster_similarity(l_manual_windows[x], il_manual_labels[x])) for x in range(0, len(l_manual_windows), 1)]

    # Major clusters
    l_som_major_between_similarity = [np.mean(between_cluster_similarity(fl_som_timeseries[x].values, fl_som_summary[x]['Cluster'].values.flatten()))
                                      for x in range(0, len(fl_som_summary), 1)]


    ### Calinski Harabasz_score ###
    # Manual
    l_manual_ch = []
    for x in range(0, len(l_manual_windows), 1):
        try:
            d_value = calinski_harabasz_score(l_manual_windows[x], labels=il_manual_labels[x])
        except:
            d_value = np.NaN

        l_manual_ch.append(d_value)

    # Major clusters
    l_som_major_ch = [calinski_harabasz_score(fl_som_timeseries[x].values, fl_som_summary[x]['Cluster'].values.flatten()) for x in range(0, len(fl_som_summary), 1)]

    ### Davies Bouldin ###
    # Manual
    l_manual_db = []
    for x in range(0, len(l_manual_windows), 1):
        try:
            d_value = davies_bouldin_score(l_manual_windows[x], labels=il_manual_labels[x])
        except:
            d_value = np.NaN

        l_manual_db.append(d_value)

    # Major clusters
    l_som_major_db = [davies_bouldin_score(fl_som_timeseries[x].values, fl_som_summary[x]['Cluster'].values.flatten()) for x in range(0, len(fl_som_summary), 1)]

    ### Output the information ###
    # Create the output structure
    df_output_data = pd.DataFrame(columns=['Within Cluster Similarity', 'Between Cluster Similarity', 'Calinski Harabasz Score', 'Davies Bouldin Score'])

    # Add the manual data
    for i_entry_manual in range(0, len(sl_input_files), 1):
        # Get the index
        s_index = 'Manual ' + os.path.basename(sl_input_files[i_entry_manual])

        # Add the data to the frame
        df_output_data.loc[s_index] = [l_manual_within_similarity[i_entry_manual], l_manual_between_similarity[i_entry_manual], l_manual_ch[i_entry_manual],
                                   l_manual_db[i_entry_manual]]


    # Add the som data
    for i_entry_som in range(0, len(sl_som_files), 1):
        # Get the index
        s_index = 'SOM ' + os.path.dirname(sl_som_files[i_entry_som])

        # Add the data to the frame
        df_output_data.loc[s_index] = [l_som_major_within_similarity[i_entry_som], l_som_major_between_similarity[i_entry_som], l_som_major_ch[i_entry_som],
                                   l_som_major_db[i_entry_som]]

    # Save the dataframe to a file
    df_output_data.to_csv('7day_output.csv')
    print('@@')