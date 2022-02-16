import numpy as np
import pandas as pd
import os
import math
import minisom
from minisom import MiniSom
import copy
import matplotlib
import matplotlib.pyplot as plt
import sklearn
from sklearn.cluster import MeanShift
import openpyxl

def peakVal(hydrograph):
    """
    Finds and returns the value of the largest peak (defined as the value with the greatest absolute difference from the mean of the array)
    in the input array
    
    Parameters
    ----------
    hydrograph : np.array
        An array containing streamflow values 
    
    Returns
    -------
    peak : float
        The greatest difference between any value in the array and the mean of the array
    
    """
    #Sets greatest peak value at zero to start
    peak = 0
    #Calculates each hydrograph's mean
    h_mean = np.mean(hydrograph, axis = 0)

    for entry in hydrograph:
        #Calculates difference between every value in the hydrograph and the individual hydrograph's mean
        diff = abs(entry - h_mean)
        #If the difference is greater than the current peak value, then the largest peak value is set as the difference
        if (diff > peak):
            peak = diff    
    return peak

def numPeaks(hydrograph, percentile = 85):
    """
    Returns the number of peaks in an array by calculating how many times the array intersects with a percentile line
    
    Parameters
    ----------
    hydrograph : np.array
        An array containing streamflow values 
    percentile : int, optional
        Sets the percentile to use when calculating the number of peaks, default is the 85th percentile
    
    Returns
    -------
    count : int
        The number of times that values in the input array intersect with the set percentile (used as number of peaks)
    """
    #Calculates the 85th percentile of each hydrograph
    #A peak is defined as being greater than the 85th percentile of the hydrograph
    percentile_val = np.percentile(hydrograph, percentile)
    #Makes a list filled with the percentile value so it can be compared to all values in the hydrograph
    percentile_line = np.empty(len(hydrograph))
    percentile_line.fill(percentile_val)
    #Calculates how many times each hydrograph intersects with the percentile line
    diff = hydrograph - percentile_line
    #Sets a condition that is true when the difference between hydrograph value and percentile line is greater than 0
    positive = diff > 0
    #Keeps track of when the difference goes from being positive to negative, which indicates an intersection
    count = np.logical_xor(positive[1:],positive[:-1]).sum()
    
    return count

if __name__ == "__main__":

    ##User enters basin and subbasin details

    #First line of CSV should be column names
    #First 2 columns of data should be datetime and streamflow values (in ft^3/s)
    fileLocation = "USGS data.csv" 
    #Specify the number of units(days) in each sliding window
    d = 7
    #Specify the number of samples taken each day
    sample_freq = 96
    #If user wants to generate metric cluster plots and SOM cell hydrograph plots, set plots to True below
    #Generating plots increases runtime
    plots = True

    #A correction factor for clustering bandwidth - user may need to tweak 
    correction = 1.25
    #The y-axis limit for fixed range cluster plots. 
    #User will most likely need to adjust based on dataset if fixed range cluster plots are desired
    min_y = 0
    max_y = 7000

    ################################################## Preprocessing #######################################################

    df = pd.read_csv(fileLocation)
    #Takes a df w/ 2 columns at a time (date/time and stream flow)
    data = df.iloc[:, 0:2]
    #Uses a smaller subset of the data for now so that testing doesn't take so long
    data = data[0:200000]

    #If the dataset has negative values, those are changed to NaN's
    if ((df.iloc[:,1] < 0).values.any()):
        #First, makes a new dataframe not including the data/time column so that it's easier to focus on values in other columns
        dates = data.iloc[:,0]
        data = data.drop(data.columns[0], axis = 1)
        #Next, change negative values to NaN, which python has specific functions for eliminating:
        #First iterate through the dataframe by column
        for column in data:
            i = 0
            #Iterate through every value in each column, using i as an index
            for value in data[column]:
               if (value <0):
               #replaces negative values with "NaN"
                   data[column][i]= "NaN"
               i += 1    
        #Puts back the data/time column
        data.insert(0, "DateTime", dates)  
 
    #Fills in NaNs where possible by linearly interpolating surrounding data
    data.iloc[:,1].interpolate(method = 'linear', limit = 2)

   #Changes sampling frequency if it isn't already hourly
    if sample_freq != 24:
	#Creates date/time column that is datetime format
        data["DateTime"] = pd.to_datetime(data.iloc[:,0])
	#Indexes by the datetime column and keeps only hourly measurements
        data.index = data['DateTime']
        hourly_data = data.resample('H').mean()
	#Makes a dataframe from the hourly data
        data = pd.DataFrame(hourly_data)
	#Inserts the new datetime column into the first column of the dataframe
        data.insert(0, "DateTime", data.index)
        sample_freq = 24

    #User must enter desired number of days per window as well as sampling frequency of data above
    window_hours = d*sample_freq

    #Makes the data column into an array to be used with the sliding windows function below
    streamFlows = np.asarray(data.iloc[:,1])

    #Divides data into sliding windows, each containing the desired number of days per window
    streamFlows = np.lib.stride_tricks.sliding_window_view(streamFlows, window_hours)[::sample_freq,:]

    #Creates column names for a correctly sized dataframe
    columns = []
    for i in range(d*sample_freq):
        #Column names are 0 to n
        columns.append(i)
    #Puts streamflow data into the data frame using column names above
    newdf = pd.DataFrame(streamFlows, columns = columns)

    #Identify rows with NaNs by taking a sum of every row that preserves NaNs and adding it as a column to the newdf
    newdf["sums"] = newdf.sum(axis = 1, skipna = False)

    ##Adds a start Date/Time column

    #Converts date times to date/time objects
    data["DateTime"] = pd.to_datetime(data.iloc[:,0])

    #Makes two arrays from the datetime objects and doesn't include the time variable in one so that the unique function can be used below
    dateTimes = np.asarray(data["DateTime"])
    uniqueCheck = np.asarray(data["DateTime"].dt.date)

    #Sorts through dates and adds unique dates to the start dates array
    uniqueDates, indices = np.unique(uniqueCheck, return_index = True)
    startDates = dateTimes[indices]
    #Adjusts list of start dates/times to account for the 6 days in the last 7-day window that will never be start dates
    #This makes the start_datetimes list have the same number of entries as the number of windows from the data
    difference = len(startDates) - len(streamFlows)
    startDates = np.delete(startDates, slice(-difference, len(startDates)))

    #Adds a column of start dates/times to the dataset
    newdf.insert(0, "Start Date", startDates)  

    #Adds a sample frequency column to the df that tells how often samples are taken each day
    freq_column = []
    for i in range (len(newdf)):
        freq_column.append(sample_freq)
    newdf.insert(1, "Samples Per Day", freq_column)
    
    #Goes through the sums column and removes rows with sums that are NaNs
    i = 0
    for entry in newdf["sums"]:
        if math.isnan(entry):
            newdf = newdf.drop(labels = i, axis = 0)
        i += 1

    print("Number of hydrographs:", len(newdf))

    ########################################## Creates folders for script file outputs ###############################################

    #Makes an overall path to hold all results
    script_dir = os.path.abspath("hydro_extraction_results")

    #Creates paths for cluster plots, distribution plots, and metric spreadsheets in outputs
    clusters = os.path.join(script_dir, "Cluster_plots/")
    fixed_cluster = os.path.join(script_dir, "Fixed_Range_Cluster_Plots/")
    dists = os.path.join(script_dir, "Distributions/")
    metrics = os.path.join(script_dir, 'Cluster_metrics/')

    #If any of the folders above do not exist, they are created
    if not os.path.isdir(script_dir):
        os.makedirs(script_dir)
    if not os.path.isdir(clusters):
        os.makedirs(clusters)
    if not os.path.isdir(fixed_cluster):
        os.makedirs(fixed_cluster)
    if not os.path.isdir(metrics):
        os.makedirs(metrics)

    ##################### Calculating Largest Peak Value and Number of Peaks per Hydrograph Prior to the SOM ###################################

    #creates a dataset without the frequency column for hydrograph analysis
    #allows the script file to be restarted from this cell after adjustments are made instead of re-reading all input data
    data = newdf.iloc[:, 2:len(newdf.columns) - 1]

    #Makes  a list of 0 through the number of hours in each window
    time_hours = range(data.shape[1])

    #Calculates means of individual hydrographs
    #hydrograph_mean = np.mean(data)
    #Calculates overall mean of all hydrographs by taking mean of all hydrograph means
    overall_mean = np.mean(np.mean(data))

    #Makes input data an array so that it can be iterated through to perform calcs on each hydrograph
    array = data.to_numpy()

    #Finds the value of the largest peak and the number of peaks in each hydrograph
    peaks = []
    intersections = []
    #iterates through every hydrograph in the input data
    for h in array:
        #Calls the peakVal function
        peak = peakVal(h)
        #Calculate the individual hydrograph's mean
        h_mean = np.mean(h)
        if h_mean == 0:
            peaks.append(0)
        #Scales the peak value by the hydrograph mean and the overall mean of all hydrographs and adds to a list of peaks
        else:
            scaled_peak = peak/h_mean*overall_mean
            peaks.append(scaled_peak)

        #Adds number of intersections with a percentile line (peaks) to a list by calling numPeaks function
        intersections.append(numPeaks(h))
    #Finds the maximum number of peaks out of all hydrographs   
    max_peaks = max(intersections)
    #Scales the numbers of peaks list by the maximum number of peaks and overall hydrograph mean
    scaled_intersects = (intersections/max_peaks*overall_mean)

    #Adds columns containing number of peaks and largest peak values (one value for each parameter for each hydrograph) to the input data  
    data["Number of Peaks"] = scaled_intersects
    data["Peaks"] = peaks
    array = data.to_numpy()

    ############################### Creates and trains a 25 x 25 som on the input data array ####################################################

    #Creates and trains a 25 x 25 som on the input data array
    #Sets a random seed so that results are reproducable
    somTest = MiniSom(25, 25, array.shape[1], random_seed = 1)
    #Uses 2500 training iterations, which was determined to be sufficient for reducing difference between iterations
    somTest.train(array,2500)

    #win_map shows which hydrographs from input data are contained in each bin of the trained SOM
    win_map = somTest.win_map(array, return_indices = True)
    #Makes a copy of win_map to reference later after changing cell names from original win_map
    win_map_copy = copy.copy(win_map)

    print("Number of SOM cells used:", len(win_map))

    #After fitting the SOM, drops the peaks and intersections from the data so that the hydrograph plots aren't skewed
    data = data.drop("Peaks", axis = 1)
    data = data.drop("Number of Peaks", axis = 1)
    array = data.to_numpy()

    ##Renames win_map dictionary with key names that are easier to iterate through

    #Preserves current key names in a list
    old_keys = list(win_map.keys())

    #Replaces every old key name with numbers from 0 to the number of SOM cells used
    for i in range(len(win_map)):
        old = old_keys[i]
        win_map[i] = win_map.pop(old)

    #Makes a list of entries from the win_map dictionary
    dictList = list(win_map.items())
    #Sorts the list of entries by how many hydroraphs are contained in each bin, from highest to lowest number
    dictList.sort(key=lambda x: len(x[1]), reverse = True)

    ################################################## Plotting hydrographs from each SOM cell ############################################
    ##Makes plots of the all hydrographs contained in each bin if user has set to generate plots
    if plots == True:
        #Creates a folder for plots of all hydrographs in each SOM cell without SOM weights shown
        hydro_plots = os.path.join(script_dir, 'SOM_Cell_Plots/')
        #If folder for hydrograph plots doesn't exist, creates a folder
        if not os.path.isdir(hydro_plots):
            os.makedirs(hydro_plots)

        # Iterates through the dictionary contianing each SOM cell and the associated hydrograph indices
        for cell, value in dictList:
            #For each hydrograph number, the corresponding row/hydrograph from the input data is referenced
            hydroData = array[value]
            #Plots the hydrograph as a function of time
            plt.rcParams["font.family"] = "serif"
            plt.rcParams["font.serif"] = "Times New Roman" 
            plt.plot(time_hours,hydroData.T)
            plt.xlabel("Hours in Window")
            plt.ylabel("Streamflow ($ft^3$/s)")
            plt.xlim(left = 0) #Starts the x axis values at 0
            plt.grid()
           #Names each plot after the SOM cell it represents
            plt.title(old_keys[cell])

            #Saves each plot as a png image in the Som_Cell_Plots folder
            fileName = str(old_keys[cell]) + ".png"
            plt.savefig(hydro_plots + fileName, dpi = 300)
            plt.close()
    
    ###Plots hydrographs from each bin along with the overall weight vector of the bin

    #Gets weights and distances for each cell in the SOM
    weights = somTest.get_weights()
    distances = somTest.distance_map()

    #Makes lists to be filled with volumes, relevant distance (distances from cells that contain hydrographs), intersections
    #and largest peak outputs for each cell
    volumes = []
    relevant_distances = []
    peaks = []
    intersections = [] 

    #Creates a pat for plots of all hydrographs in each SOM cell with SOM weights shown
    weight_plots = os.path.join(script_dir, 'SOM_Cell_Plots_Weight/')

    for cell, value in dictList:
        #For each hydrograph number, the corresponding number window from the input data is plotted
        hydroData = array[value]
        #The cell number is read in as coordinates in the weight vector, and the weight vector is referenced accordingly
        m, n = old_keys[cell]
        #The last two elements of the weight vector are for the number of interesections and peak value parameters, and are not plotted
        weight = weights[m, n, :-2]

        #Calculates the volume under the weight curve for each SOM cell and adds it to a list
        volumes.append(np.trapz(weight, time_hours))
        #Makes a list of distances for cells that have associated hydrographs (instead of including distances for all cells on the map)
        relevant_distances.append(distances[m][n])
        #Calls the peakVal function for each weight
        peaks.append(peakVal(weight))
        #Adds number of intersections with a percentile line (peaks) for each weight to a list by calling numPeaks function
        intersections.append(numPeaks(weight))
        if plots == True:
            #If user is generating plots and plot folder does not exist, creates a folder to hold plots
            if not os.path.isdir(weight_plots):
                os.makedirs(weight_plots)

            #Plots the hydrograph as a function of time from the beginning of the window
            plt.rcParams["font.family"] = "serif"
            plt.rcParams["font.serif"] = "Times New Roman"
            plt.plot(time_hours,hydroData.T)
            plt.plot(time_hours, weight.T, linestyle = "", marker = "o")
            #Uncomment line below to displays the percentile line from the cell weight as a black dashed line
            #plt.plot(time_hours, percentile_line, linestyle = '--', color = 'k', linewidth = 2)
            plt.xlabel("Hours in Window")
            plt.ylabel("Streamflow ($ft^3$/s)")
            plt.xlim(left = 0) #Starts the x axis values at 0
           #Names each plot after the bin it represents
            plt.title(old_keys[cell])
            plt.grid()
            fileName = str(old_keys[cell]) + ".png"
            plt.savefig(weight_plots + fileName, dpi = 300)
            plt.close()
 
    ################################################# Mean-Shift Clustering ######################################################
    #Data must be numpy array to fit
    relevant_distances = np.array(relevant_distances)
    volumes = np.array(volumes)
    peaks = np.array(peaks)
    intersections = np.array(intersections)

    #Stacks rows of data so that they are all in one array instead of array of arrays
    flat = relevant_distances.flatten()
    flat_volumes = volumes.flatten()
    flat_peaks = peaks.flatten()
    flat_intersections = intersections.flatten()

    #Reshapes the arrays so that they are in the correct form for the fit_predict function
    flat_distances = flat.reshape(len(relevant_distances),1)
    flat_volumes = flat_volumes.reshape(len(volumes),1)
    flat_peaks = flat_peaks.reshape(len(peaks), 1)
    flat_intersections = flat_intersections.reshape(len(intersections), 1)

    #Normalizes volume and distance outputs from 0 to 1 based on min and max values in each array
    scaler = sklearn.preprocessing.MinMaxScaler()
    volumes_scaled = scaler.fit_transform(flat_volumes)
    distances_scaled = scaler.fit_transform(flat_distances)
    peaks_scaled = scaler.fit_transform(flat_peaks)
    intersect_scaled = scaler.fit_transform(flat_intersections)

    #Re-flattens the scaled volumes so that they can be combined with distances in an array
    flat_volumes_scaled = volumes_scaled.flatten()
    flat_distances_scaled = distances_scaled.flatten()
    flat_peaks_scaled = peaks_scaled.flatten()
    flat_intersect_scaled = intersect_scaled.flatten()

    #Makes a 5D input for the clustering model (distance, volume, peak value, number of peaks)
    dist_vol_peaks_intersect = np.array([flat_distances_scaled, flat_volumes_scaled, flat_peaks_scaled, flat_intersect_scaled])

    #Estimates the best bandwidth parameter to use in mean shift clustering model
    bw = sklearn.cluster.estimate_bandwidth(dist_vol_peaks_intersect.T)
    #Divides bandwidth by a correction factor to adjust number of clusters
    bw = bw/correction

    #Initializes a mean shift clustering model with the bandwidth from the previous cell
    clustering = MeanShift(bandwidth = bw)

    #Assigns each SOM cell to a cluster based on distance, volume, peak value, and number of peaks
    label = clustering.fit_predict(dist_vol_peaks_intersect.T)

    #Finds the number of clusters
    u_labels = np.unique(label)
    print("Number of Clusters: ", len(u_labels))

    #Adds coordinates for SOM cells containing hydrographs to the dataframe
    cellList = []
    for cell, value in dictList:
        cellList.append(old_keys[cell])

    #Makes a data frame with the input distance data, the cluster assigned to each data point, and the cell index for each point
    df = pd.DataFrame(flat_distances_scaled)
    df["Volume"] = volumes_scaled
    df["Cluster"] = label
    df["Cell Num"] = range(0,len(flat))
    df["Cell Index"] = cellList

    ############################################# Plotting the clusters and generating results ###################################################
    #Dictates which excel file the results will be written to
    writer = pd.ExcelWriter('hydro_extraction_results/results.xlsx')


    #Creates lists to be filled with a table for each cluster
    key_dfs = []
    som_cluster_dfs = []
    hydro_cluster_dfs = []
    #iterates through each cluster number
    for i in u_labels:
        print("Working on cluster ", i)
        #Finds the indices in the dataframe with one cluster label at a time
        indices = df.index[df['Cluster'] == i].tolist()
        #Keeps track of which cells  (referenced by a tuple) are in each cluster
        cluster_hydros = []
        weightList = []
        cell_index = []
        areaList = []
        areas = []
        min_diff = []
        max_diff = []
        ave_slopes = []
        #Goes through the bin numbers in each cluster and adds all hydrographs from each bin to a list
        for c in df["Cell Index"][indices]:
             #Filters out bins without hydrographs
            if c in win_map_copy.keys():
                #Uses winmap to determine which hydrographs correspond to each bin
                cluster_hydros.append(win_map_copy.get(c))
                #Gets cell reference dimensions and uses them to pull weights for a specific cell
                m, n = c
                weight = weights[m, n, :-2]
                #Calculates the area under each cell's weight curve and adds it to a list of areas for all cells in the cluster
                areaList.append(np.trapz(weight, time_hours))
                #Adds all cell weights for the cluster to one list
                weightList.append(weight)
                #Adds cell indices that have associated hydrographs to a list
                cell_index.append(c)

        #Stops current iteration and starts next cluster if list of weights is empty
        if len(weightList) == 0:
            continue

        #Converts list to a dataframe with every cell as one row so that an average weight row can be calculated     
        weightdf = pd.DataFrame(weightList)
        #Adds the average weight row to the end of the dataframe
        weightdf.loc[len(weightdf.index)] = weightdf.mean(axis = 0)
        #Pulls the average weight row from the dataframe and converts to an array so that it can be transposed
        weightarray = np.array(weightdf.loc[len(weightdf.index) - 1])
        #Calculates the area under the cluster's average weight curve
        ave_area = np.trapz(weightarray, time_hours)
        areas.append(areaList)

        ######################################### Calculates metrics for each cluster ############################################
    
         #Generates column names for a correctly sized dataframe to hold cluster metrics
        columns = []
        for n in range(d*sample_freq):
        #Column names are 0 to n
            columns.append(n)
        #Initializes a dataframe to hold info about SOM weights from each cluster
        som_cluster_df = pd.DataFrame(columns = columns)
        #Iterates through every cell in the cluster and calculates metrics based on weight
        j = 0
        for w in weightList:
            #Calculates cluster metrics to be output in a spreadsheet for each cluster
            #Difference between cell weight and average weight
            diff = w - weightarray
            #Maximum difference between each cell weight and ave weight
            max_diff.append(max(abs(diff)))
            #Minimum difference between each cell weight and ave weight
            min_diff.append(min(abs(diff)))
            #Calculates the best fit slope for each weight curve
            m, b = np.polyfit(time_hours, w, 1)
            #Adds the slope to a list of slopes with one slope for each weight curve
            ave_slopes.append(m)
            m_ave, b_ave = np.polyfit(time_hours, weightarray, 1)

            #Fills the weight vector for each SOM in the cluster into a dataframe
            som_cluster_df.loc[j] = w
            j += 1
        #Gets rid of hydrographs that contain missing values
        clean = [ele for ele in cluster_hydros if ele != None]
        #Keeps track of the number of hydrographs in each SOM cell
        num_hydrographs = [len(value) for value in clean]

        ##################################### Plots hydrographs and weight for each cluster ######################################

        #Initializes a list to contain hydrograph data for the cluster
        full_hydros = []
        cell_mean = []
        #Iterates through the list of hydrographs and pulls data for each one to plot it
        for value in clean:
            hydrograph = array[value]
            #Adds each hydrograph to a list
            full_hydros.append(hydrograph) 
            #calculates mean of each SOM cell
            cell_mean.append(np.mean(hydrograph))
            #Plots the hydrograph as a function of time from the beginning of the window
            plt.rcParams["font.family"] = "serif"
            plt.rcParams["font.serif"] = "Times New Roman"
            plt.plot(time_hours,hydrograph.T)
            plt.plot(time_hours, weightarray.T, linewidth = 5)
            #plt.plot(time_hours, weightarray.T, linestyle = "", marker = "o")
            plt.xlabel("Hours in Window")
            plt.ylabel("Streamflow ($ft^3$/s)")
            plt.xlim(left = 0) #Starts the x axis values at 0
           #Names each plot after the bin it represents
            plt.title( i)
            #Saves plots to pngs in the home directory
            fileName = str(i) + ".png"
            plt.savefig(clusters + fileName, dpi = 300)
        plt.close()
        #Makes fixed range plots for the example
        for value in clean:
            plt.rcParams["font.family"] = "serif"
            plt.rcParams["font.serif"] = "Times New Roman"
            hydrograph = array[value]
            #Plots the hydrograph as a function of time from the beginning of the window
            plt.plot(time_hours,hydrograph.T)
            plt.plot(time_hours, weightarray.T, linestyle = "", marker = "o")
            plt.xlabel("Hours in Window")
            plt.ylabel("Streamflow ($ft^3$/s)")
            plt.xlim(left = 0) #Starts the x axis values at 0
            plt.ylim(bottom = min_y, top = max_y)
           #Names each plot after the bin it represents
            plt.title( i)
            #Saves plots to pngs in the home directory
            fileName = str(i) + " fixed.png"
            plt.savefig(fixed_cluster + fileName, dpi = 300)
        plt.close()

        ################################# Calculates and plots distributions for each cluster #####################################

        #Creates lists to be filled with distributions for each cluster
        hydro_mean = []
        hydro_vol = []
        peak_vals = []
        intersections = []
        cell_intersects = []
        #Calculates overall mean for all hydrographs in cluster
        overall_mean = np.mean(cell_mean)
        #Iterates through every cell in the cluster and every hydrograph with each cell
        for cell in full_hydros:
            single_cell_intersects = []
            for h in cell:
                #Calculates hydrograph mean
                hydro_mean.append(np.mean(h))
                #If the mean of the hydrograph is 0, appends 0 for peak val and hydro volume
                if hydro_mean == 0 or hydro_mean == 'NaN':
                    peaks_vals.append(0)
                    hydro_vol.append(0)
                elif overall_mean == 0:
                    peaks_vals.append(0)
                    hydro_vol.append(0)
                else:
                    #Calculates scaled hydrograph volume
                    vol = (np.trapz(h, time_hours))
                    hydro_vol.append(vol/np.mean(h)*overall_mean)
                    #Calls the peak val function to calculate the largest peak in each hydrograph
                    peak = peakVal(h)
                    scaled_peak = peak/h_mean*overall_mean
                    peak_vals.append(scaled_peak)
                #Calls the numPeaks function to calculate the number of peaks in each hydrograph
                intersections.append(numPeaks(h))
                single_cell_intersects.append(numPeaks(h))
            cell_intersects.append(sum(single_cell_intersects))
        #Determines the maximum number of peaks in any of the hydrographs and uses it for scaling
        max_peaks = max(intersections)
        if (max_peaks == 0):
            peakNum = 0
        else:
            #Scales the intersection count by the individual and overall hydrograph means and adds to a list of intersections
            peakNum = intersections/max_peaks*overall_mean
            peakNum = list(peakNum)

        if plots == True:
            if not os.path.isdir(dists):
                os.makedirs(dists)
            #Outputs histograms for distribution parameters from every SOM Cell and saves to a file within the results folder
            #List of parameters to be plotted
            params = [peak_vals, peakNum, hydro_mean, hydro_vol]
            #List of labels for each histogram's x axis 
            xlabs = ["Largest Peak Value", "Number of Peaks", "Individual Hydrograph Means", "Hydrograph Volume"]
            #List of names for each distributions image to be saved as
            pngNames = [" PeakVal.png", " NumPeaks.png", " Mean.png", " Volume.png"]
            n = 0
            #Iterates through every parameter and plots the parameter for every hydrograph in the cluster as a histogram
            for distribution in params:
                plt.rcParams["font.family"] = "serif"
                plt.rcParams["font.serif"] = "Times New Roman"
                plt.hist(distribution, 300)
                plt.title("Cluster " + str(i))
                plt.xlabel(xlabs[n])
                plt.ylabel("Frequency")
                plt.xlim(left = 0) #Starts the x axis values at 0
                plt.grid()
                plt.savefig(dists + str(i) + pngNames[n], dpi = 300)
                plt.close()
                n += 1

        ############################# Generates and formats result tables to be output ########################################

        #Generates a list of cluster labels according to current iteration number to be added to df
        cluster_labels = [i for k in range(len(areaList))]
     
         #Makes a df for each cluster containing the following metrics
        clusterDf = pd.DataFrame({"Cell Index": cell_index,
                                  "Cluster" : cluster_labels, 
                                  "Hydrograph Volume": areaList, 
                                  "Min Difference": min_diff, 
                                  "Max Difference": max_diff, 
                                  "Average Slope": ave_slopes, 
                                  "Num. Intersections": cell_intersects})
        clusterDf.loc[len(weightdf.index)] = ["average", i, ave_area, 0, 0, m_ave,0]
        #Writes each cluster to a csv file
        filename = str(i) + " metricData.csv"
        clusterDf.to_csv(metrics + filename)

        #Adds columns to the SOM Cluster df listing the cell index and number of hydrographs contained in each cluster
        som_cluster_df.insert(0, "Cell Index", cell_index)                     
        som_cluster_df.insert(1,"Hydrographs in cell", num_hydrographs)    
        som_cluster_dfs.append(som_cluster_df)

        #Makes a dataframe out of the hydrographs from each SOM cell
        dfs = []
        for h in full_hydros:
            dfs.append(pd.DataFrame(h))
        #Concatenates all the SOM cell hydrograph dataframes into one large dataframe of hydrographs for the cluster
        hydro_cluster_df = pd.concat(dfs) 

        #Makes a column of zeros in the hydrograph df as a placeholder for hydrograph numbers
        hydro_cluster_df.insert(0, "Hydrograph Number", np.zeros(len(hydro_cluster_df)))
        m = 0
        #Goes through the list of hydrographs in each cell and adds them to the first column of the hydro_cluster df
        for value in clean:
            for n in value:
                hydro_cluster_df.iloc[m,0] = n
                m += 1
        #Adds the hydrograph dataframe to a list of dataframes so that output can be written to an excel file at the end of the loop
        hydro_cluster_dfs.append(hydro_cluster_df)

        #Creates a list of cell indices that can be read next to the list of hydrograph numbers to indicate which cell each hydrograph is in
        df_cell_index = []
        df_cluster = []
        k = 0
        #Goes through the hydrographs in each cell
        for n in clean:
            #Duplicates the cell index for every hydrograph contained in the cell
            for a in range(len(n)):
                df_cell_index.append(cell_index[k])
                #Also makes a list containing cluster index that is the same length as the number of hydrographs in the cluster
                df_cluster.append(i)
            k += 1    

        #Adds hydrograph number, cell index for each hydrograph, and cluster for each hydrograph to a dataframe
        key_df = pd.DataFrame(columns = ["Hydrograph Number", "SOM Cell Index", "Cluster"])
        key_df["Hydrograph Number"] = hydro_cluster_df["Hydrograph Number"]
        key_df["SOM Cell Index"] = df_cell_index
        key_df["Cluster"] = df_cluster
        key_dfs.append(key_df)

    print("Writing results to excel file")
    #Concatenates dataframes from every cluster so that they are all in one table
    key_df = pd.concat(key_dfs)
    #Sorts the table by hydrograph number so that hydrographs are in order
    key_df = key_df.sort_values("Hydrograph Number")
    #Writes the table to a sheet in excel
    key_df.to_excel(writer, sheet_name = "Hydrograph Key", index = False)
    writer.save()

    #Writes the other dataframes to individual sheets for each cluster in the same excel file 
    for d in range(len(som_cluster_dfs)):
        hydro_cluster_dfs[d].to_excel(writer, sheet_name = str(d) + " Hydrographs", index = False)
        som_cluster_dfs[d].to_excel(writer, sheet_name = str(d) + " SOM Weights", index = False)
        writer.save()
    
    writer.close()

    print("Done")