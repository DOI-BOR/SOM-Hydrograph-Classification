def calculate_peak_value(hydrograph):
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


def count_number_of_peaks(hydrograph, percentile=85):
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