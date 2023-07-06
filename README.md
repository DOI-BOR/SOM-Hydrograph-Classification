# Hydrologic Typing
## Overview
Stochastic rainfall/runoff models are at the forefront of hydrologic modeling state-of-practice. These models have increasingly been used by the Reclamation Technical 
Service Center (TSC) to estimate flood magnitudes and associated return periods, along with uncertainty, for detailed flood hazard studies such as issue evaluations 
(IEs) and corrective action studies (CASs). Stochastic rainfall/runoff models simulate many thousands of potential flood realizations across frequency space to estimate 
probabilistic floods and support risk analyses. The resulting products of these studies are typically flood frequency curves representing peaks, volumes, and water 
surface elevations produced from a large number of modeled hydrographs. One challenge that arises with these large datasets comes when hydrographs must be used for 
additional analyses beyond determination of existing hydrologic loads, such as design, modification, or operational changes. In these scenarios, working with a smaller 
number of hydrographs becomes necessary. The process of selecting a subset of hydrographs is currently manual, time consuming, and dependent upon the judgement of the 
person tasked with selection.

The current project developed an automated hydrograph classification workflow using a two-stage classification procedure, first a self-organizing map (SOM) machine 
learning (ML) method followed by mean shift clustering. The SOM method groups hydrographs by evaluating their similarity in the shape and magnitude. The SOM groups are 
further refined with the mean shift clustering operation to yield a small number of hydrograph clusters that are representative of the range of behavior at a site. The 
developed ML workflow is an automated process with minimal user input that runs rapidly and scales to the number of hydrographs produced by stochastic rainfall/runoff 
models. The ML hydrograph classification workflow was tested across multiple gage and model instances. In each of these cases, the ML workflow was robust and produced a 
hydrograph classification that is representative of the site.


## Instructions
### Descriptions of Files Included in Bundle
Files can be obtained from the project Github/Gitlab repository along with example datasets that can be used to test the script file performance.

1. Environment File: environment.yml
This file is used to create an environment for running the script. It downloads all packages that are required for running the script so that the user does not have to 
do any additional set up (other than adjusting user inputs in the script itself).
2. SEFM script file: som_script.py
This script contains code to run the SOM and clustering algorithms for simulated SEFM input data.
3. Gage script file: som_script_gage.py
This script contains code to run the SOM and clustering algorithms for observed gage data.

### Input Data Format/Preprocessing
This section describes how the input data should be formatted prior to running the script file. The formatting requirements are different for simulated and gage data.

#### Simulated Data
SEFM data should be formatted as each hydrograph contained in one .plt file within a folder. The script begins reading streamflow values in the third line from the top 
of the file. Therefore, each hydrograph file may have up to two header lines that are not streamflow values. Each file should contain one column of streamflow values in 
units of ft3/s.

#### Gage Data
Gage data should be contained within one comma separated (csv) file. There should be two columns: the first with dates and times for each measurement entry (formatted as 
month/day/year hour:minute with hours ranging from 0 to 23) and the second column with streamflow measurements in units of ft3/s. Only one header line containing column 
names should be present above the streamflow values.


### Environment Setup
This section provides instructions for setting up the environment to be used for running the script once the bundle of setup files has been downloaded.
1. Install Miniforge and select the option to “add to path” during the installation. Choose the most current version of Miniforge that is available for the operating system 
being utilized.
2. Open a command prompt window such as Git Bash or Windows Command Prompt within the setup folder.
3. In the command line, type “conda env create -f environment.yml” and hit enter to create a build environment from the provided configuration file.
4. In the command line, type “conda activate som”. The configuration file has already automatically named this new environment “som”.
a. To verify that the environment was correctly installed and available, type “conda env list” into the command line, and check that “som_scrip_env” is listed in the output.

### User Inputs to the Script
This section provides details on user inputs to the script that must be entered according to properties of the input data. The script file can be edited through any text 
editor, including Windows Notepad or Notepad++. It can also be edited though a Python integrated development environment, such as PyCharm.
1. In line 78, set fileName or fileLocation equal to the name of the file containing input data.
2. In line 80, set d equal to the number of days that make up one hydrograph. For gage data, this number will dictate how many days are in each sliding window created from 
the input data.
3. In line 82, set sample_freq equal to the number of datapoints recorded in one day for the input data. For example, if the data is hourly, then “sample_freq = 24”.
4. In line 85, the user can set plots equal to either “True” or “False” to choose whether intermediate plots (hydrographs in SOM cells, distribution plots for each cluster, 
etc.) will be generated. Cluster plots are generated every time the script runs regardless of the user’s choice. Setting plots to “False” will reduce runtime.
5. In line 88, the user can choose to adjust the correction factor that adjusts cluster sorting depth. Increasing this value will increase the number of clusters. The default 
correction value was chosen based on sensitivity tests for multiple types of data. Users should review the output classifications prior to adjusting this parameter.

### Running the Script File
This section describes how to run the script file in a command prompt window.
1. Prior to running the script, type “activate som” in the command line to ensure that the previously created environment is being used.
2. In the command line, type either “python som_script_SEFM.py” or “python som_script_gage.py” depending on whether the input data will be simulated from SEFM or observed 
gage data.
3. As the script runs, it will output updates on its progress in the command window. The datasets used for testing required around 30 minutes of runtime. However, computation 
time will vary based on the size of the dataset and the computer configuration.

### Classification Outputs
The classification process outputs numerous plots and files at the end of the analysis. These include class assignments for the hydrograph realizations as well as diagnostic 
data. The diagnostic data can be optionally disabled to improve the solution times. It is strongly recommended that users carefully review the outputs prior to using the 
classification in case modifications to the algorithm input parameters are necessary to refine the assignments of the hydrograph realizations.

#### Results Spreadsheet
The “Hydrograph Key” sheet contains a list of every hydrograph from the input data as well as which SOM node and final mean shift cluster the hydrograph realization was 
classified.

All sheets after the “Hydrograph Key” are in pairs, with two sheets for each cluster. The first sheet contains all hydrograph data from the cluster. The second sheet lists 
the SOM cells within the cluster and the weight vector for each cell.

#### Cluster Metrics Folder
Contains a csv file for each cluster that gives the SOM nodes contained in the mean shift cluster. For each SOM node, the parameters include the average hydrograph volume, 
minimum/maximum difference between a hydrograph value in the SOM node and the node weight vector, average slope of the SOM node weight vector, and number of intersections of 
hydrographs with the SOM node weight vector.

#### Cluster Plots Folder 
Contains a png image for each cluster. Each image shows a plot of every hydrograph contained in the cluster along with a thicker line that represents the average nodal weight 
vector of the cluster.

The x and y-axis ranges are not held constant between plots so that the shape over time in each cluster is more apparent.

#### Fixed range Cluster Plots Folder
Contains a png image for each cluster. Each image shows a plot of every hydrograph contained in the cluster along with a thicker line that represents the average nodal weight 
vector of the cluster.

The x and y-axis ranges are held constant in every plot so that flow magnitude can be compared between clusters.

#### SOM Cell Plots Folder
Contains a png image for every SOM node. Each image shows a plot of every hydrograph contained in each node of the SOM.

The x and y-axis ranges are not held constant between plots so that shape over time is more apparent.

#### SOM Cell Plots Weight Folder
Contains a png image for every SOM node. Each image shows a plot of every hydrograph contained in one node of the SOM along with a thicker line that represents the weight vector 
of the SOM cell.

The x and y-axis ranges are not held constant between plots so that shape over time is more apparent.

#### Distributions Folder
Contains four distribution plots for each cluster: hydrograph mean, number of peaks, largest peak value, and hydrograph volume.

Each distribution plot is generated by calculating the parameter for each SOM node included in the cluster using the hydrographs within the SOM node.

### Interpreting the Classification
Users should review the classification output critically to confirm that it is capturing behavior for the basin under analysis. While the particular features under investigation 
will change among basins, the output structure is intended to highlight different features at various stages of the classification to facilitate the user review. The following 
descriptions illustrate some the features that could be reviewed for each analysis output.

#### Results Spreadsheet
This output can be used to determine which cluster or SOM node a specific hydrograph was sorted into, or to examine the weights of the SOM nodes contained within each cluster.

#### Cluster Metrics Folder
The csv files contained in this folder can be used to identify which SOM nodes have been sorted into each cluster. The user could then use the SOM node indices from these files 
to gain further understanding of the classification represented in the cluster by referencing the SOM node plots to visualize the hydrographs in the SOM node.

The other metrics, especially hydrograph volume, displayed in each cluster’s file can also be used to gauge the difference between the flow magnitudes and shapes in different 
clusters, and can be used as a check to make sure that the clusters are distinct from each other.

#### Cluster/Fixed Range Cluster Plots Folders
The images in the Cluster Plots folder can be used to view the shape over time in each cluster and to assess whether the cluster weight line is an accurate representation of each 
cluster’s characteristics.

The images in the Fixed Range Cluster Plots folder can be used to assess how each cluster’s flow magnitude compares to the others.

#### SOM Cell Plot Weight Folders
These plots can be used to view the hydrographs categorized into each SOM node, and visually check that the categorizations seem distinct from one another. The plots with the SOM 
node weight vector included can be used to check whether the SOM node is accurately representing the characteristics of the hydrographs within it.

#### Distributions Folder
The histograms in this folder are visualizations of the parameters contained in the Cluster Metrics spreadsheets. Therefore, they can be used similarly to the Cluster Metrics to 
confirm that the clusters each have distinct properties such as mean and volume distribution as well as to gain a deeper understanding of what types of hydrographs each cluster 
represents.

