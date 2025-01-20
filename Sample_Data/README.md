# hdp-hmm-te Sample Data
There are two types of sample time series data included here:  Two dimensional data of loops (with 1, 2, or 3 loops per frame),  and three dimensional time series of randomly selected from spherical normal random variables with different variances.

## Usage
The time series data (leaf_series_data and Var_Change_Points files) must first be converted to a Julia serialization object (.jls file) using the timeSeries_to_PIs function available in the TDA_Tools package in the Source_Code folder.  The syntax is shown below.

timeSeries_to_PIs(time_series, output_file, window_sz, slide; jump=1, max_dim=1, pi_sz=5, stack=false, converter=nothing)

time_series is an array of points (as tuples)

output_file is where the data will be stored (as a .jls file)

window_sz is the number of data points that will be used to create the persistence image

slide is how much the window will advance each iteration

jump allows you to pick how many samples in the window to use for computation (jump = 10 would take every 10th sample)

max_dim is the maximum dimension used in the persistene image (increasing this will greatly slow down the computation)

pi_sz is the size of the (typically) square matrix representing the persistence image.
        If this is an int, all dimensions will have the same base size.
        You can specify an array of sizes for each dimenion (array should be length max_dim + 1).
        
stack=true will stack all PIs into a single N X 1 vector

The method will return a converter object for producing persistence images which can be passed back to the method on the test data (so that both sets of images are produced with the same process).

The state data files must be modified to the correct length depending on the choice of window size and slide rate.
