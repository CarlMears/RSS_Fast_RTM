# RSS_Fast_RTM
This is partly a work in progress.  The only part that is really complete is the method_1 code, which is entirely in the file
AtmWts_method_1.py.  

The file contains test code at the bottom.

To run the test code, you need to use era5_monthly.py and global_map.py.

The test code is set up to profile.  To turn off profiling, set perform_profile to False, or delete the profiling code.

The test comparison results in ./test are from the IDL version of the method, calculated using t2m for surface temperature.

The environment can be created with: 
    conda create -n rss -c conda-forge numba matplotlib python=3.7 ipython netCDF4 numpy xarray cartopy
    conda activate rss