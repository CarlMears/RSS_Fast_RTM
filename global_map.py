
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, addcyclic, shiftgrid
import numpy as np

def global_map(map_data,lats,lons,figsize = (10,5),projection='moll',units = 'units go here',lon_0 = 0.0):
    fig = plt.figure(figsize=figsize)
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)
    # Setup the map. See http://matplotlib.org/basemap/users/mapsetup.html
    # for other projections.
    m = Basemap(projection='cyl', llcrnrlat=-90, urcrnrlat=90,\
                llcrnrlon=0, urcrnrlon=360, resolution='c', lon_0=lon_0)
    m.drawcoastlines()
    m.drawmapboundary()
    # Make the plot continuous
    map_cyclic, lons_cyclic = addcyclic(map_data, lons)
    # Shift the grid so lons go from -180 to 180 instead of 0 to 360.
    map_cyclic, lons_cyclic = shiftgrid(180.0, map_cyclic, lons_cyclic, start=False)
    
    #shiftdata(lonsin, datain=None, lon_0=None)
    # Create 2D lat/lon arrays for Basemap
    lon2d, lat2d = np.meshgrid(lons_cyclic, lats)
    # Transforms lat/lon into plotting coordinates for projection
    x, y = m(lon2d, lat2d)
    # Plot of air temperature with 11 contour intervals
    cs = m.contourf(x, y, map_cyclic, 21, cmap=plt.cm.PuOr_r)
    cbar = plt.colorbar(cs, orientation='horizontal', shrink=0.5)
    cbar.set_label(units)
    
    print
