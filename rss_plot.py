# -*- coding: utf-8 -*-
"""
Beginnings of a global mapper for "RSS style" global maps
by RSS style I mean element[0,0] is lat=-90+dellat/2.0,long = 0 + dellon/2.0

This depends on cartopy, which depends on a lot on components.  To install in
anaconda, use

conda install -c scitools cartopy

@author: Carl Mears
"""

    
def bounds_pretty(x,vmin_in=-999.0,vmax_in=999.0,lower_zero=False):
    from math import log,ceil,floor
    import numpy as np
    
    if type(x).__module__ != np.__name__:
        raise Exception, "Input must be a numpy array"

    vmin = vmin_in
    vmax = vmax_in
    if ((vmin_in ==  -999.0) or (vmax_in == 999.0)):
        data_range = [np.nanmin(x),np.nanmax(x)]
        if lower_zero:
            data_range[0] = 0.0
        diff  = data_range[1] - data_range[0]
        if diff == 0:
            # special case of no variability
            vmin = round(data_range[0]-1.0,1)
            vmax = round(data_range[1]+1.0,1)
        else:
            l = log(diff/3.,10)
            pow10 = floor(l)
            x = l - pow10
            y = pow(10.0,x)
            delta = 1.0
            if (y < 1.5):
                delta = 1.0
            if ((y >= 1.5) and (y < 3.0)):
                delta = 2
            if ((y >= 3.0) and (y<7.0)):
                delta = 5
            if ((y >= 7.0) and (y<10.0)):
                delta = 1
                pow10 = pow10 + 1
            delta = delta*pow(10.0,pow10)
            vmin = delta*floor(data_range[0]/delta)
            vmax = delta*ceil(data_range[1]/delta)
            
        if vmin_in <> -999.0:
            vmin = vmin_in
        if vmax_in <> 999.0:
            vmax = vmax_in
    return [vmin,vmax]
    
def plot_global_map(map_in,proj = 'Mollweide',vmin_in = -999.0,vmax_in = 999.0,lower_zero=False,title='Plot',units=' ',central_longitude = 90.0,ct_name = 'jet'):
    
    import matplotlib.pyplot as plt
    import numpy as np
    import cartopy.crs as ccrs
    
    figsize = (8.0,4.0)
    fig = plt.figure(figsize=figsize)
    
    if proj == 'Rectangular' :
        ax = plt.axes(projection=ccrs.PlateCarree(central_longitude = central_longitude))
    elif proj == 'Mollweide':
        ax = plt.axes(projection=ccrs.Mollweide(central_longitude = central_longitude))
    else:
        print 'Can not find projection '+proj
        print 'Setting proj to rectangular'
        proj = 'Rectangular'
        ax = plt.axes(projection=ccrs.PlateCarree(central_longitude = central_longitude))
        
    nlats, nlons = map_in.shape
    lats = np.rad2deg(np.linspace(-np.pi/2, np.pi/2, nlats+1))
    lons = np.rad2deg(np.linspace(0.0,2.0*np.pi, nlons+1))


    acmap = plt.get_cmap(ct_name)
    [vmin,vmax] = bounds_pretty(map_in,vmin_in,vmax_in,lower_zero)
    
    if proj == 'Rectangular' :
        p = plt.pcolormesh(lons, lats, map_in, cmap=acmap,transform=ccrs.PlateCarree(),vmin=vmin,vmax=vmax)
    elif proj == 'Mollweide':
        p = plt.pcolormesh(lons, lats, map_in, cmap=acmap,transform=ccrs.PlateCarree(),vmin=vmin,vmax=vmax)

    ax.coastlines()
    ax.set_global()
    cb = plt.colorbar(p, orientation='horizontal',shrink = 0.6)
    cb.ax.set_title(title)
    plt.show()
