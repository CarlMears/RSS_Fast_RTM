
def global_map(a, vmin=0.0, vmax=30.0, cmap=None, plt_colorbar=False,title=''):
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import cartopy.crs as ccrs

    img_extent = [-180.0, 180.0, -90.0, 90.0]
    fig = plt.figure(figsize=(10, 5))  # type: Figure
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree(),title=title)
    for item in ([ax.title]):
        item.set_fontsize(16)
    map = ax.imshow(np.flipud(np.roll(a, 720, axis=1)), cmap=cmap, origin='upper', transform=ccrs.PlateCarree(),
                    norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax), extent=img_extent)
    if plt_colorbar:
        cbar = fig.colorbar(map, shrink=0.7, orientation='horizontal')
        cbar.ax.tick_params(labelsize=14)
    ax.coastlines()
    ax.set_global()
    return fig, ax
