import numpy as np
from netCDF4 import Dataset
from numba import jit

from matplotlib import pyplot as plt
import timeit

import cProfile, pstats, io
from pstats import SortKey

#comment out the line below to see how much slower regular python is

@jit(nopython = True)
def AtmLevelWts_Numba(  weighting_function = None,
                        surface_weight = None,
                        space_weight = None,
                        pressure = None,
                        pressure_bounds = None,
                        surface_pressure = None,
                        temp_profiles = None,
                        ts = None,
                        num_lats = None,
                        num_lons = None,
                        num_levels = None,
                        ps = None,
                        levels = None,
                        land_frac = None,
                        surface = None):
    '''

    :param weighting_function:
    :param surface_weight:
    :param space_weight:
    :param pressure:
    :param pressure_bounds:
    :param surface_pressure:
    :param temp_profiles:
    :param ts:
    :param num_lats:
    :param num_lons:
    :param num_levels:
    :param ps:
    :param levels:
    :param land_frac:
    :param surface:
    :return:
    '''

    MAX_SURF_PRESSURE = 1100.0
    MIN_SURF_PRESSURE = 500.0
    level_wts = np.zeros((num_levels,num_lats,num_lons))
    surface_wts = np.zeros((num_lats,num_lons))
    space_wts = np.zeros((num_lats,num_lons))
    tbs = np.zeros((num_lats,num_lons))


    for ilat in np.arange(0, num_lats):
        for ilon in range(0, num_lons):
            ps_flt = ps[ilat, ilon]

            ps_int = int(np.round(ps_flt))
            if ps_int > MAX_SURF_PRESSURE:
                raise ValueError('Surface pressure > 1100 hPa')
            if ps_int < MIN_SURF_PRESSURE:
                raise ValueError('Surface pressure < 500 hPa')
            for ps_index in np.arange(0,600):
                if ps_int == surface_pressure[ps_index]:
                    break
            wt_func  = weighting_function[:, ps_index]
            surf_wt  = surface_weight[ps_index]
            space_wt = space_weight[ps_index]

            above_surf = np.where(levels < ps_int)
            lowest_level = above_surf[0][-1]

            p_lower = ps_int
            p_upper = levels[lowest_level]
            index_lower = np.where(p_lower >= pressure)[0][0]
            index_upper = np.where(p_upper >= pressure)[0][0]

            p_layer = p_lower - 0.5 - np.arange(0, index_upper - index_lower)
            T_lower_wt_temp = 1.0 - (np.log(p_layer / p_lower) / np.log(p_upper / p_lower))
            T_upper_wt_temp = np.log(p_layer / p_lower) / np.log(p_upper / p_lower)

            T_lower_wt = np.sum(T_lower_wt_temp * wt_func[index_lower:index_upper])
            T_upper_wt = np.sum(T_upper_wt_temp * wt_func[index_lower:index_upper])

            surf_wt = surf_wt + T_lower_wt
            wt_ref = np.zeros((num_levels))
            wt_ref[lowest_level] += T_upper_wt

            # Now step through the rest of the levels
            for i in np.arange(lowest_level, 0, -1):
                p_lower = levels[i]
                p_upper = levels[i - 1]
                index_lower = np.where(p_lower >= pressure)[0][0]
                index_upper = np.where(p_upper >= pressure)[0][0]

                if index_upper > index_lower:
                    p_layer = p_lower - 0.5 - np.arange(0, index_upper - index_lower)
                    T_lower_wt_temp = 1 - (np.log(p_layer / p_lower) / np.log(p_upper / p_lower))
                    T_upper_wt_temp = np.log(p_layer / p_lower) / np.log(p_upper / p_lower)
                    T_lower_wt = np.sum(T_lower_wt_temp * wt_func[index_lower:index_upper])
                    T_upper_wt = np.sum(T_upper_wt_temp * wt_func[index_lower:index_upper])
                else:
                    T_lower_wt = 0.0
                    T_upper_wt = 0.0
                # print(i, T_lower_wt, T_upper_wt)
                wt_ref[i - 1] = wt_ref[i - 1] + T_upper_wt
                wt_ref[i] = wt_ref[i] + T_lower_wt

            p_lower = levels[0]
            p_upper = pressure[-1]
            index_lower = np.where(p_lower >= pressure)[0][0]
            index_upper = np.where(p_upper >= pressure)[0][0]

            # now the very top levels
            if ((index_upper > index_lower) and (index_upper > 0) and (index_lower > 0)):
                p_layer = p_lower - 0.5 - np.arange(0, index_upper - index_lower)
                T_lower_wt_temp = 1 - (np.log(p_layer / p_lower) / np.log(p_upper / p_lower))
                T_upper_wt_temp = np.log(p_layer / p_lower) / np.log(p_upper / p_lower)
                t_lower_wt = np.sum(T_lower_wt_temp * wt_func[index_lower:index_upper])
                t_upper_wt = np.sum(T_upper_wt_temp * wt_func[index_lower:index_upper])
            else:
                t_lower_wt = 0.0
                t_upper_wt = 0.0

            # for this last level, we include the weight from the level all the way to
            # the highest pressure

            wt_ref[0] = wt_ref[0] + t_lower_wt
            wt_ref[0] = wt_ref[0] + t_upper_wt

            level_wts[:,ilat,ilon] = wt_ref
            surface_wts[ilat,ilon] = surf_wt
            space_wts[ilat,ilon]   = space_wt

            tbs[ilat,ilon] = ts[ilat,ilon]*surf_wt + np.sum(temp_profiles[:,ilat,ilon]*wt_ref) + space_wt*2.730

    return tbs,level_wts,surface_wts,space_wts

class AtmWt():
    '''Class for level weights for MSU/AMSU'''

    def __init__(self, channel='TLT',surface = 'ocean',RTM_Data_Path='./data/',verbose=True):

        path = RTM_Data_Path + 'wt_tables/'
        nc_file = path + 'std_atmosphere_wt_function_msu_chan_'+channel+'_'+surface+'_by_surface_pressure.1100.V4.nc'
        if verbose:
            print('Reading: ' + nc_file)
        nc_fid = Dataset(nc_file, 'r')
        self.surface = surface
        pressure = np.array(nc_fid.variables['pressure'][:])  # extract/copy the data
        self.pressure = pressure
        pressure_bounds = np.array(nc_fid.variables['pressure_bounds'][:])
        self.pressure_bounds = pressure_bounds
        surface_pressure = np.array(nc_fid.variables['surface_pressure'][:])
        surface_pressure = surface_pressure.astype(np.int32)
        self.surface_pressure = surface_pressure
        surface_weight = np.array(nc_fid.variables['surface_weight'][:])
        self.surface_weight = surface_weight

        space_weight = np.array(nc_fid.variables['space_weight'][:])
        self.space_weight = space_weight
        weighting_function = np.array(nc_fid.variables['weighting_function'][:,:])
        self.weighting_function = weighting_function

    def AtmLevelWts(self, temp_profiles = None, ps = None, ts = None,levels = None,land_frac=None):

        '''

        :param temp_profiles: numpy array of temperature profiles to be converted to MSU equivalent  Assumed to be 3D [levels,lat,lon]
                                                                                                             Assumed to ordered low P to high P
                                                                                                             Units = K
        :param ps:            numpy array containing surface pressure to be  converted.  Assumed to be 2D [lat,lon].
                                                                                         Units = hPa
        :param ts:            numpy array containing surface temperature to be converted.  Assumed to be 2D [lat,lon].
                                                                                           Units K
        :param levels:        numpy array of level pressures for the profiles in temp proffiles.  Assumed to ordered low P to high P.
                                                                                                  Units = hPa
        :param land_frac:     numpy array of land fraction.  Assumed to be 2D [lat,lon].
                                                             Units = fraction, 0.0 to 1.0
        :return values:
                tbs          numpy array of MSU equivalent brightness temperatures, 2D, [lat,lon]
                level_wts    numpy array of level weights, 3D, [level_index,lat,lon]
                surface_wts  numpy array of surface weights, 2D, [lat,lon]
                space_wts    numpy array of "space" weights, 2D, [lat,lon].  Multiplied by 2.73K in routine
        '''

        sz1 = temp_profiles.shape
        sz2 = ps.shape
        sz3 = levels.shape

        try:
            assert(sz1[0] == sz3[0])
            assert(sz1[1] == sz2[0])
            assert(sz1[2] == sz2[1])
        except:
            raise ValueError('Array sizes do not match  in AtmLevelWts')

        tbs,level_wts,surface_wts,space_wts = AtmLevelWts_Numba(weighting_function = self.weighting_function,
                                          surface_weight = self.surface_weight,
                                          space_weight = self.space_weight,
                                          pressure = self.pressure,
                                          surface_pressure = self.surface_pressure,
                                          pressure_bounds = self.pressure_bounds,
                                          temp_profiles = temp_profiles,
                                          ts = ts,
                                          num_lats = sz1[1],
                                          num_lons = sz1[2],
                                          num_levels = sz1[0],
                                          ps = ps,
                                          levels = levels,
                                          land_frac = land_frac,
                                          surface = self.surface)
        return tbs,level_wts,surface_wts,space_wts

if __name__ == '__main__':

    from era5_monthly import read_era5_monthly_means_3D, read_era5_monthly_means_2D
    from global_map import global_map

    import xarray as xr
    year = 2000
    month = 1
    channel = 'TLS'
    use_t2m = True
    perform_profile = True
    d = read_era5_monthly_means_3D(year=year, month=month, variable='temperature',
                                   era5_path='./era5/monthly/3D/')
    t = (d['T'][0, :, :, :]).astype(np.float32)        #Temperature in Kelvin
    levels = (d['levels'][:]).astype(np.float32)       #Pressure Levels in hPa


    ps = read_era5_monthly_means_2D(year=year, month=month, variable='surface_pressure',
                                    era5_path='./era5/monthly/2D/')['PS'][0, :, :]
    #ps in ERA5 is in Pa
    #convert to hPa
    ps = (ps/100.0).astype(np.float32)

    if use_t2m:   #both types of ts are in Kelvin
        ts = read_era5_monthly_means_2D(year=year, month=month, variable='2m_temperature',
                                    era5_path='./era5/monthly/2D/')['T2m'][0, :, :]
        #this ts is the 2m air temperature.
        #  advantages of use:  more closelt tied to observations
        #  disadvantage:  Not really what the satellite sees.  The difference between T2m and Tskin can be large
        #                 under daytime and night time clear sky conditions
    else:
        ts  = read_era5_monthly_means_2D(year=year, month=month, variable='skin_temperature',
                                    era5_path='./era5/monthly/2D/')['TSkin'][0, :, :]
        # this ts is the skin temperature as relevant to long-wave IR
        # advantage of use:  closer to microwave skin temperature, at least for moist soils
        # disadvantage:  appears to be a sort of free parameter in the model which is adjusted to satisfy
        #                energy balance
        #                for dry soil, Longwave IR and MW penetration depth are very different.

    ts = ts.astype(np.float32)

    land_frac = read_era5_monthly_means_2D(year=year, month=month, variable = 'land_sea_mask',
                                     era5_path='./era5/monthly/2D/')['land_frac'][0, :, :]

    sea_ice_frac = read_era5_monthly_means_2D(year=year, month=month, variable = 'sea_ice_cover',
                                     era5_path='./era5/monthly/2D/')['SeaIce'][0, :, :]

    sea_ice_frac[np.isnan(sea_ice_frac)] = 0.0
    sea_ice_frac[sea_ice_frac < -1.0] = 0.0
    not_ocean = land_frac + sea_ice_frac
    not_ocean[not_ocean > 1.0] = 1.0

    #initialize AtmWt classes.
    AtmWt_MSU_Ocean = AtmWt(channel = channel,surface = 'ocean')
    AtmWt_MSU_Land  = AtmWt(channel = channel,surface = 'land')

    if perform_profile:
        #start the profiler
        pr = cProfile.Profile()
        pr.enable()

    # calculate the Tbs and  weights.
    print('Performing Ocean Calculation')
    tbs_ocean,level_wts_ocean,surface_wts_ocean,space_wts_ocean = AtmWt_MSU_Ocean.AtmLevelWts(temp_profiles=t,ts = ts, ps=ps, levels=levels,land_frac=not_ocean)
    print('Performing Land Calculation')
    tbs_land,level_wts_land, surface_wts_land, space_wts_land   = AtmWt_MSU_Land.AtmLevelWts(temp_profiles=t,ts = ts, ps=ps, levels=levels,land_frac=not_ocean)

    if perform_profile:
        # end profiled section
        pr.disable()

        #report the profiles info.
        s = io.StringIO()
        sortby = SortKey.TIME
        ps1 = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps1.print_stats(0.01)
        print(s.getvalue())

    # combine the land and ocean results together.
    surface_wts_combined = not_ocean * surface_wts_land + (1.0 - not_ocean)*surface_wts_ocean
    tbs_combined = not_ocean * tbs_land + (1.0 - not_ocean)*tbs_ocean

    # adjust to match RSS plotting routine
    not_ocean_to_plot = np.roll(np.flipud(not_ocean),shift = 180,axis=(1))
    surface_wts_land   = np.roll(np.flipud(surface_wts_land),shift = 180,axis=(1))
    surface_wts_ocean   = np.roll(np.flipud(surface_wts_ocean),shift = 180,axis=(1))
    surface_wts_combined = np.roll(np.flipud(surface_wts_combined),shift=180,axis=1)

    ps = np.roll(np.flipud(ps),shift=180,axis=1)
    ts = np.roll(np.flipud(ts),shift=180,axis=1)

    tbs_land = np.roll(np.flipud(tbs_land),shift=180,axis=1)
    tbs_ocean = np.roll(np.flipud(tbs_ocean),shift=180,axis=1)
    tbs_combined = np.roll(np.flipud(tbs_combined),shift=180,axis=1)

    # make plots for sanity check.
    global_map(surface_wts_combined,vmin = 0.0,vmax = 0.3,plt_colorbar = True,title = channel+' surface weights')
    global_map(tbs_combined,vmin = 190.0,vmax = 240.0,plt_colorbar = True,title = channel+' Brightness Temperature(K)')
    global_map(not_ocean_to_plot,vmin = 0.0,vmax = 1.0,plt_colorbar = True,title = channel+' land and  ice fraction')
    plt.show()

    # read in data from IDL calculation
    compare_file_name = f"./test/ERA5_test_case_msu_channel_{channel}_{year:04}_{month:02}.nc"
    d = xr.open_dataset(compare_file_name)
    #roll it in longitude by 180 degrees to match the python  maps
    tbs_from_idl = np.roll(d['Tb'].values, shift=180, axis=1)
    tbs_land_from_idl = np.roll(d['Tb_land'].values, shift=180, axis=1)
    tbs_ocean_from_idl = np.roll(d['Tb_ocean'].values, shift=180, axis=1)
    not_ocean_idl = np.roll(d['land_fraction'].values, shift=180, axis=1)

    global_map(tbs_combined,vmin = 220.0,vmax = 280.0,plt_colorbar = True,title = channel+' Brightness Temperature(K), from Python')
    global_map(tbs_from_idl,vmin = 220.0,vmax = 280.0,plt_colorbar = True,title = channel+' Brightness Temperature(K), from IDL')
    global_map(tbs_combined - tbs_from_idl,vmin  =-0.02,vmax = 0.02,plt_colorbar = True,title = channel+' Brightness Temperature(K), from IDL')

    land_diff = tbs_land - tbs_land_from_idl
    ocean_diff = tbs_ocean - tbs_ocean_from_idl
    comb_diff = tbs_combined - tbs_from_idl

    print(np.min(land_diff),np.max(land_diff),np.mean(land_diff),np.std(land_diff))
    print(np.min(ocean_diff),np.max(ocean_diff),np.mean(ocean_diff),np.std(ocean_diff))
    print(np.min(comb_diff),np.max(comb_diff),np.mean(comb_diff),np.std(comb_diff))

    print()
    plt.show()
    print




    print



