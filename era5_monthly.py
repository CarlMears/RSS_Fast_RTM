def read_era5_monthly_means_3D(year=1979,month=1,variable = 'temperature',era5_path  = 'A:/ERA5/monthly/temperature_3D/'):
    import numpy as np
    from netCDF4 import Dataset

    short_names = {
        'temperature': 'T',
        'specific humidity': 'Q',
        'specific cloud liquid water content': 'CLD'
    }

    ecmwf_names = {
        'temperature': 't',
        'specific humidity': 'q',
        'specific cloud liquid water content': 'q'
    }

    try:
        short_name = short_names[variable]
    except:
        raise ValueError('variable not in short name dictionary')

    try:
        ecmwf_name = ecmwf_names[variable]
    except:
        raise ValueError('variable not in ecmwf name dictionary')

    year_str = str(year).zfill(4)
    month_str = str(month).zfill(2)
    nc_file = era5_path + year_str+'/'+'era5_'+short_name+'_3D_360_181_'+year_str+'_'+month_str+'.nc'

    nc_fid = Dataset(nc_file, 'r')
    data = np.array(nc_fid.variables[ecmwf_name])
    lons = np.array(nc_fid.variables['longitude'])
    lats = np.array(nc_fid.variables['latitude'])
    levels = np.array(nc_fid.variables['level'])

    d = {short_name : data,
         'lats' : lats,
         'lons' : lons,
         'levels': levels,
         'name' : short_name}

    return d

def read_era5_monthly_means_2D(year=1979, month=1, variable='temperature',
                                            era5_path='A:/ERA5/monthly/2D/'):

    import numpy as np
    from netCDF4 import Dataset

    short_names = {
        'surface_pressure':'PS',
        '2m_temperature': 'T2m',
        'skin_temperature':'TSkin',
        '10m_wind_speed': 'W10',
        'sea_ice_cover': 'SeaIce',
        'land_sea_mask': 'land_frac',
        'total_column_water_vapour':'TPW'
    }

    ecmwf_names = {
        'surface_pressure': 'sp',
        '2m_temperature': 't2m',
        'skin_temperature': 'skt',
        '10m_wind_speed': 'si10',
        'sea_ice_cover': 'siconc',
        'land_sea_mask': 'lsm',
        'total_column_water_vapour': 'tcwv'
    }


    try:
        short_name = short_names[variable]
    except:
        raise ValueError('variable not in short name dictionary')

    try:
        ecmwf_name = ecmwf_names[variable]
    except:
        raise ValueError('variable not in ecmwf name dictionary')

    year_str = str(year).zfill(4)
    month_str = str(month).zfill(2)
    nc_file = era5_path + year_str + '/' + 'era5_' + short_name + '_2D_360_181_' + year_str + '_' + month_str + '.nc'

    nc_fid = Dataset(nc_file, 'r')
    data = np.array(nc_fid.variables[ecmwf_name])
    lons = np.array(nc_fid.variables['longitude'])
    lats = np.array(nc_fid.variables['latitude'])


    d = {short_name : data,
         'lats' : lats,
         'lons' : lons,
         'name' : short_name}

    return d


def download_and_save_era5_monthly_means_3D(year=1979,month=1,variable = 'temperature',output_path  = 'A:/ERA5/monthly/3D/'):
    import cdsapi
    from os import makedirs
    c = cdsapi.Client()

    short_names = {
        'temperature': 'T',
        'specific humidity': 'Q',
        'specific cloud liquid water content': 'CLD'
    }

    try:
        short_name = short_names[variable]
    except:
        raise ValueError('variable not in short name dictionary')

    year_str = str(year).zfill(4)
    month_str = str(month).zfill(2)
    nc_file = output_path + year_str+'/'+'era5_'+short_name+'_3D_360_181_'+year_str+'_'+month_str+'.nc'
    path = output_path + year_str + '/'
    makedirs(path,exist_ok = True)

    c.retrieve(
        'reanalysis-era5-pressure-levels-monthly-means',
        {
            'format': 'netcdf',
            'product_type': 'monthly_averaged_reanalysis',
            'variable': variable,
            'pressure_level': [
                '1', '2', '3',
                '5', '7', '10',
                '20', '30', '50',
                '70', '100', '125',
                '150', '175', '200',
                '225', '250', '300',
                '350', '400', '450',
                '500', '550', '600',
                '650', '700', '750',
                '775', '800', '825',
                '850', '875', '900',
                '925', '950', '975',
                '1000',
            ],
            "grid": "1.0/1.0",
            'year': year_str,
            'month': month_str,
            'time': '00:00',
        },nc_file)


def download_and_save_era5_monthly_means_2D(year=1979, month=1, variable='temperature',
                                            output_path='A:/ERA5/monthly/2D/'):
    import cdsapi
    from os import makedirs
    c = cdsapi.Client()

    short_names = {
        'surface_pressure':'PS',
        '2m_temperature': 'T2m',
        'skin_temperature':'TSkin',
        '10m_wind_speed': 'W10',
        'sea_ice_cover': 'SeaIce',
        'land_sea_mask': 'land_frac',
        'total_column_water_vapour':'TPW',
        'total_precipitation':'TotPr'
    }

    try:
        short_name = short_names[variable]
    except:
        raise ValueError('variable not in short name dictionary')

    year_str = str(year).zfill(4)
    month_str = str(month).zfill(2)
    nc_file = output_path + year_str + '/' + 'era5_' + short_name + '_2D_360_181_' + year_str + '_' + month_str + '.nc'
    path = output_path + year_str + '/'
    makedirs(path, exist_ok=True)

    c.retrieve(
        'reanalysis-era5-single-levels-monthly-means',
        {
            'format': 'netcdf',
            'product_type': 'monthly_averaged_reanalysis',
            'variable': variable,
            'year': year_str,
            'month': month_str,
            "grid": "1.0/1.0",
            'time': '00:00',
        }, nc_file)