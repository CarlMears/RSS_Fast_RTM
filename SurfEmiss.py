
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import interp1d
import math
from netCDF4 import Dataset
from numba import jit

RTM_Data_Path = '//sounder/c/idl_library/MSU_AMSU_simulation/method_2_for_LLNL/data/'

@jit(nopython=True)
def InterpOceanEmissivityFast(T,W,fov,emiss_table,T0,Delta_T,Delta_W):
    t_scaled = (T-T0)/Delta_T
    t1 = np.int32(np.floor(t_scaled))
    wt = np.float32(t_scaled - t1)
    w_scaled = W/Delta_W
    if w_scaled > 30.0 :
        w1 = np.int32(29)
        ww = 1.0
    else:
        w1 = np.int32(np.floor(w_scaled))
        ww = np.float32(w_scaled-w1)
        
    emiss_interp = (1.0-wt)*((1.0 - ww)*emiss_table[fov,  w1,t1] + ww*emiss_table[fov,  w1+1,t1]) + \
                        wt *((1.0 - ww)*emiss_table[fov,w1,t1+1] + ww*emiss_table[fov,w1+1,t1+1])
    return emiss_interp
    
@jit(nopython=True)
def InterpSeaIceEmissivityFast(T,fov,emiss_table,T0,Delta_T):
    t_scaled = (T-T0)/Delta_T
    t1 = np.int32(np.floor(t_scaled))
    wt = np.float32(t_scaled - t1)
        
    emiss_interp = (1.0-wt)*emiss_table[fov,t1] + wt *emiss_table[fov,t1+1]
    return emiss_interp

class SurfEmiss():
    '''Class for calculating surface emissivity'''
    
    def __init__(self,channel = 2,RTM_Data_Path = '//sounder/c/idl_library/MSU_AMSU_simulation/method_2_for_LLNL/data/'):
        
        path = RTM_Data_Path +'emiss_tables/'
        ocean_nc_file = path + 'msu_'+str(channel)+'_emiss_table_W.nc'
        print('Reading: '+ocean_nc_file)
        nc_ocean_emiss_fid = Dataset(ocean_nc_file, 'r') 
        
        temperature               = np.array(nc_ocean_emiss_fid.variables['temperature'][:])  # extract/copy the data
        self.temperature_ocean    = temperature
        wind                      = np.array(nc_ocean_emiss_fid.variables['wind'][:])
        self.wind_ocean                 = wind
        fov                       = np.array(nc_ocean_emiss_fid.variables['fov'][:])
        self.fov_ocean            = fov
        self.emissivity_ocean     = np.array(nc_ocean_emiss_fid.variables['emissivity'][:])
        nc_ocean_emiss_fid.close()
        
        self.T0_ocean = self.temperature_ocean[0]
        self.Delta_T_ocean = self.temperature_ocean[1]-self.temperature_ocean[0]
        self.Delta_W_ocean = self.wind_ocean[1] - self.wind_ocean[0]
        
        
        self.emissivity_ocean_interpolating_function = RegularGridInterpolator((self.fov_ocean,self.wind_ocean, self.temperature_ocean), self.emissivity_ocean)


        
        path = RTM_Data_Path +'emiss_tables/'
        sea_ice_nc_file             = path + 'msu_'+str(channel)+'_emiss_table_sea_ice.nc'
        print('Reading: '+sea_ice_nc_file)
        nc_sea_ice_emiss_fid      = Dataset(sea_ice_nc_file, 'r') 
        
        temperature               = np.array(nc_sea_ice_emiss_fid.variables['temperature'][:])  # extract/copy the data
        self.temperature_sea_ice  = temperature
        fov                       = np.array(nc_sea_ice_emiss_fid.variables['fov'][:])
        self.fov_sea_ice          = fov
        self.emissivity_sea_ice   = np.array(nc_sea_ice_emiss_fid.variables['emissivity'][:])
        nc_sea_ice_emiss_fid.close()
        
        self.T0_sea_ice = self.temperature_sea_ice[0]
        self.Delta_T_sea_ice = self.temperature_sea_ice[1]-self.temperature_sea_ice[0]

        self.emissivity_sea_ice_interpolating_function = RegularGridInterpolator((self.fov_sea_ice,self.temperature_sea_ice), self.emissivity_sea_ice)

        
    def OceanEmissivity(self,fov,temperature,wind_speed):
        emiss = self.emissivity_ocean_interpolating_function((fov,temperature,wind_speed))
        return emiss
         
    def OceanEmissivityFast(self,fov,temperature,wind_speed):
        emiss = InterpOceanEmissivityFast(temperature,wind_speed,fov,self.emissivity_ocean,self.T0_ocean,self.Delta_T_ocean,self.Delta_W_ocean)
        return emiss
         
    def SeaIceEmissivity(self,fov,temperature):
         emiss = self.emissivity_sea_ice_interpolating_function((fov,temperature))
         return emiss
         
    def SeaIceEmissivityFast(self,fov,temperature):
        emiss = InterpSeaIceEmissivityFast(temperature,fov,self.emissivity_sea_ice,self.T0_sea_ice,self.Delta_T_sea_ice)
        return emiss

         
    
  