import numpy as np
#from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import interp1d
from netCDF4 import Dataset
from numba import jit
import math

@jit(nopython=True)
def find_abs_q(profiles,abs_table_q,T0,Delta_T,Delta_P,Delta_Q):
    abs_interp_arr = np.zeros(len(profiles))
    
    for level_index,level_data in enumerate(profiles):
        q = np.float32(level_data[0])
        p = np.float32(level_data[1])
        t = np.float32(level_data[2])

        
        #abs_table_q = absorptivity
        sz = abs_table_q.shape
          
        t_scaled = (t-T0)/Delta_T
        t1 = np.int32(np.floor(t_scaled))
        if t1 < 0:
            t1 = 0
        if t1 > (sz[2]-2):
            t1 = sz[2]-2 
        wt = t_scaled-t1
    
        p_scaled = p/Delta_P
        p1 = np.int32(np.floor(p_scaled))
        wp = p_scaled-p1
        if p1 < 0:
           p1=0
        if p1 > (sz[1]-2):
           p1 = sz[1]-2
    
    
        q_scaled = q/Delta_Q
        q1 = np.int32(np.floor(q_scaled))
        if q1 < 0 :
            q1 = 0
        if q1 > (sz[0]-2):
            q1 = sz[0]-2
        wq = q_scaled-q1
        
        #abs_interp = np.float32((1.0-wt)*((1.0-wp)*((1.0-wq)*abs_table_q[q1,p1,    t1] + wq*abs_table_q[q1+1,p1,  t1]) + \
        #                             wp*((1.0-wq)*abs_table_q[q1,p1+1,  t1] + wq*abs_table_q[q1+1,p1+1,t1])) + \
        #                   wt*((1.0-wp)*((1.0-wq)*abs_table_q[q1,p1,  t1+1] + wq*abs_table_q[q1+1,p1,  t1+1]) + \
        #                             wp*((1.0-wq)*abs_table_q[q1,p1+1,t1+1] + wq*abs_table_q[q1+1,p1+1,t1+1])))
        
        #I think this is faster than the above....
        abs_table_q_sub = abs_table_q[q1:q1+2,p1:p1+2,t1:t1+2]
        
        abs_interp = np.float32((1.0-wt)*((1.0-wp)*((1.0-wq)*abs_table_q_sub[0,0,0] + wq*abs_table_q_sub[1,0,0]) + \
                                     wp*((1.0-wq)*abs_table_q_sub[0,1,0] + wq*abs_table_q_sub[1,1,0])) + \
                           wt*((1.0-wp)*((1.0-wq)*abs_table_q_sub[0,0,1] + wq*abs_table_q_sub[1,0,1]) + \
                                     wp*((1.0-wq)*abs_table_q_sub[0,1,1] + wq*abs_table_q_sub[1,1,1])))
                                     
        abs_interp_arr[level_index] = abs_interp
                                  
    return abs_interp_arr

@jit(nopython=True)
def calc_cld_Absorptivity(abs_interp,L):
     abs_interp_np_m = abs_interp*L*0.1
     return abs_interp_np_m
 

class AtmAbs():
    '''Class for calculating Oxygen and Water Vapor Absorption'''
    
    def __init__(self,channel = 2,RTM_Data_Path = './data/'
):
        
        
        path = RTM_Data_Path +'abs_tables/'
        nc_file = path + 'msu_'+str(channel)+'_abs_table_q_per_Pa.nc'
        print 'Reading: '+nc_file
        nc_fid = Dataset(nc_file, 'r') 
        
        temperature       = np.array(nc_fid.variables['temperature'][:])  # extract/copy the data
        self.temperature  = temperature
        pressure          = np.array(nc_fid.variables['pressure'][:])
        self.pressure     = pressure
        spec_hum          = np.array(nc_fid.variables['specific_humidity'][:])
        self.spec_hum     = spec_hum
        
        absorptivity = np.array(nc_fid.variables['absorptivity'][:])
        self.absorptivity = absorptivity
        
        # now set up the interpolator
        
        #self.absorptivity_interpolating_function = RegularGridInterpolator((self.spec_hum, self.pressure, self.temperature), self.absorptivity)
        
        # set up variables ofr numba versions
        self.T0 = self.temperature[0]
        self.Delta_T = self.temperature[1] - self.temperature[0]
        self.Delta_P = self.pressure[1] - self.pressure[0]
        self.Delta_Q = self.spec_hum[1] - self.spec_hum[0]
    

    def Absorptivity(self,atm_profiles):
        oxyvap_abs_arr = find_abs_q(atm_profiles,self.absorptivity,self.T0,self.Delta_T,self.Delta_P,self.Delta_Q)
        return oxyvap_abs_arr
    
        
class CldAbs():

    def __init__(self,channel = 2,RTM_Data_Path = './data/'
):
        path = RTM_Data_Path +'abs_tables/'
        nc_file = path + 'msu_'+str(channel)+'_cld_abs_table.nc'
        print 'Reading: '+nc_file
        nc_fid = Dataset(nc_file, 'r') 
        temperature       = np.array(nc_fid.variables['temperature'][:])  # extract/copy the data
        self.temperature  = temperature
        absorptivity = np.array(nc_fid.variables['absorptivity'][:])
        self.absorptivity = absorptivity

        # now set up the interpolator
        
        self.absorptivity_interpolating_function = interp1d(self.temperature, self.absorptivity)


     
    def Absorptivity(self,T,L): #T in Kelvin, L in Kg/m^3
        
        abs_interp = self.absorptivity_interpolating_function(T)
        
        # this returns the absorption coefficient for the (absurd) density of
        # 1 g/cm3
    
        abs_interp_np_m = calc_cld_Absorptivity(abs_interp,L)    # calculate absorption in nepers/cm
    
        return abs_interp_np_m


        
        


        
        
        
        

