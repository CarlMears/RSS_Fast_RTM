# -*- coding: utf-8 -*-
'''
Test Atmospheric Profiles
'''
from AtmosProfile import AtmosProfile
import numpy as np
from AtmAbs import AtmAbs
from AtmAbs import CldAbs
from RSS_RTM import RSS_RTM_2
from SurfEmiss import SurfEmiss
from eraint import read_eraint_atmos,read_eraint_surf,read_eraint_invar
from numba import jit
import cProfile, pstats, io
from pstats import SortKey
from global_map import global_map
import matplotlib.pyplot as plt


# initialize the absorption and emissivity classes with the MSU channel

def calc_TMT_map(eraint_surf,eraint_prof,eraint_invar):
    
    #initialize the surface emissivity class
    SurfEmiss_2 = SurfEmiss(channel = 4,RTM_Data_Path = './data/')
    
    #initialize the RTM.  This initializes the absorption classes
    RTM = RSS_RTM_2(channel = 4,RTM_Data_Path = './data/')
    msu_theta = np.array([0.0,10.71,21.51,32.51,43.91,56.19])
    
    tmt_wts = np.array([1.0/9.0,2.0/9.0,2.0/9.0,2.0/9.0,2.0/9.0,0.0])
    
    tb_array = np.zeros((6))
    tb_array_combined_surf = np.zeros((6))
            
    tbup_array = np.zeros((6))        
    tbdw_array = np.zeros((6))
    
    tmt_map = np.zeros_like(eraint_invar['z_surf'])
    z = eraint_prof['z']
    t = eraint_prof['t']
    q = eraint_prof['q']
    p = eraint_prof['level']
    
    for i,lat in enumerate(eraint_surf['lat']):
        print(lat)
        for j,lon in enumerate(eraint_surf['lon']):
    
            Z = np.flipud(z[:,i,j])/9.80665 # convert to geopotential height
            T = np.flipud(t[:,i,j])
            Q = np.flipud(q[:,i,j])
            P = np.flipud(p)
            
            Z_surf = eraint_invar['z_surf'][i,j]/9.80665
            Land_frac = eraint_invar['lsm'][i,j]
            Ocean_frac = 1.0 - Land_frac
            Seaice_frac = eraint_surf['ci'][i,j]
            if Seaice_frac > 0.00001:
                Seaice_frac = Seaice_frac * Ocean_frac
                Ocean_frac = Ocean_frac - Seaice_frac
            
            # truncate profiles at the surface pressure
            PS = 0.01 * eraint_surf['sp'][i,j]
    
            w = np.where(P < PS)
    
            Z = Z[w]
            T = T[w]
            Q = Q[w]
            P = P[w]
            
            num_layers = len(Z)
    
            # create the profile
            test_prof = AtmosProfile(num_layers = num_layers,name = 'test1')
            
            # add the standard atmosphere data
            
            test_prof.addprofiledata(Z,name='Z')
            test_prof.addprofiledata(P,name='P')
            test_prof.addprofiledata(T,name='T')
            test_prof.addprofiledata(Q,name='Q')
    
            #extrapolate the standard atmosphere profile to the surface at z = 0.0
         
            test_prof.extrapolate_to_surface(z_surf = Z_surf)
    
            #no clouds for now
            test_prof.set_cloud_zero()
    
            #extrapolate the vapor and the clouds to the surface
            test_prof.extrapolate_to_surface(z_surf = Z_surf)
    
            # set the wind speed (needed for ocean emissivity)
            test_prof.addsurfacedatum(eraint_surf['w10'][i,j],name='WS')
            
            # set the surface temperature
            test_prof.addsurfacedatum(eraint_surf['skt'][i,j],name='TS')
    
            # figure out which surfaces to do the calculation for...
            surfaces_to_do = []
            weights_to_do = []
            if Land_frac > 0.000001:
                surfaces_to_do.append('Land')
                weights_to_do.append(Land_frac)
            if Seaice_frac > 0.000001:
                surfaces_to_do.append('Ice')
                weights_to_do.append(Seaice_frac)
            if Ocean_frac > 0.000001:
                surfaces_to_do.append('Ocean')
                weights_to_do.append(Ocean_frac)        
    
            if abs(-1.0 + np.array(weights_to_do).sum()) > 0.0001:
                print ('Weight Error')
                for weight_index,weight in enumerate(weights_to_do):
                    weights_to_do[weight_index] = weight/np.array(weights_to_do).sum()
                    
            
            tb_array_combined_surf = np.zeros((6))
            
            for surf_index,surf_string in enumerate(surfaces_to_do):
                test_prof.definesurface(surf_string = surf_string)
                for fov in range(0,6):
                    #calculate the surface emissivity for this fov
                    test_prof.calc_emissivity(SurfEmiss_2,fov = fov)
                    
                    # perform the RTM
                    tbdw,tbup,tb = RTM.CalcTb(test_prof,
                                              theta = msu_theta[fov])
                    tb_array[fov] = tb
                    tbup_array[fov] = tbup
                    tbdw_array[fov] = tbdw
                    
                tb_array_combined_surf = tb_array_combined_surf + weights_to_do[surf_index]*tb_array

            
            '''for fov in range(0,6):
                #test_prof.definesurface(surf_string = 'Ocean')
                #calculate the surface emissivity for this fov
                test_prof.calc_emissivity_surface_type_weighted(SurfEmiss_2,fov=0,
                     surface_wts = np.array([Ocean_frac,Seaice_frac,Land_frac],dtype=np.float32))  
        
                # perform the RTM
                tbdw,tbup,tb = RTM.CalcTb(test_prof,theta = msu_theta[fov])
                tb_array[fov] = tb
                tbup_array[fov] = tbup
                tbdw_array[fov] = tbdw'''
                
 
            tmt  = np.sum(tb_array_combined_surf*tmt_wts)        
            tmt_map[i,j] = tmt  
    return tmt_map
      
if __name__ == '__main__':      
    eraint_surf = read_eraint_surf(2003,1,0,eraint_path = './daily/')
    eraint_prof = read_eraint_atmos(2003,1,0,eraint_path = './daily/')
    eraint_invar = read_eraint_invar(eraint_path = './daily/')
    
    from time import time
    t = time() 
    tmt_map = calc_TMT_map(eraint_surf,eraint_prof,eraint_invar)
    deltat = time() - t
    print (deltat)
    
    global_map(np.flipud(tmt_map), vmin=200.0, vmax=270.0, plt_colorbar=True,title='')
    plt.show()