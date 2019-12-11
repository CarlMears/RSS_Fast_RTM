# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 13:58:21 2016

@author: mears
"""
import numpy as np
from netCDF4 import Dataset

def read_eraint_atmos(year,month,time_step,eraint_path = '//ops1p/n/data/model/ERA-Int/daily/'):
    
    eraint_file = eraint_path + 'atmos_profile/' + str(year) + '/'+ 'ERA_Int_Daily_Profiles_'+str(year) + '_' + str(month).zfill(2) + '.nc'
    
    print 'Reading: '+eraint_file
    nc_fid = Dataset(eraint_file, 'r') 
    temperature       = np.array(nc_fid.variables['t'][time_step,:,:,:]).clip(min=0.0)  # extract/copy the data
    spec_hum          = np.array(nc_fid.variables['q'][time_step,:,:,:]).clip(min=0.0)
    cloud_water       = np.array(nc_fid.variables['clwc'][time_step,:,:,:]).clip(min=0.0)
    lat               = np.array(nc_fid.variables['latitude'][:])
    lon               = np.array(nc_fid.variables['longitude'][:])
    time              = np.array(nc_fid.variables['time'][time_step])
    level             = np.array(nc_fid.variables['level'][:])
    geopotent         = np.array(nc_fid.variables['z'][time_step,:,:,:])
    nc_fid.close()
    eraint_profiles = dict([('t', temperature), ('q', spec_hum), ('clwc', cloud_water),('lat',lat),('lon',lon),('time',time),('level',level),('z',geopotent)])
    
    return eraint_profiles
    
def read_eraint_surf(year,month,time_step,eraint_path = '//ops1p/n/data/model/ERA-Int/daily/'):
    
    eraint_file = eraint_path + 'surface/' + str(year) + '/'+ 'ERA_Int_surface_data_'+ str(month).zfill(2) +  '_'+str(year)+'_480x241.nc'
      
    print 'Reading: '+eraint_file
    nc_fid = Dataset(eraint_file, 'r') 
    
    ci                 = np.array(nc_fid.variables['ci'][time_step,:,:])         #sea ice cover
    lat                = np.array(nc_fid.variables['latitude'][:])
    lon                = np.array(nc_fid.variables['longitude'][:])
    skt                = np.array(nc_fid.variables['skt'][time_step,:,:]).clip(min=0.0)        # skin temperature
    sp                 = np.array(nc_fid.variables['sp'][time_step,:,:]).clip(min=0.0)         # surface pressure
    sst                = np.array(nc_fid.variables['sst'][time_step,:,:]).clip(min=0.0)
    t2m                = np.array(nc_fid.variables['t2m'][time_step,:,:]).clip(min=0.0)
    tcw                = np.array(nc_fid.variables['tcw'][time_step,:,:]).clip(min=0.0)
    tcwv               = np.array(nc_fid.variables['tcwv'][time_step,:,:]).clip(min=0.0)
    time               = np.array(nc_fid.variables['time'][:])
    u10                = np.array(nc_fid.variables['u10'][time_step,:,:]).clip(min=0.0)
    v10                = np.array(nc_fid.variables['v10'][time_step,:,:]).clip(min=0.0)
    w10 = np.sqrt(u10*u10 + v10*v10)
    
    eraint_surf = dict([('ci', ci), ('skt', skt), ('sp', sp),('sst',sst),('t2m',t2m),('tcw',tcw),('tcwv',tcwv),('time',time),('u10',u10),('v10',v10),('w10',w10),('lat',lat),('lon',lon)])
    
    return eraint_surf

def read_eraint_invar(eraint_path = '//ops1p/n/data/model/ERA-Int/daily/'):
    
    eraint_file = eraint_path +'surface/' + 'ERA_Int_surface_invariant_480x241.nc'
      
    print 'Reading: '+eraint_file
    nc_fid = Dataset(eraint_file, 'r') 
    
    lsm = np.array(nc_fid.variables['lsm'][0,:,:])         #land sea mask
    z_surf = np.array(nc_fid.variables['z'][0,:,:])
    
    eraint_invar = dict([('lsm',lsm),('z_surf',z_surf)])
    
    return eraint_invar

    

      
