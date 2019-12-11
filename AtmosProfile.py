

import numpy as np
import collections
import math
#import atmos

class AtmosProfile():
    '''Class that contains data about a single atmospheric profiles, along with surface data'''
    
     
    def __init__(self,num_layers = 30,name = 'Default'):
        self.data = collections.OrderedDict()
        self.surface_type = 'Ocean'
        self.num_layers = num_layers    
        self.allowed_profile_names = ['Z','P','T','Q','L']
        self.allowed_surface_names = ['ZS','TS','PS','QS','LS','WS','EM']
        self.allowed_surface_types = ['Land','Ice','Ocean']

        
    def __str__(self):
        '''Prints out info from Profile First prints out surface scalars Then prints the profile data'''
        return_string = 'An instance of class AtmosProfile containing the following\n'
        return_string += 'Surface Data:\n---------------------------------\n'
        return_string += 'Surface Type: '+self.surface_type+'\n'
        keylist = self.data.keys()
        for key in self.allowed_surface_names:
            if key in keylist:
                return_string += '{:>10}{:10.4f}'.format(key,self.data[key]) + '\n'
        num_profiles = 0
        profile_key_list = []
        for key in self.allowed_profile_names:
            if key in keylist:
                num_profiles += 1
                num_layers = len(self.data[key])
                profile_key_list.append(key)
        prof_array = np.ndarray([num_profiles,num_layers])
        for index,key in enumerate(profile_key_list):
            prof_array[index,:]=self.data[key]
        return_string += 'Profile Data:\n---------------------------------\n'
        for key in profile_key_list:
            return_string += '{:^12}'.format(key)
        return_string += '\n'
        for row in range(0,num_layers):
            for index,key in enumerate(profile_key_list):
                return_string += '{:12.6f}'.format(prof_array[index,row]) 
            return_string += '\n'
        return return_string
           
    def definesurface(self,surf_string = 'Ocean'):        
        ''' define the surface type for the profile '''
        if surf_string in self.allowed_surface_types:
            self.surface_type = surf_string
        else:
            print surf_string+' is not an allowed surface type'
            print 'surface type not changed'
        
        

    def addprofiledata(self,data,name = 'Default'):
        '''Add data to profile'''   
       # allowed_profile_names = ['Z','T','P','Q','L']
        if name in self.allowed_profile_names:
            if self.data.has_key(name):
                print 'Profile data with name = '+name+' already exists -- no data added'
                print 'Use replaceprofiledata instead'
            else:
                if isinstance(data,np.ndarray):
                    if len(data) == self.num_layers:
                        self.data[name] = data
                    else:
                        print 'data is of wrong length'
                        print 'no data added to profile'
                else:
                    print 'data is not an ndarray'
                    print 'no data added to profile'
        else:
            print name+' is not an allowed name'
            print 'no data added to profile'
            
    def addsurfacedatum(self,datum,name ='Default',replace = True):
        '''Add surface datum to profile'''
        #  allowed_surface_names = ['TS','PS','WS','EM']

        if name in self.allowed_surface_names:
            if (self.data.has_key(name) and not replace):
                print 'Surface data with name = '+name+' already exists -- no data added'
                print 'Use replacesurface datum instead'
            else:
                if np.isscalar(datum):
                    self.data[name] = datum
                else:
                    print 'datum is not a scalar'
                    print 'no surface datum added to profile'
        else:
            print name+' is not an allowed name'
            print 'no surface datum added to profile'
            
    def extrapolate_to_surface(self,z_surf = 0.0):
        '''adds surface data by extrapolating the profile data to Z = 0.0'''
        
        R_GAS    =  8.3145112   #6.0221367d23 * 1.380658d-23   ;J/mol/K
        M_W_AIR  =  2.8966e-2                    #kg/mol
        M_W_H2O  =  1.8015324e-2                 #kg/mol
        G_0      =  9.80665                      #N/kg

        if self.data.has_key('Z'):
            if self.data.has_key('T'):
                # extrapolate T to Z=0 assuming linear dependence of T on Z
                T_surface = self.data['T'][0] - (self.data['T'][1]-self.data['T'][0]) * (self.data['Z'][0] - z_surf)/(self.data['Z'][1]-self.data['Z'][0])
                self.addsurfacedatum(T_surface,'TS')
                self.addsurfacedatum(z_surf,'ZS')
                # extrapolate P to surface using mean layer temperature
                mean_layer_temp = 0.5*(self.data['TS'] + self.data['T'][0])
                dz = self.data['Z'][0] - z_surf
                arg = dz*(G_0/((R_GAS/M_W_AIR)*mean_layer_temp))
                p_surface = self.data['P'][0] * math.exp(arg)
                self.addsurfacedatum(p_surface,'PS')

                
                
            else:
                print 'Profile has no T variable'
                print 'Extrapolation not performed'
            
        else:
            print 'Profile has no Z variable'
            print 'Extrapolation not performed'
        if self.data.has_key('Q'):
            self.addsurfacedatum(self.data['Q'][0],'QS')
        if self.data.has_key('L'):
            self.addsurfacedatum(self.data['L'][0],'LS')
            
    
    def estimate_vapor(self,surface_RH = 0.8,scale_height = 1500.0):
        ''' adds estimated vapor to the profile based on RH at the surface and 
            an exponential decay of PV with height '''
            
        # Check to make sure T and Z are present
            
        if (self.data.has_key('Z') and self.data.has_key('TS') and self.data.has_key('PS')):
                
            #es = atmos.calculate('es', T = self.data['TS']) #
            es = 0.61078*np.exp((17.27*self.data['TS'])/(self.data['TS']+273.3))
            #Tetens Equation
            
            pv = surface_RH*es*np.exp(-1.0*self.data['Z']/scale_height)

            r = 287.058/18.015*pv/(100.0*self.data['PS']-pv)
            
            q = r/(1.0+r)
            
            #q = atmos.calculate('qv',p = 100.0*self.data['PS'],e = pv)
            
            self.addprofiledata(q,name = 'Q')
            
        else:
            print 'Profile has no Z variable and/or TS or PS missing'
            print 'Vapor Estimation not performed'

    def set_cloud_zero(self):
        ''' adds estimated vapor to the profile based on RH at the surface and 
            an exponential decay of PV with height '''
        if self.data.has_key('T'):
            cld = np.zeros_like(self.data['T'])
            self.addprofiledata(cld,name = 'L')
        else:
            print 'Profile has no T variable'
            print 'L not added' 
            
    def set_emissivity(self,emissivity = 0.9):
        self.addsurfacedatum(emissivity,name = 'EM')
        
    def calc_emissivity(self,SurfEmiss,fov=0):  
        if self.surface_type == 'Ocean':
            emiss = SurfEmiss.OceanEmissivityFast(fov,self.data['WS'],self.data['TS'])
            emiss = float(emiss)
            self.addsurfacedatum(emiss,name = 'EM')
        if self.surface_type == 'Ice':
            emiss = SurfEmiss.SeaIceEmissivityFast(fov,self.data['TS'])
            emiss = float(emiss)
            self.addsurfacedatum(emiss,name = 'EM')
        if self.surface_type == 'Land':
            self.addsurfacedatum(0.9,name = 'EM')
                
    
          
    
