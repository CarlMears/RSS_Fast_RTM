# -*- coding: utf-8 -*-
"""
RSS RTM module

"""
from AtmosProfile import AtmosProfile
import numpy as np
from AtmAbs import AtmAbs
from AtmAbs import CldAbs
#import math
from numba import jit


@jit(nopython=True)
def atm_tran_3(nlev,tht,t,p,tabs):

#-----------------------------------------------------------------------------
#	compute atmospheric downwelling and upwelling brightness temperatures
#	and upward transmittance at each pressure level (altitude)
#	
#	In this version, the integration is done in P, with no reference to Z
#	the absorption coefficient is assumed to be linear in P between levels
#	
#	the calculation is done in real*8, and then converted back to real*4
#	
#-----------------------------------------------------------------------------
#	input:
#     nlev           number of atmosphere levels
#     tht            earth incidence angle [in deg]
#     tabs(0:nlev)   atmospheric absorption coefficients [nepers/hPa]
#     t(0:nlev)      temperature profile[in k]
#	    p(0:nlev)      pressure (Pa)
#-----------------------------------------------------------------------------
#     output:
#     tran_up   total atmospheric transmission from each level to the TOA -- tran_up[0] = transmission of entire atmosphere
#     tran_dwn  total atmospheric transmission from each level to the surface
#     tbdw			downwelling brightness temperature t_bd [in k]
#     tbup			upwelling   brightness temperature t_bu [in k]
#     opacty    opacity of each layer (optional)
#-----------------------------------------------------------------------------  
# convert ot real*8

  # define arrays
    tran_up  = np.zeros(nlev)
    tran_dwn = np.zeros(nlev)
    opacty   = np.zeros(nlev-1)  # these are layer variables
    tavg     = np.zeros(nlev-1)  # these are layer variables
    ems      = np.zeros(nlev-1)  # these are layer variables
 
    sectht   = 1.0/np.cos(np.radians(tht))

    for i in range(1,nlev):

        opacty[i-1]= sectht*0.5*(tabs[i-1]+tabs[i])*(p[i]-p[i-1])
        tavg[i-1]  = 0.5*(t[i-1]+t[i])
        ems[i-1]   = 1.-np.exp(opacty[i-1])  # ems(0) is the total emissivity of first layer									                  # between temperature levels 0 and 1, etc.
        
    
    sumop=0.0
    sumdw=0.0
    tran_dwn[0] = 1.00
	
    for i in range(1,nlev):
        tran_dwn[i] = np.exp(sumop+opacty[i-1])
        sumdw += (tavg[i-1]-t[1])*ems[i-1]*tran_dwn[i-1]
        sumop += opacty[i-1]
        
    sumop=0.0
    sumup=0.0
    for i in range(nlev-1,0,-1):
        tran_up[i]  = np.exp(sumop)
        sumup += (tavg[i-1]-t[1])*ems[i-1]*tran_up[i]
        sumop += opacty[i-1]
        


    tran_up[0] = np.exp(sumop)
    tbavg=(1.-tran_up[0])*t[1]
    tbdw=tbavg+sumdw
    tbup=tbavg+sumup

    return tran_up,tran_dwn,tbdw,tbup,opacty
	



class RSS_RTM_2():
    '''Class to implement RSS Method 2 MSU/AMSU simulator'''
    
    
    def __init__(self,channel=2,RTM_Data_Path = './data/'):
        self.AtmAbs = AtmAbs(channel = channel,RTM_Data_Path=RTM_Data_Path)
        self.CldAbs = CldAbs(channel = channel,RTM_Data_Path=RTM_Data_Path)
    
    
    def CalcTb(self,profile,theta = 0.0):
        ''' Calculates Brightness Temperatures for a given atmospheric and surface state '''
        
        # make sure that all required infomation is present
        # the following is necessary
        data_OK = True
        profile_must_contain = ['T','P','Q','L']
        for key in profile_must_contain:
            if not (key in profile.data):
                data_OK = False
        surface_must_contain = ['TS','PS','QS','LS','EM']
        for key in surface_must_contain:
            if not (key in profile.data):
                data_OK = False
                
        if data_OK:
            
            # add surface data to atmospheric profiles
            
            nlev = len(profile.data['T'])
            T = np.zeros(nlev+1)
            T[0] = profile.data['TS']
            T[1:nlev+1] = profile.data['T']
            
            P = np.zeros(nlev+1)
            P[0] = profile.data['PS']
            P[1:nlev+1] = profile.data['P']
            
            Q = np.zeros(nlev+1)
            Q[0] = profile.data['QS']
            Q[1:nlev+1] = profile.data['Q']
            
            L = np.zeros(nlev+1)
            L[0] = profile.data['LS']
            L[1:nlev+1] = profile.data['L']
            
            # calculate absorptivity for each level            
            aatm = self.AtmAbs.Absorptivity(np.stack((Q,P,T)))
            acld = self.CldAbs.Absorptivity(np.stack((T,L)))
            TotAbs = aatm + acld
            
            # do the rtm
            tran_up,tran_dwn,tbdw,tbup,opacty = atm_tran_3(nlev+1,theta,T,P,TotAbs)
            
            tb = ((tbdw*(1.0 - profile.data['EM'])) + profile.data['EM']*T[0])*tran_up[0]+tbup
            
            return tbdw,tbup,tb
            
            
            
            
            
        
        
    