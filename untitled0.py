# -*- coding: utf-8 -*-
"""
RSS RTM module

"""
from AtmosProfile import AtmosProfile
import numpy as np
from AtmAbs import AtmAbs
from AtmAbs import CldAbs


class RSS_RTM_2():
    '''Class to implement RSS MEthod 2 MSU/AMSU simulator'''
    
    def CalcTb(profile,channel = 2):
        ''' Calculates Brightness Temperatures for a given atmospheric and surface state '''
        
        # make sure that all required infomation is present
        # the following is necessary
        data_missing = False
        
        
    