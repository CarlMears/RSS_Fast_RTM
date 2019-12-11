# -*- coding: utf-8 -*-
"""
Test AtmAbs Class
"""
import numpy as np
from AtmAbs import AtmAbs
from AtmAbs import CldAbs



AtmAbs_2 = AtmAbs(channel = 2)
CldAbs_2 = CldAbs(channel = 2)

T = 284.0
P = 214.0
Q = 0.01
L = 1.0e-7

atmabs = AtmAbs_2.Absorptivity([T,P,Q])
cldabs = CldAbs_2.Absorptivity([T,L])

print atmabs
print cldabs
