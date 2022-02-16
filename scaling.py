#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 15:28:02 2021

@author: chiche
"""

from ModuleScale import EnergyScale, GeomagneticScale, DensityScale, CerenkovStretch
import matplotlib.pyplot as plt
import numpy as np
   
#def myscale(sim_file, primary, energy, zenith, azimuth):
def myscale(RefShower, TargetShower, SimulatedShower):
     
    Nant = RefShower.nant
    
    
    # Translation in the shower plane
    TargetShower.pos, TargetShower.traces = RefShower.GetinShowerPlane()
    
    # Energy scaling
    TargetShower.traces[:,Nant:], kE = EnergyScale(RefShower, TargetShower)
    
    # Geomagnetic scaling
    TargetShower.traces[:,2*Nant:3*Nant], kgeo = GeomagneticScale(RefShower, TargetShower)
    
    # Density scaling

    TargetShower.traces[:,2*Nant:3*Nant], TargetShower.traces[:,3*Nant:4*Nant],\
    TargetShower.xmaxpos, krho_geo, krho_ce = DensityScale(RefShower, TargetShower) 
    
    
    TargetShower.xmaxpos = SimulatedShower.xmaxpos # for the tests only 
        
    # Layout and traces stretching
    TargetShower.pos, TargetShower.traces[:,Nant:], kstretch = CerenkovStretch(RefShower, TargetShower)

    # Back in the geographic plane
    TargetShower.pos, TargetShower.traces = TargetShower.GetinGeographicFrame()
        
    # TODO: include magnetic field scaling
    
    return TargetShower, krho_geo
        