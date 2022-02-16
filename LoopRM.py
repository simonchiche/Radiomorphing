#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 02:24:48 2021

@author: chiche
"""

import glob
import numpy as np
from RunRadioMorphing import run


simulations = glob.glob("./TargetShowers/*.hdf5") # in lyon get the full library

ILDFvxbAll, ILDFvxvxbAll , ItotAll , krho_all, Ref_thetaAll, Target_thetaAll, dplane_all = \
[], [], [], [], [], [], []
print(simulations)
for i in range(len(simulations)):
    
    energy =  float(simulations[i].split("_")[2])
    zenith =  float(simulations[i].split("_")[3])
    azimuth = float(simulations[i].split("_")[4])
    dplane =  0#float(simulations[i].split("_")[7][:-5])  
    
    print("dplane!!!!", dplane)
    filename = simulations[i].split("/")[-1]
    
    path = "./Simulations/SelectedPlane/theta_%.1f/" %(zenith) + filename
    


    ref_sim =  glob.glob(path)
    
    if(len(ref_sim)>0): continue # we check that the shower is not already in the lib
            
    
    ILDFvxb, ILDFvxvxb, Itot, krho_geo, Ref_zenith, Target_zenith = run(energy, zenith, azimuth, dplane, simulations[i])
    
    ILDFvxbAll.append(ILDFvxb), ILDFvxvxbAll.append(ILDFvxvxb), ItotAll.append(Itot)
    krho_all.append(krho_geo), Ref_thetaAll.append(Ref_zenith), Target_thetaAll.append(Target_zenith)
    dplane_all.append(dplane)
    
    #print(ILDFvxb, ILDFvxvxb)
    

#np.savetxt("zenith_scaling_check.txt", \
#np.transpose([ILDFvxbAll, ILDFvxvxbAll, ItotAll, krho_all, Ref_thetaAll, Target_thetaAll, dplane_all]))
    
#np.savetxt("zenith_scaling_check.txt", error_peak)
    


