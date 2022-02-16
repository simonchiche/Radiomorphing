#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 14:51:26 2021

@author: chiche
"""


import os
import h5py
import numpy as np
import hdf5fileinout as hdf5io
import sys
import glob
import matplotlib.pyplot as plt


# =============================================================================
#                           Energy scale
# =============================================================================
    
def EnergyScale(RefShower, TargetShower):
    
    Nant = RefShower.nant
    ref_energy = RefShower.energy
    target_energy = TargetShower.energy
    
    if(TargetShower.fluctuations):
        
        ref_energy = np.random.normal(ref_energy, 0.1*ref_energy)
    
    kE = target_energy/ref_energy
    
    scaled_traces = TargetShower.traces[:,Nant:]*kE
    
    return scaled_traces, kE

# =============================================================================
#                       Geomagnetic angle scale
# =============================================================================

def GeomagneticScale(RefShower, TargetShower):
    
    Nant = RefShower.nant
    
    ref_alpha = RefShower.get_alpha()
    target_alpha = TargetShower.get_alpha()
        
    kgeo  = np.sin(target_alpha)/np.sin(ref_alpha)

    ref_zenith = RefShower.zenith
    
    
    LimitZenith = 180
    
    print(kgeo)
        
    if(ref_zenith>LimitZenith): # TODO: refine this condition, rather a condition over theta
    
        vxb, vxvxb = TargetShower.pos[:,1], TargetShower.pos[:,2]
        cos_eta = vxb/np.sqrt(vxb**2 + vxvxb**2)
        
        Evxb, Evxvxb = TargetShower.traces[:,2*Nant:3*Nant], TargetShower.traces[:,3*Nant:4*Nant] 
        
        w = RefShower.get_w()
                
        w_kxkxb = []
        Ece = []
        Egeo = []
        
        for j in range(len(w)):
            if(abs(vxb[j])<1):
                w_kxkxb.append(w[j])
                Ece.append(abs(Evxvxb[:,j]))
                Egeo.append(abs(Evxb[:,j]))
          
        Evxb_scaled = np.zeros(np.shape(Evxb))
        
        for i in range(len(w)):
            
            diff = abs(w_kxkxb - w[i])
            minimum = np.argmin(diff)
            Evxb_scaled[:,i] = -(Egeo[minimum]*kgeo + Ece[minimum]*cos_eta[i])
            
    else: 
        Evxb_scaled = TargetShower.traces[:,2*Nant:3*Nant]*kgeo
    
    
    return Evxb_scaled, kgeo
    
 
# =============================================================================
#                          Density scaling
# =============================================================================
         
def DensityScale(RefShower, TargetShower):
    
    Nant = RefShower.nant 
    
    xmax_target  = TargetShower.getXmaxPosition()
            
    XmaxHeight_target, DistDecayXmax = TargetShower._dist_decay_Xmax()    
    XmaxHeight_ref = RefShower.getSphericalXmaxHeight()
            
    rho_ref = TargetShower._getAirDensity(XmaxHeight_ref, "linsley")
    rho_target = TargetShower._getAirDensity(XmaxHeight_target, "linsley")
        
    #krho =  1/np.sqrt(rho_ref/rho_target) # previous implementation
    
    ############fit Egeo #####
    
    def fit_broken_law(rho_bins):

        Phi0 = 1010
        gamma1 = -1.0047
        gamma2 = 0.2222
        rho_break  = 3.5e-4
        beta = 2.5
        rho0 = 1.87e-6
        y = (Phi0*(rho_bins/rho0)**(-gamma1))*(1+ (rho_bins/rho_break)**((gamma2- gamma1))/beta)**(-beta)
    
        return y
    
    b = 1.194878
    krho_geo = (fit_broken_law(rho_target)/fit_broken_law(rho_ref))
    krho_ce = (rho_target**b)/(rho_ref**b)
    
    scaled_traces_geo = TargetShower.traces[:,2*Nant:3*Nant]*krho_geo
    scaled_traces_ce = TargetShower.traces[:,3*Nant:4*Nant]*krho_ce
            
    return scaled_traces_geo, scaled_traces_ce, xmax_target, krho_geo, krho_ce
    
# =============================================================================
#                       Cerenkov Stretch   
# =============================================================================

def CerenkovStretch(RefShower, TargetShower):
    
    cerangle_ref = RefShower.get_cerenkov_angle()
    cerangle_target = TargetShower.get_cerenkov_angle()
    
    kstretch = cerangle_ref/cerangle_target
    w = RefShower.get_w()/kstretch
        
    v, vxb, vxvxb =  TargetShower.pos[:,0], TargetShower.pos[:,1], TargetShower.pos[:,2]
    eta = np.arctan2(vxvxb, vxb)
    Distplane = TargetShower.distplane  
    d = Distplane*np.tan(w*np.pi/180.0)
    
    vxb_scaled = d*np.cos(eta) 
    vxvxb_scaled = d*np.sin(eta)
    
    scaled_pos = np.array([v,vxb_scaled, vxvxb_scaled]).T
    scaled_traces = TargetShower.traces[:,176:]*kstretch
                
    return scaled_pos, scaled_traces, kstretch
    




