#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 12:57:06 2021

@author: chiche
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import scipy
#from coreRadiomorphing import *

# function to cdo the checks for the radiomorphing

font = {'family' : 'DejaVu Sans',
        'weight' : 'normal',
        'size'   : 14}

plt.rc('font', **font)

def Scalingcheck(TargetShower, SimulatedShower):
    
    posSP_sim = SimulatedShower.GetinShowerPlane()[0]
    vxb_sim, vxvxb_sim = posSP_sim[:,1], posSP_sim[:,2]
    eta_sim  =  np.arctan2(vxvxb_sim, vxb_sim)
    
    posSP_target = TargetShower.GetinShowerPlane()[0]
    vxb_target, vxvxb_target = posSP_target[:,1], posSP_target[:,2]
    eta_target  = np.arctan2(vxvxb_target, vxb_target)
        
    TargetTraces = TargetShower.traces
    SimulatedTraces = SimulatedShower.traces
    n = TargetShower.nant
        
    w_scaled = TargetShower.get_w()
    w_sim = SimulatedShower.get_w()
    
    Ex_scaled, Ey_scaled, Ez_scaled, Etot_scaled = np.zeros(n), np.zeros(n), \
    np.zeros(n), np.zeros(n)
    Ex_sim, Ey_sim, Ez_sim, Etot_sim = np.zeros(n), np.zeros(n),\
    np.zeros(n), np.zeros(n),

    for i in range(n):
        
        Ex_scaled[i]=(max(abs(TargetTraces[:,n + i]))) # TODO: use the hilbert envelope
        Ey_scaled[i]=(max(abs(TargetTraces[:,2*n + i])))
        Ez_scaled[i] =(max(abs(TargetTraces[:,3*n + i])))
        Etot_scaled[i] = np.sqrt(Ex_scaled[i]**2 + Ey_scaled[i]**2 + Ez_scaled[i]**2)
        
        Ex_sim[i]=(max(abs(SimulatedTraces[:,n + i])))
        Ey_sim[i]=(max(abs(SimulatedTraces[:,2*n + i])))
        Ez_sim[i]=(max(abs(SimulatedTraces[:,3*n + i])))
        Etot_sim[i] = np.sqrt(Ex_sim[i]**2 + Ey_sim[i]**2 + Ez_sim[i]**2)
    
    
    l = np.arange(0,n,1)
    nstarshape = 160 # number of antennas without the cross check antennas
    m = np.arange(0,40,1)
    
    # w of antennas on the vxb axis
    wvxb_scaled =  w_scaled[(l<nstarshape) & ((l)%4==0)]
    wvxb_scaled[m%2==0] = -wvxb_scaled[m%2==0] 
    wvxb_sim =  w_sim[(l<nstarshape) & ((l)%4==0)]
    wvxb_sim[m%2==0] = -wvxb_sim[m%2==0]
    
    # w of antennas on the vxvxb axis
    wvxvxb_scaled = w_scaled[(l<nstarshape) & ((l-2)%4==0)]
    wvxvxb_scaled[m%2==0] = -wvxvxb_scaled[m%2==0]
    wvxvxb_sim = w_sim[(l<nstarshape) & ((l-2)%4==0)]
    wvxvxb_sim[m%2==0] = -wvxvxb_sim[m%2==0]
    
# =============================================================================
#                        LDF comparison
# =============================================================================
    
    # LDF of antennas on the vxb axis
    LDFvxb_scaled = Etot_scaled[(l<nstarshape) & ((l)%4==0)]
    LDFvxb_sim = Etot_sim[(l<nstarshape) & ((l)%4==0)]
    arg_scaled = np.argsort(wvxb_scaled)
    arg_sim = np.argsort(wvxb_sim)
    abscisse_scaled = np.sort(wvxb_scaled)
    abscisse_sim = np.sort(wvxb_sim)
    y_scaled = LDFvxb_scaled[arg_scaled]  
    y_sim = LDFvxb_sim[arg_sim]  
    I1 = integrate.trapz(y_scaled, x = abscisse_scaled)  
    I2 = integrate.trapz(y_sim, x = abscisse_sim)  
    DeltaI = (I1-I2)/I1
    plt.scatter(wvxb_scaled, LDFvxb_scaled)
    plt.scatter(wvxb_sim, LDFvxb_sim)
    plt.xlabel("$\omega$ [Deg.]")
    plt.ylabel("LDF along $k \\times B$")
    plt.legend(["scaled", "simulated, $\delta I/I = %.3f$" %DeltaI], loc = "lower right", fontsize =12)
    plt.tight_layout()
    plt.savefig("LDFvxb_Ea%.2f_tha%.2f_pha%.2f_Eb%.2f_thb%.2f_phb%.2f.pdf" \
    %(SimulatedShower.energy, SimulatedShower.zenith, SimulatedShower.azimuth, \
    TargetShower.energy, TargetShower.zenith, TargetShower.azimuth))
    plt.show()
    
    
    # LDF of antennas on the vxvxb axis
    LDFvxvxb_scaled = Etot_scaled[(l<nstarshape) & ((l-2)%4==0)]
    LDFvxvxb_sim = Etot_sim[(l<nstarshape) & ((l-2)%4==0)]  
    arg_scaled = np.argsort(wvxvxb_scaled)
    arg_sim = np.argsort(wvxvxb_sim)
    abscisse_scaled = np.sort(wvxvxb_scaled)
    abscisse_sim = np.sort(wvxvxb_sim)
    y_scaled = LDFvxvxb_scaled[arg_scaled]  
    y_sim = LDFvxvxb_sim[arg_sim]  
    I1 = integrate.trapz(y_scaled, x = abs(abscisse_scaled))  
    I2 = integrate.trapz(y_sim, x = abs(abscisse_sim)) 
    DeltaI = (I1-I2)/I1
    plt.scatter(wvxvxb_scaled, LDFvxvxb_scaled)
    plt.scatter(wvxvxb_sim, LDFvxvxb_sim)
    plt.xlabel("$\omega$ [Deg.]")
    plt.ylabel("LDF along $k \\times k \\times B$")
    plt.legend(["scaled", "simulated, $\delta I/I = %.3f$" %DeltaI], loc = "lower right", fontsize =12)
    plt.tight_layout()
    plt.savefig("LDFvxvxb_Ea%.2f_tha%.2f_pha%.2f_Eb%.2f_thb%.2f_phb%.2f.pdf" \
    %(SimulatedShower.energy, SimulatedShower.zenith, SimulatedShower.azimuth, \
    TargetShower.energy, TargetShower.zenith, TargetShower.azimuth))
    plt.show()
    


    
# =============================================================================
#                     LDF: relative deviations
# =============================================================================

    LDFvxb_diff = []
    LDFvxb_reldiff = []
    LDFvxvxb_diff  = []
    LDFvxvxb_reldiff  = []
    wvxb_diff = []
    wvxvxb_diff = []
    
    for i in range(len(wvxb_sim)):
        diff_vxb = abs(wvxb_scaled - wvxb_sim[i])
        diff_vxvxb = abs(wvxvxb_scaled - wvxvxb_sim[i])
        
        if(min(diff_vxb)<0.1):
            minimum = np.argmin(diff_vxb)
            LDFvxb_diff.append(LDFvxb_scaled[minimum] - LDFvxb_sim[i])
            LDFvxb_reldiff.append((LDFvxb_scaled[minimum] - LDFvxb_sim[i])/LDFvxb_sim[i])
            wvxb_diff.append(wvxb_sim[i])
        
        if(min(diff_vxvxb)<0.1):
            minimum = np.argmin(diff_vxvxb)
            LDFvxvxb_diff.append(LDFvxvxb_scaled[minimum] - LDFvxvxb_sim[i])
            LDFvxvxb_reldiff.append((LDFvxvxb_scaled[minimum] - LDFvxvxb_sim[i])/LDFvxvxb_sim[i])
            wvxvxb_diff.append(wvxvxb_sim[i])
    
    plt.scatter(wvxb_diff, LDFvxb_reldiff)
    plt.xlabel("$\omega$ [Deg.]")
    plt.ylabel("relative deviation $LDF_{k \\times B}$")
    plt.tight_layout()
    plt.savefig("LDFvxb_reldiff_Ea%.2f_tha%.2f_pha%.2f_Eb%.2f_thb%.2f_phb%.2f.pdf" \
    %(SimulatedShower.energy, SimulatedShower.zenith, SimulatedShower.azimuth, \
    TargetShower.energy, TargetShower.zenith, TargetShower.azimuth))    
    plt.show()
    
    plt.scatter(wvxvxb_diff, LDFvxvxb_reldiff)
    plt.xlabel("$\omega$ [Deg.]")
    plt.ylabel("relative deviation $LDF_{k \\times k \\times B}$")
    plt.tight_layout()
    plt.savefig("LDFvxvxb_reldiff_Ea%.2f_tha%.2f_pha%.2f_Eb%.2f_thb%.2f_phb%.2f.pdf" \
    %(SimulatedShower.energy, SimulatedShower.zenith, SimulatedShower.azimuth, \
    TargetShower.energy, TargetShower.zenith, TargetShower.azimuth))    
    plt.show()
    
# =============================================================================
#                              All antennas
# =============================================================================
# =============================================================================
#                           Relative deviation
# =============================================================================

# TODO : use the antennas position to check i f we are comparing antennas that are close
            
    w_scaledAll = w_scaled[0:160]
    w_simAll =  w_sim[0:160]      
    
    Etot_all = []
    w_diff = []

    for i in range(len(w_scaledAll)):
        
        #minimum = np.argmin(abs(r_sim[i] - r_target))
        wdiff = abs(w_scaledAll - w_simAll[i])
        weta = abs(eta_target - eta_sim[i])*180/np.pi
        for j in range(len(wdiff)):
            if((weta[j]<5) & (wdiff[j] <0.05)):
                if(i == 24): print(j)
                Etot_all.append((Etot_scaled[j] - Etot_sim[i])/Etot_sim[i])
                w_diff.append(w_simAll[i])
    

# =============================================================================
#                          Total integral
# =============================================================================

    positions  = SimulatedShower.pos  
    core = SimulatedShower.get_center()
    x, y, z  = positions[:,0], positions[:,1], positions[:,2]
    x,y,z = x- core[0], y - core[1], z - core[2]
    r = np.sqrt(x**2 + y**2 + z**2) 
    
    arg = np.argsort(r[0:160])
    r = r[0:160][arg]
    Etot_sim = Etot_sim[0:160][arg]
      
    abscisse = r
    abscisse = abscisse
    y = Etot_sim**2
    Integral_sim = scipy.integrate.trapz(y*abs(abscisse), x = abscisse) 
    
    
    positions2  = TargetShower.pos  
    core2 = TargetShower.get_center()
    x2, y2, z2  = positions2[:,0], positions2[:,1], positions2[:,2]
    x2, y2, z2 = x2 - core2[0], y2 - core2[1], z2 - core2[2]
    r2 = np.sqrt(x2**2 + y2**2 + z2**2) 
    

    
    arg = np.argsort(r2[0:160])
    r2 = r2[0:160][arg]
    Etot_scaled = Etot_scaled[0:160][arg]
      
    abscisse = r2
    y = Etot_scaled**2
    y = y[0:160]
    abscisse = abscisse[0:160]
    Integral_scaled = scipy.integrate.trapz(y*abs(abscisse), x = abscisse) 
    DeltaI = (Integral_sim-Integral_scaled)/Integral_sim
    
    plt.scatter(w_diff, Etot_all)
    plt.xlabel("$\omega$ [Deg.]")
    plt.ylabel("relative deviation $E_{tot}$")
    plt.legend(["$\delta I/I = %.3f$" %DeltaI], loc = "lower right", fontsize =12)
    plt.tight_layout()
    plt.savefig("Etot_reldiff_Ea%.2f_tha%.2f_pha%.2f_Eb%.2f_thb%.2f_phb%.2f.pdf" \
    %(SimulatedShower.energy, SimulatedShower.zenith, SimulatedShower.azimuth, \
    TargetShower.energy, TargetShower.zenith, TargetShower.azimuth))    
    plt.show()
    
    
    plt.scatter(w_scaled[0:160], Etot_scaled[0:160])
    plt.scatter(w_sim[0:160], Etot_sim[0:160])
    plt.xlabel("$\omega$ [Deg.]")
    plt.ylabel("E $[\mu V/m]$")
    plt.legend(["scaled", "simulated, $\delta I/I = %.3f$" %DeltaI], loc = "lower right", fontsize =12)
    plt.savefig("Etot_comparison_Ea%.2f_tha%.2f_pha%.2f_Eb%.2f_thb%.2f_phb%.2f.pdf" \
    %(SimulatedShower.energy, SimulatedShower.zenith, SimulatedShower.azimuth, \
    TargetShower.energy, TargetShower.zenith, TargetShower.azimuth))    
    plt.show()
    
    
    
    