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
from scipy.signal import hilbert
#from coreRadiomorphing import *

# function to cdo the checks for the radiomorphing

font = {'family' : 'DejaVu Sans',
        'weight' : 'normal',
        'size'   : 14}

plt.rc('font', **font)

path = "./result_scaling/"
path_ldf_vxb = path + "LDF/vxb/"
path_ldf_vxvxb = path + "LDF/vxvxb/"
path_all_array = path + "All_array/"


def plot_LDF_vxb(Etot_scaled, Etot_sim, sel_vxb,  wvxb_scaled, wvxb_sim, SimulatedShower, TargetShower, RefShower):
    
    LDFvxb_scaled = Etot_scaled[sel_vxb]
    LDFvxb_sim = Etot_sim[sel_vxb]
    arg_scaled = np.argsort(wvxb_scaled)
    arg_sim = np.argsort(wvxb_sim)
    abscisse_scaled = np.sort(wvxb_scaled)
    abscisse_sim = np.sort(wvxb_sim)
    y_scaled = LDFvxb_scaled[arg_scaled]  
    y_sim = LDFvxb_sim[arg_sim]  
    I1 = integrate.trapz(y_scaled, x = abscisse_scaled)  
    I2 = integrate.trapz(y_sim, x = abscisse_sim)  # TODO: check maybe a just a trnaslation of w needed
    DeltaI = (I1-I2)/I1
    
    plt.scatter(wvxb_scaled, LDFvxb_scaled)
    plt.scatter(wvxb_sim, LDFvxb_sim)
    plt.xlabel("$\omega$ [Deg.]")
    plt.ylabel("LDF along $k \\times B$")
    plt.legend(["scaled", "simulated, $\delta I/I = %.3f$" %DeltaI], loc = "lower right", fontsize =12)
    plt.tight_layout()
    plt.savefig(path_ldf_vxb + "LDFvxb_Ea%.2f_tha%.2f_pha%.2f_Eb%.2f_thb%.2f_phb%.2f.pdf" \
    %(RefShower.energy, RefShower.zenith, RefShower.azimuth, \
    TargetShower.energy, TargetShower.zenith, TargetShower.azimuth)) 
    plt.show()
    np.savetxt("LDF_data3.txt", np.transpose([wvxb_scaled, LDFvxb_scaled]))
    
    return LDFvxb_sim, LDFvxb_scaled, DeltaI

def plot_LDF_vxvxb(Etot_scaled, Etot_sim, sel_vxvxb, wvxvxb_scaled, wvxvxb_sim, SimulatedShower, TargetShower, RefShower):

    # LDF of antennas on the vxvxb axis
    LDFvxvxb_scaled = Etot_scaled[sel_vxvxb]
    LDFvxvxb_sim = Etot_sim[sel_vxvxb]  
    arg_scaled = np.argsort(wvxvxb_scaled)
    arg_sim = np.argsort(wvxvxb_sim)
    abscisse_scaled = np.sort(wvxvxb_scaled)
    abscisse_sim = np.sort(wvxvxb_sim)
    y_scaled = LDFvxvxb_scaled[arg_scaled]  
    y_sim = LDFvxvxb_sim[arg_sim]  
    I1 = integrate.trapz(y_scaled, x = abscisse_scaled)  
    I2 = integrate.trapz(y_sim, x = abscisse_sim) 
    DeltaI = (I1-I2)/I1
    
    plt.scatter(wvxvxb_scaled, LDFvxvxb_scaled)
    plt.scatter(wvxvxb_sim, LDFvxvxb_sim)
    plt.xlabel("$\omega$ [Deg.]")
    plt.ylabel("LDF along $k \\times k \\times B$")
    plt.legend(["scaled", "simulated, $\delta I/I = %.3f$" %DeltaI], loc = "lower right", fontsize =12)
    plt.tight_layout()
    plt.savefig(path_ldf_vxvxb + "LDFvxvxb_Ea%.2f_tha%.2f_pha%.2f_Eb%.2f_thb%.2f_phb%.2f.pdf" \
    %(RefShower.energy, RefShower.zenith, RefShower.azimuth, \
    TargetShower.energy, TargetShower.zenith, TargetShower.azimuth)) 
    plt.show()
    
    return LDFvxvxb_sim, LDFvxvxb_scaled, DeltaI

def LDF_relative_deviation(wvxb_scaled, wvxb_sim, wvxvxb_scaled, wvxvxb_sim, \
LDFvxb_scaled, LDFvxb_sim, LDFvxvxb_scaled, LDFvxvxb_sim, SimulatedShower, TargetShower, RefShower):

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
    plt.savefig(path_ldf_vxb + "LDFvxb_reldiff_Ea%.2f_tha%.2f_pha%.2f_Eb%.2f_thb%.2f_phb%.2f.pdf" \
    %(RefShower.energy, RefShower.zenith, RefShower.azimuth, \
    TargetShower.energy, TargetShower.zenith, TargetShower.azimuth))    
    plt.show()
    
    plt.scatter(wvxvxb_diff, LDFvxvxb_reldiff)
    plt.xlabel("$\omega$ [Deg.]")
    plt.ylabel("relative deviation $LDF_{k \\times k \\times B}$")
    plt.tight_layout()
    plt.savefig(path_ldf_vxvxb + "LDFvxvxb_reldiff_Ea%.2f_tha%.2f_pha%.2f_Eb%.2f_thb%.2f_phb%.2f.pdf" \
    %(RefShower.energy, RefShower.zenith, RefShower.azimuth, \
    TargetShower.energy, TargetShower.zenith, TargetShower.azimuth))    
    plt.show()
    
    

def Scalingcheck(TargetShower, SimulatedShower, RefShower):
    
    posSP_sim, SimulatedTraces = SimulatedShower.GetinShowerPlane()
    vxb_sim, vxvxb_sim = posSP_sim[:,1], posSP_sim[:,2]
    eta_sim  =  np.arctan2(vxvxb_sim, vxb_sim)
    
    posSP_scaled, TargetTraces = TargetShower.GetinShowerPlane()
    vxb_scaled, vxvxb_scaled = posSP_scaled[:,1], posSP_scaled[:,2]
    eta_scaled  = np.arctan2(vxvxb_scaled, vxb_scaled)

    n = TargetShower.nant
    
    print("w_scaled, w_sim")    
    w_scaled = TargetShower.get_w()
    w_sim = SimulatedShower.get_w()
    
    Ev_scaled, Evxb_scaled, Evxvxb_scaled, Etot_scaled = np.zeros(n), np.zeros(n), \
    np.zeros(n), np.zeros(n)
    Ev_sim, Evxb_sim, Evxvxb_sim, Etot_sim = np.zeros(n), np.zeros(n),\
    np.zeros(n), np.zeros(n),

    for i in range(n):
        
        Ev_scaled[i]=(max(abs(hilbert(TargetTraces[:,n + i])))) # TODO: use the hilbert envelope
        Evxb_scaled[i]=(max(abs(hilbert(TargetTraces[:,2*n + i]))))
        Evxvxb_scaled[i] =(max(abs(hilbert(TargetTraces[:,3*n + i]))))
        Etot_scaled[i] = np.sqrt(Ev_scaled[i]**2 + Evxb_scaled[i]**2 + Evxvxb_scaled[i]**2)
        
        Ev_sim[i]=(max(abs(hilbert(SimulatedTraces[:,n + i]))))
        Evxb_sim[i]=(max(abs(hilbert(SimulatedTraces[:,2*n + i]))))
        Evxvxb_sim[i]=(max(abs(hilbert(SimulatedTraces[:,3*n + i]))))
        Etot_sim[i] = np.sqrt(Ev_sim[i]**2 + Evxb_sim[i]**2 + Evxvxb_sim[i]**2)
    
    
    l = np.arange(0,n,1)
    nstarshape = 160 # number of antennas without the cross check antennas
    m = np.arange(0,40,1)
    
    sel_vxb = (l<nstarshape) & ((l)%4==0)
    sel_vxvxb = (l<nstarshape) & ((l-2)%4==0)
    
    # w of antennas on the vxb axis
    wvxb_scaled =  w_scaled[sel_vxb]
    wvxb_sim =  w_sim[sel_vxb]
    wvxb_scaled[m%2==0] = -wvxb_scaled[m%2==0] 
    wvxb_sim[m%2==0] = -wvxb_sim[m%2==0]
    
    # w of antennas on the vxvxb axis
    wvxvxb_scaled = w_scaled[sel_vxvxb]
    wvxvxb_sim = w_sim[sel_vxvxb]
    wvxvxb_scaled[m%2==0] = -wvxvxb_scaled[m%2==0]
    wvxvxb_sim[m%2==0] = -wvxvxb_sim[m%2==0]
    
# =============================================================================
#                        LDF comparison
# ============================================================================
    
    # LDF of antennas on the vxb axis
    LDFvxb_sim, LDFvxb_scaled, ILdfvxb = plot_LDF_vxb(Etot_scaled, Etot_sim, sel_vxb,  \
    wvxb_scaled, wvxb_sim, SimulatedShower, TargetShower, RefShower)
    
    
    # LDF of antennas on the vxvxb axis
    LDFvxvxb_sim, LDFvxvxb_scaled, ILdfvxvxb = plot_LDF_vxvxb(Etot_scaled, Etot_sim, \
    sel_vxvxb, wvxvxb_scaled, wvxvxb_sim, SimulatedShower, TargetShower, RefShower)
    
# =============================================================================
#                     LDF: relative deviations
# =============================================================================

    LDF_relative_deviation(wvxb_scaled, wvxb_sim, wvxvxb_scaled, wvxvxb_sim, \
    LDFvxb_scaled, LDFvxb_sim, LDFvxvxb_scaled, LDFvxvxb_sim, SimulatedShower, TargetShower, RefShower)
    
# =============================================================================
#                              All antennas
# =============================================================================
# =============================================================================
#                           Relative deviation
# =============================================================================

# TODO : use the antennas position to check if we are comparing antennas that are close
            
    w_scaledAll = w_scaled[0:160]
    w_simAll =  w_sim[0:160]      
    
    Etot_all = []
    w_diff = []

    for i in range(len(w_scaledAll)):
        
        #minimum = np.argmin(abs(r_sim[i] - r_target))
        wdiff = abs(w_scaledAll - w_simAll[i])
        weta = abs(eta_scaled - eta_sim[i])*180/np.pi
        for j in range(len(wdiff)):
            if((weta[j]<5) & (wdiff[j] <0.05)):
                #if(i == 24): print(j)
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
    
    #plt.scatter(x,y)

    
    
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
    
    #plt.scatter(x2, y2)
   # plt.show()
    

    
        
    arg = np.argsort(r2[0:160])
    r2 = r2[0:160][arg]
    Etot_scaled = Etot_scaled[0:160][arg]
      
    abscisse = r2
    y = Etot_scaled**2
    plt.show()
    #y = y[0:160]
    abscisse = abscisse[0:160]
    Integral_scaled = scipy.integrate.trapz(y*abs(abscisse), x = abscisse) 
    DeltaI = (Integral_sim-Integral_scaled)/Integral_sim
    
        
    
    plt.scatter(w_diff, Etot_all)
    plt.xlabel("$\omega$ [Deg.]")
    plt.ylabel("relative deviation $E_{tot}$")
    plt.legend(["$\delta I/I = %.3f$" %DeltaI], loc = "lower right", fontsize =12)
    plt.tight_layout()
    plt.savefig(path_all_array + "Etot_reldiff_Ea%.2f_tha%.2f_pha%.2f_Eb%.2f_thb%.2f_phb%.2f.pdf" \
    %(RefShower.energy, RefShower.zenith, RefShower.azimuth, \
    TargetShower.energy, TargetShower.zenith, TargetShower.azimuth))   
    plt.show()
    
    
    plt.scatter(w_scaled[0:160], Etot_scaled[0:160])
    plt.scatter(w_sim[0:160], Etot_sim[0:160])
    plt.xlabel("$\omega$ [Deg.]")
    plt.ylabel("E $[\mu V/m]$")
    plt.legend(["scaled", "simulated, $\delta I/I = %.3f$" %DeltaI], loc = "lower right", fontsize =12)
    plt.tight_layout()
    plt.savefig(path_all_array + "Etot_comparison_Ea%.2f_tha%.2f_pha%.2f_Eb%.2f_thb%.2f_phb%.2f.pdf" \
    %(RefShower.energy, RefShower.zenith, RefShower.azimuth, \
    TargetShower.energy, TargetShower.zenith, TargetShower.azimuth))    
    plt.show()
    
    
    return ILdfvxb, ILdfvxvxb, DeltaI
    

    


                