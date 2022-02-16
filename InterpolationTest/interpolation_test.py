# -*- coding: utf-8 -*-
"""
Created on Thu May 21 22:39:15 2020

@author: Simon
"""

import numpy as np
import matplotlib.pyplot as plt
import glob
import os


from module_signal_process import add_noise_traces, digitalize_traces, filter_traces
from module_polarisation import get_in_shower_plane, get_stokes_parameters_time_all, get_time_window, get_mean_stokes_parameters_sp, get_polarisation_from_stokes, get_reconstruction, ce2geo_ratio, correct_early_late, get_max_stokes_parameters_sp, get_polarisation_from_absolute_traces, get_field_along_axis, sinusoidal_fit_all, Stokes_parameters_geo, get_mean_stokes_parameters_geo, get_Eb, get_core_reconstruction 
from module_polarisation_plot import plot_polarisation, plot_polarisation_vxvxb, plot_stokes_parameters, plot_total_field, plot_field_along_axis, plot_time_window, plot_polar_frac, plot_ce2geo_antennaid, plot_ce2geo_wcut, plot_ce2geo_wall, plot_ce2geo_harm_tim, plot_identification_geographic, plot_core_reconstruction

trace_data = glob.glob('./data/' + '*trace.npy')
#filename_cut = os.path.splitext(filename_all[0])[0][17:]
Traces = dict()
x, y, z, vcrossB, vcross_vcrossB, Etot = dict(), dict(), dict(), dict(), dict(), dict()
w, alpha = dict(), dict()
Ev, Evxb , Evxvxb, Ex, Ey, Ez = dict(), dict(), dict(), dict(), dict(), dict()

for i in range(len(trace_data)):
    
    # Temporal traces of the electric field. The first 176 rows corresponds to the
    # time related to the traces of each antenna. The 176 following rows to the
    # Ex component of the traces, the 176 following rows to the Ey component etc
    path_cut = trace_data[i].split('trace.npy')[0]
    positons_data = path_cut + 'positions.npy'
    p2p_data = path_cut + 'p2p.txt'
    print(p2p_data,i)
    parameters_data = path_cut + 'parameters.npy'
    filename_all = (path_cut + '.hdf5').split('_.hdf5')[0] + '.hdf5'
    Traces[i] = np.load(trace_data[i])  
    Traces2 = np.load(trace_data[i])
    
    print(np.shape(Traces[i]), " shape ")
    print(Traces[i][:,226], "traces")
    
    # Antennas positions in the geographic plane and peak to peak electric field
    positions = np.load(positons_data)
    x[i],y[i],z[i] = positions[:,0], positions[:,1], positions[:,2]
    #E_all = np.load(p2p_data)
    Etot[i], Ex[i], Ey[i], Ez[i] = np.loadtxt(p2p_data,unpack = True)#E_all[:,0], E_all[:,1], E_all[:,2], E_all[:,3]
    

    parameter = np.load(parameters_data)
    azimuth, elevation, zenith, inclination, energy, time_sample, x_Xmax, \
    y_Xmax, z_Xmax = parameter[0], parameter[1], parameter[2], parameter[3], \
    parameter[4], parameter[5], parameter[6], parameter[7], parameter[8]
    
    time_sample = int(len(Traces[i][:,0]))    
    
    n = len(x[i]) #number of antennas
    trigger = 0
    
#plt.scatter(x[0], y[0],c = Etot[0], cmap ='jet')
#plt.scatter(x[1], y[1],c = Etot[1], cmap ='jet')

#plt.scatter(x[0][160:], y[0][160:],c = Etot[0][160:], cmap ='jet')
#plt.scatter(x[1], y[1],c = Etot[1], cmap ='jet')
#plt.show()
    

n = 176
t, ex, ey, ez = dict(), dict(), dict(), dict()
##crossy, crossx, crossz =dict(), dict(), dict()  # electric field for cross check
crossx, crossy, crossz = [], [], []

for i in range(704):
    if((i>=2*n -16) & (i<2*n)): crossx.append(Traces[0][:,i])
    if((i>=3*n -16) & (i<3*n)): 
        crossy.append(Traces[0][:,i])
        print(i)
    if((i>=4*n -16) & (i<4*n)): crossz.append(Traces[0][:,i])

time_diff = dict()

peakdeviation = np.zeros(16)
peakdeviation_noscale = np.zeros(16)

np.savetxt("crossy_corrected.txt", [crossy[0]])
np.savetxt("Traces_test.txt", Traces2)

print(max(abs(Traces[0][:,513])), ".....")
print(max(abs(crossy[1])), "...............")

error_peak = []

for i in range(16):
    
        #plt.plot(Traces[1][:,i], Traces[1][:,16*2+i])
        Load = True
        try:
            #t[i], ex[i], ey[i], ez[i] = np.loadtxt("./OutputDirectory/DesiredTraces_%d.txt" %i, unpack = True)
            t, ex, ey, ez =  np.loadtxt("./OutputDirectory/DesiredTraces_%d.txt" %i, unpack = True)
        
        except(OSError):
            Load = False
            print(i, "error")
        if(Load):
           # t[i], ex[i], ey[i], ez[i] = np.loadtxt("./OutputDirectory/DesiredTraces_%d.txt" %i, unpack = True)
            t, ex, ey, ez =  np.loadtxt("./OutputDirectory/DesiredTraces_%d.txt" %i, unpack = True)
            peak_ref =  np.argmax(ey[i])
            peak_target = np.argmax(crossy[i])
            
            #deltat = abs(t[peak_ref] - Traces[0][:,160+peak_target])
            #plt.plot(t- t[peak_ref], ey, label = "interpolation")
            plt.plot(Traces[0][:,160+i] - Traces[0][peak_target,160+i], crossy[i], label = "simulation")
            ##print(crossy[512+ i][1000], "!!!!")
            # old abscisse Traces[0][:,160+i]
            argt0 = np.argmax(abs(crossy[i]))
            #plt.xlim(Traces[0][argt0,160+i] - 70, Traces[0][argt0,160+i] +170)
            #time_diff[i] = t[i] - Traces[0][:,i] 
            plt.xlabel("Time [ns]")
            plt.ylabel("E [$\mu V/m$]")
            plt.legend()
            plt.tight_layout()
            #plt.savefig("./images/InterpolationTest_Ey_antenna_noscale%.d.pdf" %i)
            plt.show()
            
            
            Etot_sim = max(np.sqrt(crossx[i]**2 + crossy[i]**2 + crossz[i]**2))
            Etot_scaled = max(np.sqrt(ex**2 + ey**2 + ez**2))
            
            error_peak.append((Etot_sim - Etot_scaled)/Etot_scaled)
        
print(error_peak)
                      
  


