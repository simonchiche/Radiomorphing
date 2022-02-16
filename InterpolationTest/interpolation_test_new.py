# -*- coding: utf-8 -*-
"""
Created on Thu May 21 22:39:15 2020

@author: Simon
"""

import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from module_signal_process import filter_traces
import scipy

#from module_signal_process import add_noise_traces, digitalize_traces, filter_traces
#from module_polarisation import get_in_shower_plane, get_stokes_parameters_time_all, get_time_window, get_mean_stokes_parameters_sp, get_polarisation_from_stokes, get_reconstruction, ce2geo_ratio, correct_early_late, get_max_stokes_parameters_sp, get_polarisation_from_absolute_traces, get_field_along_axis, sinusoidal_fit_all, Stokes_parameters_geo, get_mean_stokes_parameters_geo, get_Eb, get_core_reconstruction 
#from module_polarisation_plot import plot_polarisation, plot_polarisation_vxvxb, plot_stokes_parameters, plot_total_field, plot_field_along_axis, plot_time_window, plot_polar_frac, plot_ce2geo_antennaid, plot_ce2geo_wcut, plot_ce2geo_wall, plot_ce2geo_harm_tim, plot_identification_geographic, plot_core_reconstruction



def test_interpolation(Traces, efield_interpolated):
    #trace_data = glob.glob('./data/' + '*trace.npy')
    
    np.savetxt("Traces_simulated.txt", Traces)
    
    ncross_check = 16
    nstarshpe = 160
    print(np.shape(Traces), " shape ")
    
    filtering = False
    if filtering:
        time_sample = int(len(Traces[:,0]))
        Traces = filter_traces(Traces, 176, time_sample)
    #for i in range(len(trace_data)):
    
    #path_cut = trace_data[i].split('trace.npy')[0]
    #Traces = np.load(trace_data[i])    
    
    # Antennas positions in the geographic plane and peak to peak electric field
    n = 176#nstarshpe + ncross_check #number of antennas
    
    use_cross_check = False
    if(use_cross_check):
        cross_time, crossx, crossy, crossz = [], [], [], []
        for i in range(4*n):
            if((i>=n -ncross_check) & (i<n)): cross_time.append(Traces[:,i])
            if((i>=2*n -ncross_check) & (i<2*n)): crossx.append(Traces[:,i])
            if((i>=3*n -ncross_check) & (i<3*n)): 
                crossy.append(Traces[:,i])
                print(i)
            if((i>=4*n -ncross_check) & (i<4*n)): crossz.append(Traces[:,i])
    
    use_starshape = not(use_cross_check)    
    if(use_starshape):
        cross_time, crossx, crossy, crossz = [], [], [], []
        for i in range(4*n):
            if((i>=0) & (i<ncross_check)): cross_time.append(Traces[:,i])
            if((i>=n) & (i<n + ncross_check)): crossx.append(Traces[:,i])
            if((i>=2*n) & (i<2*n + ncross_check)): 
                crossy.append(Traces[:,i])
                print(i)
            if((i>=3*n) & (i<3*n + ncross_check)): crossz.append(Traces[:,i])

    error_peak = []
    
    crossx = np.array(crossx)
    crossy = np.array(crossy)
    crossz = np.array(crossz)
    
    
    for i in range(16):
        
        Load = True
        try:
           # t, ex, ey, ez = np.loadtxt("./OutputDirectory/DesiredTraces_%d.txt" \
            #%i, unpack = True)
            t, ex, ey, ez = efield_interpolated[i][:,0], efield_interpolated[i][:,1], \
            efield_interpolated[i][:,2], efield_interpolated[i][:,3]
        except(IndexError):
            Load = False
            print(i, "error")
        if(Load):
            t, ex, ey, ez = efield_interpolated[i][:,0], efield_interpolated[i][:,1], \
            efield_interpolated[i][:,2], efield_interpolated[i][:,3]
            #t, ex, ey, ez = np.loadtxt("./OutputDirectory/DesiredTraces_%d.txt" \
            #%i, unpack = True)
            
            post_filtering = False
            if post_filtering:
                time_sample = int(len(t))
                Traces_filtered = filter_traces(np.transpose(np.array([t,ex,ey,ez])), 1, time_sample)
                t, ex, ey, ez = Traces_filtered[:,0], Traces_filtered[:,1], Traces_filtered[:,2], Traces_filtered[:,3]
                
                time_sample = int(len(cross_time[0]))
                cross_array = np.transpose(np.array([cross_time[i], crossx[i], crossy[i], crossz[i]]))
                Traces_filtered_cross = filter_traces(cross_array, 1, time_sample)
                crossx[i], crossy[i], crossz[i] = Traces_filtered_cross[:,1], Traces_filtered_cross[:,2], Traces_filtered_cross[:,3]
                
            #print(np.shape(Traces_filtered))
            
            #print(t - Traces_filtered[:,0])
            
            np.savetxt("efield_int%.d.txt" %i, np.array([t, ex, ey, ez]))
            etot = np.sqrt(ex**2 + ey**2 + ez**2)
            cross_tot =  np.sqrt(crossx**2 + crossy**2 + crossz**2)
            
            
            cross_time[i] = cross_time[i] - cross_time[i][0]
            t = t -t[0]
            i_max_t = np.argmax(abs(etot))
            tmax = t[i_max_t]
            i_max_cross = np.argmax(abs(cross_tot[i]))
            cross_tmax = cross_time[i][i_max_cross]
            
            t_a = 0.5*np.arange(0, len(cross_tot[i]), 1)
            t_b = 0.5*np.arange(0, len(etot), 1)
                        
            
            a = scipy.integrate.trapz(cross_tot[i], x = t_a) 
            b = scipy.integrate.trapz(etot, x= t_b) 
            print(a,b)
            
            print(tmax, cross_tmax)
            plt.plot(t_a, cross_tot[i], label = "simulation") 
            #plt.plot(cross_time[i], cross_tot[i], label = "simulation") 
            #plt.plot(t, etot, label = "interpolation")
            plt.plot(t_b, etot, label = "interpolation")
            

            #print(crossy[i][1000])
            plt.xlabel("Time [ns]")
            plt.ylabel("E [$\mu V/m$]")
            plt.xlim(120,210)
            plt.legend()
            plt.tight_layout()
            plt.savefig("./InterpolationTest_Etot_antenna_scale_int3D%.d.pdf" %i)
            plt.show()
            


            Etot_sim = max(np.sqrt(crossx[i]**2 + crossy[i]**2 + crossz[i]**2))
            Etot_scaled = max(np.sqrt(ex**2 + ey**2 + ez**2))
            
            error_peak.append((Etot_sim - Etot_scaled)/Etot_scaled)
            
        
    return error_peak
    
    


            
            
#Etot, Ex, Ey, Ez = np.loadtxt(p2p_data,unpack = True)

#parameter = np.load(parameters_data)
#azimuth, elevation, zenith, inclination, energy, time_sample, x_Xmax, y_Xmax, \
#z_Xmax = parameter[0], parameter[1], parameter[2], parameter[3], parameter[4],\
#parameter[5], parameter[6], parameter[7], parameter[8]        
            
            
            
            




