#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 22:36:22 2022

@author: chiche
"""
import h5py
import hdf5fileinout as hdf5io
from coreRadiomorphing_ground import Shower
import numpy as np
import glob
import matplotlib.pyplot as plt

def extractData(sim_file):
    
    simu_path = './' + sim_file
    print(sim_file)
    InputFilename = simu_path
    filehandle = h5py.File(InputFilename, 'r')

    #Shower event access
    RunInfo = hdf5io.GetRunInfo(InputFilename)         
    NumberOfEvents = hdf5io.GetNumberOfEvents(RunInfo)
    EventNumber = NumberOfEvents-1 
    EventName = hdf5io.GetEventName(RunInfo,EventNumber)       
    #All simulations info

    #Shower parameters
    Zenith = hdf5io.GetEventZenith(RunInfo,EventNumber)
    Azimuth = hdf5io.GetEventAzimuth(RunInfo,EventNumber)
    Primary = "electron"
    Energy = hdf5io.GetEventEnergy(RunInfo,EventNumber)
    XmaxDistance = hdf5io.GetEventXmaxDistance(RunInfo,EventNumber)
    #SlantXmax = hdf5io.GetEventSlantXmax(RunInfo,EventNumber) # Slant ?
    Energy = hdf5io.GetEventEnergy(RunInfo,EventNumber)
    #HadronicModel = hdf5io.GetEventHadronicModel(RunInfo,EventNumber)    
    
    #Shower info
    EventInfo = hdf5io.GetEventInfo(InputFilename,EventName)
    XmaxPosition = hdf5io.GetEventXmaxPosition(EventInfo)
    BFieldIncl = hdf5io.GetEventBFieldIncl(EventInfo) #BfieldIncl/Bfieldecl
    #BFieldDecl = hdf5io.GetEventBFieldDecl(EventInfo)
    GroundAltitude = hdf5io.GetGroundAltitude(EventInfo)    
    #Antannas info
    AntennaInfo = hdf5io.GetAntennaInfo(InputFilename,EventName)


    NumberOfAntennas = hdf5io.GetNumberOfAntennas(AntennaInfo)
    IDs_bis = hdf5io.GetAntIDFromAntennaInfo(AntennaInfo)
    X = hdf5io.GetXFromAntennaInfo(AntennaInfo)
    Y = hdf5io.GetYFromAntennaInfo(AntennaInfo)
    Z = hdf5io.GetZFromAntennaInfo(AntennaInfo)
    Positions = np.transpose([X,Y,Z])

    #Traces
    AntennaID = IDs_bis[0]
    Efield_trace = hdf5io.GetAntennaEfield(InputFilename,EventName,AntennaID,OutputFormat="numpy")
    time_sample = len(Efield_trace[:,0])
    
    trace_x = np.zeros([time_sample,NumberOfAntennas])
    trace_y = np.zeros([time_sample,NumberOfAntennas])
    trace_z = np.zeros([time_sample,NumberOfAntennas])
    Time = np.zeros([time_sample,NumberOfAntennas])
    
    for i in range(NumberOfAntennas):
    
        AntennaID = IDs_bis[i]
        path = str(EventName)+"/AntennaTraces/"+str(AntennaID)+"/efield"
        Efield_trace = filehandle[path]
        Time[:,i] = Efield_trace['Time']
        trace_x[:,i] = Efield_trace['Ex']
        trace_y[:,i] = Efield_trace['Ey']
        trace_z[:,i] = Efield_trace['Ez']
               
    Traces = np.transpose(np.concatenate((np.transpose(Time), np.transpose(trace_x), \
                                          np.transpose(trace_y), np.transpose(trace_z))))
              
    Nant = len(X)
    Injection = 1e5 # TODO: get it from the hdf5
    fluctuations = True
        
    
    RefShower = Shower(Primary, Energy, Zenith, Azimuth, Injection, Nant, BFieldIncl, GroundAltitude,
                        Positions, Traces, XmaxDistance, XmaxPosition[0], fluctuations)

    return RefShower  


efield_data = glob.glob("./OutputDirectory/*.txt")

Ndes = 176
k = 0
nbins = 1143
morphed_Traces = []
for i in range(Ndes):
    
    try:
        morphed_Traces.append(np.loadtxt("./OutputDirectory/DesiredTraces_%d.txt" %i))
        print(i, k)
        k = k +1
    
    except IOError:
        morphed_Traces.append(np.zeros([nbins, 4]))
        continue
  
#morphed_Traces = np.array(morphed_Traces)  
 
simulations = glob.glob("./TargetShowers/*.hdf5")
TargetShower = extractData(simulations[0])

sim_Traces = TargetShower.traces


for i in range(Ndes):
    
    plt.plot(sim_Traces[:, i + 2*Ndes])
    plt.plot(morphed_Traces[i][ :, 2])
    plt.show()


# =============================================================================
#                   plot in plane
# =============================================================================
 
sim_pos = TargetShower.pos
Etot_sim = np.zeros(Ndes)
Etot_morphed = np.zeros(Ndes)

for i in range(Ndes):   
    
    Ex = max(sim_Traces[:,i + Ndes])
    Ey = max(sim_Traces[:,i + 2*Ndes])
    Ez = max(sim_Traces[:,i + 3*Ndes])
    Etot_sim[i] = np.sqrt(Ex**2 + Ey**2 + Ez**2)
    
    Ex = max(morphed_Traces[i][:,1])
    Ey = max(morphed_Traces[i][:,2])
    Ez = max(morphed_Traces[i][:,3])
    Etot_morphed[i] = np.sqrt(Ex**2 + Ey**2 + Ez**2)
    
plt.scatter(sim_pos[:,0], sim_pos[:,1], c= Etot_sim, cmap = 'jet')
plt.colorbar()
plt.show()

plt.scatter(sim_pos[:,0], sim_pos[:,1], c= Etot_morphed, cmap = 'jet')
plt.colorbar()
plt.show()


plt.scatter(sim_pos[:,0], sim_pos[:,1], c= abs(Etot_morphed - Etot_sim)/Etot_sim, cmap = 'jet')
plt.colorbar()
plt.show()




    