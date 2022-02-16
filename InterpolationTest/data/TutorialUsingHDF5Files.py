import matplotlib.pyplot as plt
import hdf5fileinout as hdf5io
import numpy as np
import glob
import os
import h5py

################################################################################
# Test of Matias functions for simulation and hdf5 files handling (Valentin)
################################################################################

filename_all = glob.glob('*.hdf5')

for i in range(1):#len(filename_all)):
    
    #Shower path
    simu_path = './' + filename_all[i]
    InputFilename = simu_path
    filehandle = h5py.File(InputFilename, 'r')
    filename_cut = os.path.splitext(filename_all[i])[0]

    #Shower event access
    RunInfo = hdf5io.GetRunInfo(InputFilename)
    #print("#RunInfo --> ", RunInfo)
    
     
    NumberOfEvents = hdf5io.GetNumberOfEvents(RunInfo)
    #print("#NumberOfEvents --> ", NumberOfEvents)
    EventNumber = NumberOfEvents-1 #Pourquoi ?
    EventName = hdf5io.GetEventName(RunInfo,EventNumber)
    #print("#EventName --> ", EventName)
    
    #print("******")
    #print("Press Enter")
    #raw_input()
    #print("******")
    
    #All simulations info
    ShowerSimInfo = hdf5io.GetShowerSimInfo(InputFilename,EventName)
    #print("#ShowerSimInfo --> ", ShowerSimInfo)
    #CPUTime = hdf5io.GetCPUTime(ShowerSimInfo)                                     #needs latest simulated events
    #print("#CPUTime --> ", CPUTime)
    
    #print("******")
    #print("Press Enter")
    #raw_input()
    #print("******")
    
    #Additional simulation parameters
    SignalSimInfo = hdf5io.GetSignalSimInfo(InputFilename,EventName)
    #print("#SignalSimInfo --> ", SignalSimInfo)
    TimeBinSize = hdf5io.GetTimeBinSize(SignalSimInfo)
    #print("#TimeBinSize --> ", TimeBinSize)
    TimeWindowMin = hdf5io.GetTimeWindowMin(SignalSimInfo) #Pourquoi négatif
    #print("#TimeWindowMin --> ", TimeWindowMin)
    TimeWindowMax = hdf5io.GetTimeWindowMax(SignalSimInfo)
    #print("#TimeWindowMax --> ", TimeWindowMax)
    
    #print("******")
    #print("Press Enter")
    #raw_input()
    #print("******")
    
    
    #Shower parameters
    Zenith = hdf5io.GetEventZenith(RunInfo,EventNumber)
    #print("#Zenith --> ", Zenith)
    Azimuth = hdf5io.GetEventAzimuth(RunInfo,EventNumber)
    #print("#Azimuth --> ", Azimuth)
    Primary = hdf5io.GetEventPrimary(RunInfo,EventNumber)
    #print("#Primary --> ", Primary)
    Energy = hdf5io.GetEventEnergy(RunInfo,EventNumber)
    #print("#Energy --> ", Energy)
    XmaxDistance = hdf5io.GetEventXmaxDistance(RunInfo,EventNumber)
    #print("#XmaxDistance --> ", XmaxDistance)
    SlantXmax = hdf5io.GetEventSlantXmax(RunInfo,EventNumber) # Slant ?
    #print("#SlantXmax --> ", SlantXmax)
    Energy = hdf5io.GetEventEnergy(RunInfo,EventNumber)
    #print("#Energy --> ", Energy)
    HadronicModel = hdf5io.GetEventHadronicModel(RunInfo,EventNumber)
    #print("#HadronicModel --> ", HadronicModel)
    
    #print("******")
    #print("Press Enter")
    #raw_input()
    #print("******")
    
    #Shower info
    EventInfo = hdf5io.GetEventInfo(InputFilename,EventName) #Energie in neutrino ?
    #print("#EventInfo --> ", EventInfo)
    XmaxPosition = hdf5io.GetEventXmaxPosition(EventInfo)
    #print("#XmaxPosition --> ", XmaxPosition)
    BFieldIncl = hdf5io.GetEventBFieldIncl(EventInfo) #BfieldIncl/Bfieldecl
    #print("#BFieldIncl --> ", BFieldIncl)      
    BFieldDecl = hdf5io.GetEventBFieldDecl(EventInfo)
    #print("#BFieldDecl --> ", BFieldDecl)
    GroundAltitude = hdf5io.GetGroundAltitude(EventInfo)
    #print("#GroundAltitude --> ", GroundAltitude) #Au niveau du sol ?
    
    #print("******")
    #print("Press Enter")
    #raw_input()
    #print("******")
    
    #Antannas info
    AntennaInfo = hdf5io.GetAntennaInfo(InputFilename,EventName)
    #print("#AntennaInfo --> ", AntennaInfo) #Cross Check,Slope A, Slope B
          
    AntennaInfoMeta = AntennaInfo.meta
    #print("#AntennaInfoMeta --> ", AntennaInfoMeta) #je ne comprends pas ça
          
    IDs = AntennaInfo['ID'].data
    #print("#IDs --> ", IDs)
    X = AntennaInfo['X'].data 
    Y = AntennaInfo['Y'].data 
    #print("#X --> ", X)   
        
    
    # AntennaInfo4 = hdf5io.GetAntennaInfo4(InputFilename,EventName)                #needs the table written
    # print("#AntennaInfo4 --> ", AntennaInfo4)
    NumberOfAntennas = hdf5io.GetNumberOfAntennas(AntennaInfo)
    #print("#NumberOfAntennas --> ", NumberOfAntennas)
    IDs_bis = hdf5io.GetAntIDFromAntennaInfo(AntennaInfo)
    #print("#IDs_bis --> ", IDs_bis)
    X = hdf5io.GetXFromAntennaInfo(AntennaInfo)
    #print("#X --> ", X)
    Y = hdf5io.GetYFromAntennaInfo(AntennaInfo)
    #print("#Y --> ", Y)
    Z = hdf5io.GetZFromAntennaInfo(AntennaInfo)
    #print("#Z --> ", Z)
    
    Positions = hdf5io.GetAntennaPositions(AntennaInfo)
    #print("#Positions --> ", Positions)
    
    #print("******")
    #print("Press Enter")
    #raw_input()
    #print("******")
    
    #One antenna info
    AntennaNumber = 12
    Position = hdf5io.GetAntennaPosition(AntennaInfo,AntennaNumber)
    #print("#Position --> ", Position)
    Slope = hdf5io.GetAntennaSlope(AntennaInfo,AntennaNumber)
    #print("#Slope --> ", Slope)
    AntennaInfo_bis = hdf5io.GetAntennaInfoFromEventInfo(EventInfo,0)
    #print("#AntennaInfo_bis --> ", AntennaInfo_bis)
    
    
    
    #print("******")
    #print("Press Enter")
    #input()
    #print("******")
    
    #Traces
    
    AntennaID = IDs_bis[0]
    Efield_trace = hdf5io.GetAntennaEfield(InputFilename,EventName,AntennaID,OutputFormat="numpy")
    time_sample = len(Efield_trace[:,0])
    
    trace_x = np.zeros([time_sample,NumberOfAntennas])
    trace_y = np.zeros([time_sample,NumberOfAntennas])
    trace_z = np.zeros([time_sample,NumberOfAntennas])
    Time = np.zeros([time_sample,NumberOfAntennas])

    trace_x2 = np.zeros([time_sample,NumberOfAntennas])
    trace_y2 = np.zeros([time_sample,NumberOfAntennas])
    trace_z2 = np.zeros([time_sample,NumberOfAntennas])
    Time2 = np.zeros([time_sample,NumberOfAntennas])
        
    
    
    print(NumberOfAntennas)
    for i in range(NumberOfAntennas):
    
        AntennaID = IDs_bis[i]
        #Efield_trace2= hdf5io.GetAntennaEfield(InputFilename,EventName,AntennaID,OutputFormat="numpy")
        path = str(EventName)+"/AntennaTraces/"+str(AntennaID)+"/efield"
        Efield_trace = filehandle[path]
        #Time2[:,i] = Efield_trace2[:,0]
        Time[:,i] = Efield_trace['Time']

        #trace_x2[:,i] = Efield_trace2[:,1]
        #trace_y2[:,i] = Efield_trace2[:,2]
        #trace_z2[:,i] = Efield_trace2[:,3]
        trace_x[:,i] = Efield_trace['Ex']
        trace_y[:,i] = Efield_trace['Ey']
        trace_z[:,i] = Efield_trace['Ez']
        np.shape(Efield_trace)
               
    
    Traces = np.transpose(np.concatenate((np.transpose(Time), np.transpose(trace_x), np.transpose(trace_y), np.transpose(trace_z))))
    np.save(filename_cut + '_' + 'trace',  Traces)
    
    
    #Voltages_trace = hdf5io.GetAntennaVoltage(InputFilename,EventName,AntennaID,OutputFormat="numpy")
    #print("See plots at the end")
    #Filtered_trace = hdf5io.GetAntennaFilteredVoltage(InputFilename,EventName,AntennaID,OutputFormat="numpy")
    #print("See plots at the end")
    
    # Slopes = hdf5io.GetSlopesFromTrace(Trace)                                     #no yet working
    # print("Second methode for getting the slopes of the antenna (from meta of Table)")
    # print("#Slopes --> ", Slopes)
    
    #print("******")
    #print("Press Enter")
    #raw_input()
    #print("******")
    
    # P2PInfo = hdf5io.GetAntennaP2PInfo(InputFilename,EventName)                   #needs tha table already written
    # print("#P2PInfo --> ", P2PInfo)
    p2p = hdf5io.get_p2p_hdf5(InputFilename,antennamax='All',antennamin=0,usetrace='efield')
    #print("#p2p --> ", p2p)
    peaktime, peakamplitude = hdf5io.get_peak_time_hilbert_hdf5(InputFilename, antennamax="All",antennamin=0, usetrace="efield", DISPLAY=False)
    #print("#peak time hilbert --> ", peaktime)
    #print("#peak amplitude hilbert --> ", peakamplitude)
    
    #print("******")
    #print("Press Enter")
    #raw_input()
    #print("******")
    
    #Creations/Savings
    # hdf5io.CreateRunInfoMeta(RunName)
    # hdf5io.CreateEventInfoMeta(RunName,EventNumber,EventInfo,ShowerSimInfo,SignalSimInfo,AntennaInfo,AntennaTraces,NLongitudinal,ELongitudinal,NlowLongitudinal,ElowLongitudinal,EdepLongitudinal,LateralDistribution,EnergyDistribution)
    # hdf5io.CreateEventInfo(EventName,Primary,Energy,Zenith,Azimuth,XmaxDistance,XmaxPosition,XmaxAltitude,SlantXmax,InjectionAltitude,GroundAltitude,Site,Date,FieldIntensity,FieldInclination,FieldDeclination,AtmosphericModel,EnergyInNeutrinos,EventInfoMeta)
    # hdf5io.CreateShowerSimInfoMeta(RunName,EventName,ShowerSimulator)
    # hdf5io.CreateShowerSimInfo(ShowerSimulator,HadronicModel,RandomSeed,RelativeThinning,WeightFactor,GammaEnergyCut,ElectronEnergyCut,MuonEnergyCut,MesonEnergyCut,NucleonEnergyCut,CPUTime,ShowerSimInfoMeta)
    # hdf5io.CreateSignalSimInfoMeta(RunName,EventName,FieldSimulator)
    # hdf5io.CreateSignalSimInfo(FieldSimulator,RefractionIndexModel,RefractionIndexParameters,TimeBinSize,TimeWindowMin,TimeWindowMax,SignalSimInfoMeta)
    # hdf5io.CreatAntennaInfoMeta(RunName,EventName,VoltageSimulator="N/A",AntennaModel="N/A",EnvironmentNoiseSimulator="N/A",ElectronicsSimulator="N/A",ElectronicsNoiseSimulator="N/A")
    # hdf5io.CreateAntennaInfo(IDs, antx, anty, antz, slopeA, slopeB, AntennaInfoMeta, P2Pefield=None,P2Pvoltage=None,P2Pfiltered=None,HilbertPeak=None,HilbertPeakTime=None)
    # hdf5io.CreateAntennaP2PInfo(IDs, AntennaInfoMeta, P2Pefield=None,P2Pvoltage=None,P2Pfiltered=None,HilbertPeakE=None,HilbertPeakV=None,HilbertPeakFV=None,HilbertPeakTimeE=None,HilbertPeakTimeV=None,HilbertPeakTimeFV=None)
    # hdf5io.CreateEfieldTable(efield, EventName, EventNumber, AntennaID, AntennaNumber,FieldSimulator, info={})
    # hdf5io.CreateVoltageTable(voltage, EventName, EventNumber, AntennaID, AntennaNumber, VoltageSimulator, info={})
    #
    # hdf5io.SaveEfieldTable(outputfilename,EventName,antennaID,efield)
    # hdf5io.SaveVoltageTable(outputfilename,EventName,antennaID,voltage)
    # hdf5io.SaveFilteredVoltageTable(outputfilename,EventName,antennaID,filteredvoltage)
    
    ################################################################################
    #PLOTS exemples
    
    #fa, ax = plt.subplots()
    #ax.plot(Efield_trace.T[0,:], Efield_trace.T[1,:], label='X-channel')
    #ax.plot(Efield_trace.T[0,:], Efield_trace.T[2,:], label='Y-channel')
    #ax.plot(Efield_trace.T[0,:], Efield_trace.T[3,:], label='Z-channel')
    #ax.set_xlabel(r"$\rm time\ (ns)$")
    #ax.set_ylabel(r"$\rm \vec{E}\ (\mu V/m)$")
    #ax.set_title(r"$\rm Electric-field\ trace$")
    #ax.legend()
    
    #fb, bx = plt.subplots()
    #bx.plot(Voltages_trace.T[0,:], Voltages_trace.T[1,:], label='X-channel')
    #bx.plot(Voltages_trace.T[0,:], Voltages_trace.T[2,:], label='Y-channel')
    #bx.plot(Voltages_trace.T[0,:], Voltages_trace.T[3,:], label='Z-channel')
    #bx.set_xlabel(r"$\rm time\ (ns)$")
    #bx.set_ylabel(r"$\rm V\ (\mu V)$")
    #bx.set_title(r"$\rm voltage\ trace$")
    #bx.legend()
    
    #fc, cx = plt.subplots()
    #cx.plot(Filtered_trace.T[0,:], Filtered_trace.T[1,:], label='X-channel') #traces filtrees
    #cx.plot(Filtered_trace.T[0,:], Filtered_trace.T[2,:], label='Y-channel')
    #cx.plot(Filtered_trace.T[0,:], Filtered_trace.T[3,:], label='Z-channel')
    #cx.set_xlabel(r"$\rm time\ (ns)$")
    #cx.set_ylabel(r"$\rm V (\mu V)$")
    #cx.set_title(r"$\rm Filtered\ voltage\ trace$")
    #cx.legend()
    
    #fd, dx = plt.subplots()
    #dx.scatter(peaktime, peakamplitude) #peaktime
    #dx.set_xlabel(r"$\rm Peak\ time\ (ns)$")
    #dx.set_ylabel(r"$\rm Peak\ amplitude\ (\mu V/m)$")
    #dx.set_title(r"$\rm Electric-field$")
    
    #plt.show()
    Positions = np.transpose([X,Y,Z])
    np.save(filename_cut + '_' + 'positions', np.transpose([X,Y,Z]))
    np.savetxt(filename_cut + '_' + 'p2p.txt', np.transpose([p2p[3], p2p[0], p2p[1], p2p[2]]))
    Elevation = Zenith - 90
    np.save(filename_cut + '_' + 'parameters', [Azimuth, Elevation, Zenith, BFieldIncl, Energy, time_sample, XmaxPosition[0,0],  XmaxPosition[0,1],  XmaxPosition[0,2]])

