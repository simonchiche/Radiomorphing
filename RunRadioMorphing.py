import numpy as np
import glob
#import grand_radiomorphing
from coreRadiomorphing_ground import process


def run(energy, zenith, azimuth, distplane, simulations):
    # Settings of the radiomorphing
    
    # reference path
    #data_dir = join(root_dir, "examples", "data")
    # folder containing your refernence shower simulations
    sim_dir = glob.glob("./Simulations/*.hdf5")
    # folder which will contain radio morphed traces afterwards
    out_dir = glob.glob("./OutputDirectory")
    # list of antenna positions you would like to simulate, stored in out_dir in the best case
    #antennas = glob.glob("./DesiredPositions/AntennasCoordinates.txt") 
    
    print(zenith, azimuth, distplane, simulations)

    # definition of target shower parameters
    
    shower = {
        "primary" : "Iron",        # primary (electron, pion)
        "energy" : energy,               # EeV
        "zenith" : 180 - zenith,               # deg (GRAND frame)
        "azimuth" : (180 + azimuth)%360,                # deg (GRAND frame)
        "injection" : 1e5,    # m (injection height in the local coordinate system)
        "altitude" : 1000.,
        "fluctuations" : False,  # enable shower to shower fluctuations
        "dplane" : distplane,
        "simulation" : simulations
        }   # m (alitude oj injection with respect to sealevel, 
                               #not necessarily eqivalent to injection height)
    # Perform the radiomorphing
    #grand_radiomorphing.process(sim_dir, shower, antennas, out_dir)
    ILDFvxb, ILDFvxvxb, IntTot, krho_geo, Ref_zenith, Target_zenith = process(sim_dir, shower, out_dir)
    
    return ILDFvxb, ILDFvxvxb, IntTot, krho_geo, Ref_zenith, Target_zenith


if __name__ == "__main__":
    run()
