import numpy as np
import glob
from coreRadiomorphing_ground import process


def run():
    # Settings of the radiomorphing
    

    # folder containing your reference shower simulations
    #sim_dir = glob.glob("./Simulations/*.hdf5")
    # folder which will contain radio morphed traces afterwards
    out_dir = glob.glob("./OutputDirectory")
    
    # list of antenna positions you would like to simulate, stored in out_dir in the best case
    #antennas = glob.glob("./DesiredPositions/AntennasCoordinates.txt") 
    
    # definition of target shower parameters
    
    shower = {
        "primary" : "Iron",        # primary (Proton, Iron)
        "energy" : 2.51,           # EeV
        "zenith" : 112.2,          # deg (GRAND frame) the Radio Morphing is tested only for 90 < theta <= 65 degrees
        "azimuth" : 270,           # deg (GRAND frame)
        "injection" : 1e5,         # m (injection height in the local coordinate system)
        "altitude" : 1000.,        # m (alitude oj injection with respect to sealevel, 
        "fluctuations" : False,    # enable shower to shower fluctuations
        "dplane" : 0,              # This option is in development, should be set to 0 for now
        "simulation" : 0#glob.glob("./TargetShowers/*.hdf5")[0] # option dedicated to the debuggging of the code
        }   
    

    # Perform the radiomorphing
    process(shower, out_dir)
    

if __name__ == "__main__":
    run()

