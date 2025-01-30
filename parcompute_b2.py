import numpy as np
import pathlib
import netCDF4 as nc
import multiprocessing
import time
import os
import hashlib 

#setting up paths
inputpath = pathlib.Path("/home/kanchana/research/migratorycrossroads/pascalsimplex/inputdata")

#loading netcdf4 file
envdataset = nc.Dataset(filename = inputpath / "pascalsimplex_env1d.nc", mode = "r")

#extracting environmental data variables
temperature = np.transpose(envdataset.variables["temperature"][:,:])
depth = envdataset.variables["depth"][:]

#no. of processors at hand:
nworkers = multiprocessing.cpu_count()
print(f"available no. of compute units: {nworkers}")

#boundary conditions
nsubpopulations = nworkers
nindividuals = 10000
ntime = 1460
totalpopulationsize = nindividuals * nsubpopulations

#state variables

#other variables
developmentalcoefficient = np.array([595.00, 388.00, 581.00]).astype(np.float32)

#generate random seeds
def generate_unique_seed(currentsubpopulation):
    
    # Combine time, process ID, and currentsubpopulation to create a unique seed
    unique_str = f"{time.time()}_{os.getpid()}_{currentsubpopulation}_{os.urandom(4)}"
    
    # Use hashlib to hash the string and convert it into an integer
    seed = int(hashlib.sha256(unique_str.encode('utf-8')).hexdigest(), 16) % (2**32)
    
    return seed

#end def

def simulator(currentsubpopulation):
    
    #conditions  
    timecondition = True
    currenttime = -1
    
    #local state variables   
    lifestate = np.ones(nindividuals).astype(np.int32)
    devstage = np.zeros(nindividuals).astype(np.int32)
    age = np.zeros(nindividuals).astype(np.int32)
    
    #generating a highly unique seed
    unique_seed = generate_unique_seed(currentsubpopulation)
    #seed the random number generator
    np.random.seed(unique_seed)
    
    verticalposition = np.random.randint(low = np.min(depth), high = np.max(depth), size = nindividuals).astype(np.int32)
    
    thermalhistory = np.repeat(0.00, nindividuals).astype(np.float32)
    
    #time loop (TIER-2)
    while timecondition:
        
        #time advances
        currenttime += 1
        
        #recycle time at the end of a calendar year
        if currenttime > ntime -1:
        
            currenttime = 0
        
        #end if
        
        #individual loop (TIER-1)
        for currentindividual in range(nindividuals):
            
            currentlifestate = lifestate[currentindividual]
        
            #processing decision
            if currentlifestate == 1:
                
                currentverticalposition = verticalposition[currentindividual]
                currentdevstage = devstage[currentindividual]
                currentage = age[currentindividual]
                
                currentthermalhistory = thermalhistory[currentindividual]
                
                currentage += 1
                
                currentverticalposition_bin = np.argmin(np.abs(depth - currentverticalposition))
                
                currenttemperaure = temperature[currentverticalposition_bin, currenttime]
                currentthermalhistory = (currentthermalhistory + currenttemperaure) / 2.00
                
                currentdevtime = 4.00 * developmentalcoefficient[currentdevstage] * (currentthermalhistory + 9.11) ** -2.05
                
                #update state variables
                thermalhistory[currentindividual] = currentthermalhistory
                age[currentindividual] = currentage
                
                #developmental stage advancement
                if currentage >= currentdevtime:
                
                    currentdevstage += 1
                    devstage[currentindividual] = currentdevstage
                
                #end if
                
                #breaking condition
                if currentdevstage >= 2:
                
                    currentlifestate = 0
                    lifestate[currentindividual] = currentlifestate
                
                #end if
                  
            #end if
        
        #end for
        
        if np.sum(lifestate[:]) == 0:
        
            timecondition = False
        
        #end if
        
    #end while
    
    return currentsubpopulation, age, thermalhistory, verticalposition, devstage
    
#end def

#parallel execution code
if __name__ == '__main__':
    
    #passing the simulator function with the shared array
    with multiprocessing.Pool() as pool:
        
        output = pool.map(simulator, range(nsubpopulations))
        
    #end with

#end if


#parsing the output
subpopulation, age, thermalhistory, verticalposition, devstage = zip(*output)

subpopulation = np.array(subpopulation)
age = np.array(age)
thermalhistory = np.array(thermalhistory)
verticalposition = np.array(verticalposition)
devstage = np.array(devstage)

