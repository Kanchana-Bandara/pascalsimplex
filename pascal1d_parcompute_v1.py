#Pan-Arctic Behavioural and Life History Simulator for Calanus 
#1D configuration - CPU parallelized 

import numpy as np
import pathlib
import netCDF4 as nc
import multiprocessing
import time
import os
import hashlib 

#1D environmental submodel
#-------------------------
#setting up directories and paths
inputpath = pathlib.Path("/home/kanchana/research/migratorycrossroads/pascalsimplex/inputdata")

#loading netcdf4 file
envdataset = nc.Dataset(filename = inputpath / "pascalsimplex_env1d.nc", mode = "r")

#inspection 
#print(envdataset)

#extracting environmental data variables
temperature = np.transpose(envdataset.variables["temperature"][:,:])
irradiance = np.transpose(envdataset.variables["irradiance"][:, :])
food1conc = np.transpose(envdataset.variables["chla"][:, :])

#extracting dimensionality variables
depth = envdataset.variables["depth"][:]
depth_s = np.array(np.where(depth <= 10)).squeeze()

#multiprocessor attributes
#-------------------------
nworkers = int(multiprocessing.cpu_count())
print(f"available processor nodes: {nworkers}")

#random seed generator (reusable in any worker-specific circumstances)
#---------------------------------------------------------------------
def generate_unique_seed(subpopulationid):
    
    # Combine time, process ID, and currentsubpopulation to create a unique seed
    unique_str = f"{time.time()}_{os.getpid()}_{subpopulationid}_{os.urandom(4)}"
    
    # Use hashlib to hash the string and convert it into an integer
    seed = int(hashlib.sha256(unique_str.encode('utf-8')).hexdigest(), 16) % (2**32)
    
    return seed

#end def

#boundary conditions
#-------------------
ntime = 1460
ndepth = len(depth)
nsupindividuals = 1000
#nvirtualindividualspersupindividual = 1000
nsubpopulations = nworkers
#totalpopulationsize = nsubpopulations * nsupindividuals * nvirtualindividualspersupindividual
seedinglevel = 100

#other variables and attributes
developmentalcoefficient = np.array([595.00, 388.00, 581.00]).astype(np.float32)

#three-tier iterative computations
#TIER-3 is paralellized in an "apply()"" style

def simulator(currentsubpopulation):
    
    #1_initializing local variables
    #-------------------------------
    
    #1.1_initial conditions
    timecondition = True
    currenttime = -1
    currentyear = 0
    
    #1.2_state variables
    #a copy of each of these is handled by each worker, which are parsed as 2D outputs at the end
    lifestate = np.zeros(nsupindividuals).astype(np.int32)
    developmentalstage = np.repeat(-1, nsupindividuals).astype(np.int32)
    age = np.repeat(-2, nsupindividuals).astype(np.int32)
    zpos = np.zeros(nsupindividuals).astype(np.int32)
    thermalhistory = np.repeat(-2.00, nsupindividuals).astype(np.float32)
      
    #3_iterative computing
    #--------------------- 
    #NB:TIER-1 (super-individuals) and TIER-2 (time) (TIER-3: subpopulation is applied across the workers)
    
    #3.1_TIER-2: time iteration
    #..........................
    while timecondition:
        
        #processing of time
        #advance the clock
        currenttime += 1
        
        #recycle time at the end of a calendar year
        if currenttime > ntime - 1:
            
            currenttime = 0
            currentyear += 1
            
        #end if
        
        #seeding and/or spawning
        #seeding
        nspaces = nsupindividuals - np.sum(lifestate)
        nseeds = seedinglevel
    
        #seeding happens only if there are sufficient spaces in the subpopulation to hold the seeds
        #this checks for empty spaces in the current subpopulation
        #the else condition is not written because no action is needed when there is no seeding to be performed
        if nspaces >= nseeds and currentyear == 0:
            
            #returns the identities (index positions) of the empty spaces in the subpopulation array
            spacesid = np.array(np.where(lifestate == 0)).squeeze()
            #selects a subset of index positions from the "spacesid" array to seed
            replacementid = np.random.choice(a = spacesid, size = nseeds, replace = False)
            
            #seeding and initializing of seeded state variables
            #variables that do not need randomization
            lifestate[replacementid] = 1
            developmentalstage[replacementid] = 0
            age[replacementid] = -1
            thermalhistory[replacementid] = 0.00
            
            #variables that require random initialization 
            #require a worker-specific seed generation (happens at seeding or spawning level)
            rsd = generate_unique_seed(subpopulationid = currentsubpopulation)
            np.random.seed(rsd)
            zpos[replacementid] = np.random.choice(a = depth_s, size = nseeds, replace = True)
            
        #end if
        
        #3.2_TIER-1: super-individual iteration
        #.......................................
        
        #for efficiency, then TIER-1 loop runs only on living super-individuals
        livingsupindividuals = np.array(np.where(lifestate == 1)).squeeze()
        nlivingsupindividuals = livingsupindividuals.size
        
        #run the TIER-1 loop only when living super individuals are present
        #otherwise, no action is taken and hence the else condition is not written
        if nlivingsupindividuals > 0:
            
            for currentsupindividual in livingsupindividuals:
                
                #developmental stage segragation
                currentdevelopmentalstage = developmentalstage[currentsupindividual]
                currentlifestate = lifestate[currentsupindividual]
                
                #these are eggs and non-feeding naupliar stages
                #includes Egg, N1 and N2
                if currentdevelopmentalstage <= 2:
                    
                    #update age
                    currentage = age[currentsupindividual]
                    currentage += 1
                    
                    #this returns the binned vertical position
                    #and its corresponding index location
                    currentzpos = zpos[currentsupindividual]
                    currentzidx = np.argmin(np.abs(depth == currentzpos))
                                        
                    #extract environmental variables
                    currenttemperature = temperature[currentzidx, currenttime]
                    currentthermalhistory = thermalhistory[currentsupindividual]
                    currentthermalhistory = (currentthermalhistory + currenttemperature) / 2.00
                    
                    thermalhistory[currentsupindividual] = currentthermalhistory
                    age[currentsupindividual] = currentage
                    
                    #process GDSR (this will be a separate function)
                    currentdevelopmentaltime = developmentalcoefficient[currentdevelopmentalstage] * (currentthermalhistory + 9.11) ** -2.05
                    
                    #developmental stage advancement
                    if currentage >= currentdevelopmentaltime:
                    
                        currentdevelopmentalstage += 1
                        developmentalstage[currentsupindividual] = currentdevelopmentalstage
                    
                    #end if
                    
                    #pseudo breaking condition (artificial)
                    if currentdevelopmentalstage >= 3:
                        
                        currentlifestate = 0
                        lifestate[currentsupindividual] = currentlifestate
                        
                    #end if 
                         
                #end if
                  
            #end for
            
        #end if
        
        if np.sum(lifestate[:]) == 0:
            
            timecondition = False
            
        #end if
        
    #end while
    
    #simulator output
    return currentsubpopulation, lifestate, age, thermalhistory, zpos, developmentalstage
    
#end def

#parallel execution code
if __name__ == '__main__':
    
    #passing the simulator function with the shared array
    with multiprocessing.Pool() as pool:
        
        output = pool.map(simulator, range(nsubpopulations))
        
    #end with

#end if

#parsing the output
subpopulation, lifestate, age, thermalhistory, verticalposition, devstage = zip(*output)

#list object outputs into numpy arrays
subpopulation = np.array(subpopulation)
lifestate = np.array(lifestate)
age = np.array(age)
thermalhistory = np.array(thermalhistory)
verticalposition = np.array(verticalposition)
devstage = np.array(devstage)


