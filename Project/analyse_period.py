import log
import numpy as np
import scipy.constants as const
import copy
import math
import pandas as pd
from EMField import EMField
from ChargedParticle import ChargedParticle
from ProtonBunch import ProtonBunch

"""
This file is an independent file that tests the simulated time period of a proton in a cyclotron against 
the expected time period given by:

T = 2*pi*m / (qB)

If you do not have the csv containing approximately 100 revolutions, the simulation will run and generate
the csv file for you. It will then print the mean time period and its error (one standard deviation) and the
time period predicted by the equation above. Finally it will print, the ratio of the two time periods; the 
simulated time period divided by the theoretical one.
"""

field = EMField([0.1,0,0], [0,0,1.6*10**(-5)])
proton = ChargedParticle('proton', const.m_p, const.e, [0,0,0], [3000,0,0])
protons = ProtonBunch(100,1) # create a random bunch obeject
protons.bunch = [proton] # overwrite bunch atrribute to to remove the pseudorandom characteristics
field.setFrequency(protons)
theoretical_period = 2*const.pi*proton.mass / (proton.charge*field.magneticMag())

def generate_file(proton,field):
    time, deltaT, duration = 0, 10**(-5), 0.0041*100

    timeSeries = []
    data_csv = []
    revolution = 0
    data_csv.append([revolution,time,0,[0,0,0]]) # ensure data always has one element to avoid index error on line 46
    log.logger.info('starting simulation')
    while time <= duration:
        time += deltaT
        timeSeries.append(time)
        previous_bunch = copy.deepcopy(protons)
        field.getAcceleration(protons.bunch, time, deltaT)
        proton.update(deltaT,field,time)

        current_bunch = copy.deepcopy(protons)

        if previous_bunch.averagePosition()[0] < 0 and current_bunch.averagePosition()[0] >= 0: # is the proton crossing the y-axis in the clockwise direction
            log.logger.info('revolution detected')
            period = time - (data_csv[-1])[1]
            revolution += 1
            data_csv.append([revolution,time,period,current_bunch.averagePosition()])
    
    log.logger.info('simulation finished')

    log.logger.info('converting to pandas dataframe')
    df = pd.DataFrame(data_csv, columns=['Revolution','Time','Measured Period','Position'])
    df.to_pickle('period_data.csv')
    log.logger.info('data pickled')
    
    return pd.read_pickle('period_data.csv')

try:
    df = pd.read_pickle('period_data.csv')
except FileNotFoundError as err:
    df = generate_file(protons,field)

print(df)
df.drop([df.index[0]], inplace=True) 
average_period = df['Measured Period'].mean() # seconds
error = df['Measured Period'].std() # seconds
print('time period from simulation: {0} ± {1} s'.format(round(average_period,4),round(error,4)))
print('time period from theory: {0} s'.format(round(theoretical_period,4)))
print('raito of simulated time period to theoretical: {0}'.format(round((average_period/theoretical_period),4)))