import log
import numpy as np
import scipy.constants as const
import matplotlib.pyplot as plt
import matplotlib
import copy
from EMField import EMField
from ProtonBunch import ProtonBunch

"""
This file generates a file called "cyclotron_data.npz" it contains a time series and four lists containing
copies of a proton bunch in a cyclotron at every time value in the time series. These seperate lists were 
updated using different numerical methods.

The numerical methods are:
    Euler
    Euler-Cromer
    Velocity Verlet
    Fourth order Runge-Kutta
"""

field = EMField([500000,0,0], [0,0,0.07]) 
protons_1 = ProtonBunch(10**(6),3) ; protons_2 = copy.deepcopy(protons_1)
protons_3 = copy.deepcopy(protons_1) ; protons_4 = copy.deepcopy(protons_1)
field.setFrequency(protons_1)

log.logger.info('Initial average kinetic energy: %s eV' % protons_1.KineticEnergy())
log.logger.info('Initial average momentum: %s kg m/s' % protons_1.momentum())
log.logger.info('Initial average position %s m' % protons_1.averagePosition())
log.logger.info('Initial bunch position spread: %s m' % protons_1.positionSpread())
log.logger.info('Initial bunch energy spread: %s eV' % protons_1.energySpread())
    
time, deltaT, duration = 0, 10**(-9), 10**(-6)*3

inital_bunch_1 = copy.deepcopy(protons_1) ; inital_bunch_2 = copy.deepcopy(protons_2)
inital_bunch_3 = copy.deepcopy(protons_3) ; inital_bunch_4 = copy.deepcopy(protons_4)

timeSeries = [0.]
eulerData = [inital_bunch_1] ; cromerData = [inital_bunch_2]
verletData = [inital_bunch_3] ; RK4Data = [inital_bunch_4]

log.logger.info('starting simulation')
while time <= duration:
    dt = protons_4.adaptiveStep(deltaT,field)
    time += dt
    timeSeries.append(time)
    field.getAcceleration(protons_1.bunch, time, dt)
    field.getAcceleration(protons_2.bunch, time, dt)
    field.getAcceleration(protons_3.bunch, time, dt)
    field.getAcceleration(protons_4.bunch, time, dt)
    protons_1.update(dt,field,time,0)
    protons_2.update(dt,field,time,1)
    protons_3.update(dt,field,time,2)
    protons_4.update(dt,field,time,3)
    temp_bunch_1 = copy.deepcopy(protons_1)
    temp_bunch_2 = copy.deepcopy(protons_2)
    temp_bunch_3 = copy.deepcopy(protons_3)
    temp_bunch_4 = copy.deepcopy(protons_4)
    eulerData.append(temp_bunch_1)
    cromerData.append(temp_bunch_2)
    verletData.append(temp_bunch_3)
    RK4Data.append(temp_bunch_4)
log.logger.info('simulation finished')

log.logger.info('writing lists to file')
np.savez('cyclotron_data',time=timeSeries,euler=eulerData,cromer=cromerData,verlet=verletData,rk=RK4Data)
log.logger.info('file writing complete')