import log
from abc import ABC, abstractmethod
from typing import Union, List
import numpy as np
import math
import copy
from statistics import mean
import scipy.constants as const
from ChargedParticle import ChargedParticle

class Bunch(ABC):
    """
    This abstract base class provides the blueprint for subclasses that will generate a bunch of
    ChargedParticle objects. 

    Attributes:
    -----------
    average_kinetic_energy: float, units: eV
        This value represents the average kinetic energy of the particle bunch. It is used as the mean
        when sampling kinetic energies from a Gaussian. 
    
    bunch_number: int
        The number of particles in the bunch.
    
    positionSigma: float, units: m
        The initial standard deviation of all the particle positions in the bunch, defaults to 1cm.
        This class assumes the same inital position standard deviation in x,y and z.

    bunch: list
        Calls on the abstract createBunch method to return the list that will be assigned to this attribute.

    Concrete Methods:
    -----------------
    createBunch
        Returns a list of Charged Particle objects representing a bunch of charged particles.

    assignPositions
        Returns an array of ndarrays, sampled from a normal distribution with a mean of [0,0,0] m.
        
    distributeEnergies 
        Returns an array of kinetic energies (in eV), sampled from anormal distribution with the 
        mean (AverageKinetic) which is passed into the child classes.
    
    assignVelocities
        Returns a list of the initial 3D velocity vectors for all the particles in the bunch in m/s.
    
    averagePosition
        Returns the average position of a particle in a bunch, units: m.
    
    averageVelocity
        Returns the average velocity of a particle in a bunch, units: m/s.

    KineticEnergy(total=False)
        Takes an optional arguement "total" (defaults to False). Returns the 
        average kinetic energy for a particle in a bunch (eV). If total is True then it returns
        the kinetic energy of the entire bunch.

    momentum(total=False)
        Takes an optional arguement "total" (defaults to False). Returns the average 
        three-momentum for a particle in a bunch (kg m/s). If total is True then it returns the 
        momentum of the entire bunch.

    gamma 
        Returns the Lorentz factor of the bunch, using its average velocity.

    positionSpread 
        Returns the standard deviation of the x-positions, y-positions and z-positions of 
        all the particles in the bunch as 3D numpy array.
    
    energySpread 
        Returns the standard deviation of all of the particles' kinetic energies.
    
    adaptiveStep 
        Reduces the integrator's time step by a factor of 100 if any of the particles 
        in the bunch are approaching, or still in the electric field.

    update 
        Sets the integrator to be used to update the position and velocity of every particle
        in the bunch 
    """

    conversion = const.physical_constants['electron volt'][0] # eV <=> joules

    @abstractmethod
    def __init__(self, particleName: str, particleMass: Union[int,float], particleCharge: Union[int,float,],
                AverageKinetic: Union[int,float], particleNumber: int = 3, positionSigma: Union[int,float] = 0.01) -> None:
        self._particleName = particleName # protect these attributes as they should not change 
        self._particleMass = particleMass
        self._particleCharge = particleCharge
        self.bunchName = self._particleName + '-bunch'
        self.bunch_number = int(particleNumber)

        self.__average_kinetic_energy= float(AverageKinetic) # make these attributes private so they cannot be accessed outside the class
        self.__positionSigma = float(positionSigma)
        
        self.bunch = self.createBunch()

        super(Bunch, self).__init__()

    def __repr__(self) -> None:
        return 'Bunch Type: {0}, Number of Particles: {1}'.format(self._particleName,self.bunch_number)

    def __add__(self, other):
        if not isinstance(other, self.__class__):
            raise TypeError('Cannot add bunches of different types together.')
        new_bunch = copy.deepcopy(self)
        new_bunch.bunch += other.bunch # add the bunches of the two passed Bunch objects together
        new_bunch.bunch_number += other.bunch_number
        new_bunch.bunchName = 'combined ' + other.bunchName
        for particle in new_bunch.bunch:
            new_name = particle.name[:-1] + str(new_bunch.bunch.index(particle)+1)
            particle.name = new_name
        return new_bunch # return a Bunch object made up of the two passed bunches

    def createBunch(self) -> List['ChargedParticle']:
        """
        This method returns a list of charged particle objects to represent a bunch of particles. This list
        is then assigned to the bunch attribute of the subclass' instance.
        """

        positions = self.assignPositions() # get the positions for all the particles in the bunch
        velocities = self.assignVelocities() # get the initial velocities of particles 
        particle_bunch= []
        i = 0
        log.logger.info('generating bunch')
        for (j,k) in zip(positions,velocities):
            i += 1
            particle_bunch.append(ChargedParticle(self._particleName+'-'+str(i), self._particleMass, 
            self._particleCharge, j, k))
        log.logger.info('bunch generated')
        return particle_bunch

    def assignPositions(self) -> np.ndarray:
        """
        This method returns a list of 3D position vectors, each ordinate in every vector is sampled from
        a Gaussian distribution and the z-ordinate in every vector is set to 0.
        """

        mu = 0. # the mean position of the bunch is the origin
        sigma = self.__positionSigma
        positions = np.random.normal(mu, sigma, (self.bunch_number,3))
        for i in positions:
            i[2] = 0. # set all z values to zero
        return positions

    def distributeEnergies(self) -> np.ndarray:
        """
        Using the inputted mean kinetic energy, this method returns a list of kinetic energies, sampled 
        from a Gaussian distribution. It takes the standard deviation as 1% of the inputted mean. If a 
        zero or negative energy is sampled it is repeatedly resampled until the value is positive.
        """

        mu = self.__average_kinetic_energy
        sigma = 0.01 * mu
        energies = np.random.normal(mu, sigma, self.bunch_number)
        while all([i>0 for i in energies]) is not True: # resample non-zero or negative energies
            for energy in energies:
                energies_list = list(energies)
                if energy <= 0:
                    energies[energies_list.index(energy)] = np.random.normal(mu, sigma)
        return energies

    def assignVelocities(self) -> list:
        """
        Using the sampled kinetic energies in the distributeEnergies method, this method will convert them
        to linear speeds and return a list of velocity vectors (in m/s) where the linear speeds make up the 
        x-component of these velocity vectors.
        """

        energies = self.distributeEnergies() # get the kinetic energies
        def gamma(Ke: float) -> float: return 1 + (Ke*self.conversion)/(self._particleMass*const.c*const.c) # function that returns gamma, given kinetic energy
        def speed(y: float) -> float: return const.c*math.sqrt(1-1/(y*y)) # function that returns a linear speed, given gamma
        return [[speed(gamma(i)),0,0] for i in energies]

    def averagePosition(self) -> np.ndarray:
        """
        Returns the average position of the bunch as 3D numpy array, units: m.
        """

        return np.array(np.mean([i.position for i in self.bunch],axis=0),dtype=float)

    def averageVelocity(self) -> np.ndarray:
        """
        Returns the average velocity of the bunch as 3D numpy array, units: m/s.
        """

        return np.array(np.mean([i.velocity for i in self.bunch],axis=0),dtype=float)

    def KineticEnergy(self, total: bool = False) -> float:
        """
        Returns either the average kinetic energy of particle in the bunch or the total momentum of the 
        bunch. The returned energy has units of eV.

        Parameters:
        -----------
        total: Boolean, optional
            Determines whether or not the average particle kinetic energy is returned or the bunch's total
            kinetic energy (defualts to False)
        """

        if total == False:
            return mean([i.KineticEnergy() for i in self.bunch])/self.conversion # units: eV
        return sum([i.KineticEnergy() for i in self.bunch])/self.conversion # units: eV

    def momentum(self, total: bool = False) -> float:
        """
        Returns either the average momentum of particle in the bunch or the total momentum of the bunch. 
        The returned value has units of kg m/s.

        Parameters:
        -----------
        total: Boolean, optional
            Determines whether or not the average particle momentum is returned or the bunch's total
            momentum (defualts to False)
        """

        if total == False:
            return np.array(np.mean([i.momentum() for i in self.bunch],axis=0),dtype=float) # units: kg m/s
        return np.array(np.sum([i.momentum() for i in self.bunch],axis=0),dtype=float) # units: kg m/s

    def positionSpread(self) -> np.ndarray:
        """
        Returns the standard deviation in the x,y,z positions of all particles in the bunch as a 3D numpy
        array, units: m. 
        """

        return np.array(np.std([i.position for i in self.bunch],axis=0),dtype=float)

    def energySpread(self) -> float:
        """
        Returns the standard deviation of the kinetic energy for all the particles in a bunch.

        Note that the returned energy is in eV.
        """

        return np.std([i.KineticEnergy() for i in self.bunch])/self.conversion # units: eV

    def gamma(self) -> float:
        """
        Returns the Lorenz factor for bunch, using the magnitude of the average velocity.
        """

        speed = np.linalg.norm(self.averageVelocity())
        return 1/(math.sqrt(1-(speed*speed)/(const.c*const.c)))

    def adaptiveStep(self, deltaT: Union[int,float], field: 'EMField') -> Union[int,float]:
        """
        For any integrator that is updating a bunch's positions and velocities, this method will reduce the
        the step size in the integrator by a factor of 100 if any of the particles in a bunch are in the 
        electric field or within 10% of the it's boundaries.

        Parameters:
        -----------
        deltaT: float/int
            The timestep the integrator is currently using.
        
        field: EMField object
            The electromagnetic field present in the simulation that the particle bunch is travelling through.
        """

        lowerBound = field.electricLowerBound - 0.1*abs(field.electricLowerBound)
        upperBound = field.electricUpperBound + 0.1*abs(field.electricUpperBound)
        electric_influence = [lowerBound<=i.position[0]<=upperBound for i in self.bunch]
        if any(electric_influence):
            return deltaT*0.01
        else:
            return deltaT

    def update(self, deltaT: Union[int,float], field: 'EMField', time: Union[int,float], set_method: int = 3) -> None:
        """
        This method loops through every particle in the bunch and calls for them to be updated. The user
        can select which update method to used by passing in an integer value from 0-3. If an invalid
        parameter is passed in the fourth order Runge-Kutta method will be used.

        Parameters:
        -----------
        deltaT: float/int
            The timestep the integrator is currently using.

        field: EMField object
            The electromagnetic field present in the simulation that the particle bunch is travelling through.
        
        time: float/int
            The current value of time in the simulation.
        
        set_method: int, optional
            Determines which update method is used; Euler(0), Euler-Cromer(1), velocity Verlet(2), fourth
            order Runge-Kutta(3) (defaults to fourth order Runge-Kutta).
        """

        if set_method == 0:
            for particle in self.bunch:
                particle.euler(deltaT)
        elif set_method == 1:
            for particle in self.bunch:
                particle.eulerCromer(deltaT)
        elif set_method == 2:
            for particle in self.bunch:
                particle.velocityVerlet(deltaT,field,time)
        elif set_method == 3:
            for particle in self.bunch:
                particle.RungeKutta4(deltaT,field,time)
        else:
            log.logger.warning('Invalid set_method parameter, defaulting to fourth order Runge-Kutta')
            for particle in self.bunch:
                particle.RungeKutta4(deltaT,field,time)