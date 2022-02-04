import log
import numpy as np
import math
import copy
import scipy.constants as const
from typing import Union, Sequence

class Particle:
    """
    Class to represent a point-like particle with the functionality to update its postion and velocity, as
    well as reutrn values such as its Lorentz factor. A ValueError will be raised if the inital speed of the
    particle is greater than the speed of light in a vacuum.

    Attributes:
    -----------
    name: str, optional
        The name of particle an instance is representing (defualts to 'Point').

    position: ndarray, optional 
        The position of the particle in the form [x,y,z] measured in metres (defaults to [0,0,0]).
    
    velocity: ndarray, optional
        The velocity of the particle in the form [vx,vy,vz] measured in m/s (defaults to [0,0,0]).

    acceleration: ndarray, optional
        The acceleration of the particle in the form [ax,ay,az] measured in m/s^2 (defaults to [0,0,0]).

    mass: float, optional
        The mass of particle in kg (defaults to 1.0).
    
    Methods:
    --------
    gamma
        Returns the Lorentz value for the particle object using its current velocity.
    
    KineticEnergy
        Returns the kinetic energy of the particle in joules.
    
    momentum
        Returns the momentum of the particle in kgm/s
    
    euler
        Updates the position and velocity of the particle using the Euler method.
    
    eulerCromer
        Updates the position and velocity of the particle using the Euler-Cromer method.
    
    velocityVerlet
        Updates the position and velocity of the particle using the velocity Verlet method.
    
    RungeKutta4
        Updates the position and velocity of the particle using the fourth order Runge-Kutta method.
    """
    def __init__(self,  Name: str = 'Point', Mass: Union[int,float] = 1.0, Position: Sequence[Union[int,float]] = np.array([0,0,0], dtype=float), 
                Velocity: Sequence[Union[int,float]] = np.array([0,0,0], dtype=float), Acceleration: Sequence[Union[int,float]] = np.array([0,0,0], dtype=float)) -> None:
        self.name = str(Name)
        self.position = np.array(Position,dtype=float)
        self.velocity = np.array(Velocity,dtype=float)
        self.acceleration = np.array(Acceleration,dtype=float)
        self.mass = float(Mass)

        if self.magnitude(self.velocity) >= const.c:
            log.logger.error('%s cannot have a speed greater than the speed of light' % self.name)
            raise ValueError('Velocity cannot be greater than the speed of light.')
        else:
            log.logger.info('%s has been generated' % self.name)

    def __repr__(self) -> None:
        return 'Particle: {0}, Mass: {1:6.3e}, Position: {2}, Velocity: {3}, Acceleration: {4}'.format(self.name,self.mass,self.position, self.velocity,self.acceleration)

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        equality = []
        instance1_values = self.__dict__.values()
        instance2_values = other.__dict__.values()
        for (i,j) in zip(instance1_values, instance2_values):
            if isinstance(i,np.ndarray) and isinstance(j,np.ndarray):
                equality.append(np.array_equal(i,j))
                continue
            equality.append(i==j)
        return all(equality)
    
    def gamma(self) -> float:
        """
        Returns the Lorentz factor for any given Particle object.

        Raises:
        -------
        ValueError
            If the speed of the particle is equal to or greater than the speed of light in a vacuum.
        """
        try:
            lorentz_factor = 1/(math.sqrt(1-(self.magnitude(self.velocity)*self.magnitude(self.velocity))/(const.c*const.c)))
        except ValueError:
            log.logger.error("%s's speed has exceeded the speed of light" % self.name)
            raise ValueError(str(self.name) + "'s speed has exceeded the speed of light")
        except ZeroDivisionError:
            log.logger.error("%s's speed is equal to the speed of light" % self.name)
            raise ZeroDivisionError(str(self.name) + "'s speed is equal to the speed of light")
        else:
            return lorentz_factor

    def magnitude(self, vector: np.ndarray) -> float:
        """
        Returns the magnitude of a vector.

        Parameters:
        -----------
        vector
            The vector whose magnitude will be calculated.
        """

        return np.linalg.norm(vector)

    def KineticEnergy(self) -> float:
        """
        Returns the current kinetic energy of the particle instance in joules.
        """

        return (self.gamma()-1)*self.mass*const.c*const.c
  
    def momentum(self) -> float:
        """
        Returns the current momnetum of the particle instance in kg m/s.
        """

        return self.gamma()*self.mass*self.velocity

    def euler(self, deltaT: Union[int,float]) -> None:
        """
        Updates the particle's position and velocity using the Euler method.

        Parameters:
        -----------
        deltaT
            The time step to be used in the update method, measured in seconds.
        
        Raises:
        -------
        ValueError
            If the updated velocity's magnitude exceeds the speed of light in a vacuum.
        """
        self.position +=  self.velocity*deltaT
        self.velocity +=  self.acceleration*deltaT
        if self.magnitude(self.velocity) >= const.c:
            log.logger.error("%s's speed is equal to or greater than the speed of light" % self.name)
            raise ValueError("%s's speed is equal to or greater than the speed of light" % self.name)

    def eulerCromer(self, deltaT: Union[int,float]) -> None:
        """
        Updates the particle's position and velocity using the Euler-Cromer method.

        Parameters:
        -----------
        deltaT
            The time step to be used in the update method, measured in seconds.

        Raises:
        -------
        ValueError
            If the updated velocity's magnitude exceeds the speed of light in a vacuum.
        """
        self.velocity +=  self.acceleration*deltaT
        self.position +=  self.velocity*deltaT
        if self.magnitude(self.velocity) >= const.c:
            log.logger.error("%s's speed is equal to or greater than the speed of light" % self.name)
            raise ValueError("%s's speed is equal to or greater than the speed of light" % self.name)

    def velocityVerlet(self, deltaT: Union[int,float], field: 'EMField', time: Union[int,float]) -> None:
        """
        Updates the particle's position and velocity using the velocity Verlet method.

        Parameters:
        -----------
        deltaT
            The time step to be used in the update method, measured in seconds.

        field
            The electromagnetic field that is accelerating the particle.
        
        time
            The current value of time in the simulation, in seconds.

        Raises:
        -------
        ValueError
            If the updated velocity's magnitude exceeds the speed of light in a vacuum.
        """
        particle_1, particle_2 = copy.deepcopy(self), copy.deepcopy(self)
        self.position += self.velocity*deltaT + 0.5*self.acceleration*deltaT*deltaT

        particle_2.position = self.position
        particle_2.velocity += deltaT*self.acceleration
        nextTime = time + deltaT
        field.getAcceleration([particle_2],nextTime,deltaT)
        particle_1.velocity = self.velocity + 0.5*deltaT * (self.acceleration + particle_2.acceleration) 

        field.getAcceleration([particle_1],nextTime,deltaT)
        self.velocity += 0.5*deltaT * (self.acceleration + particle_1.acceleration)
        if self.magnitude(self.velocity) >= const.c:
            log.logger.error("%s's speed is equal to or greater than the speed of light" % self.name)
            raise ValueError("%s's speed is equal to or greater than the speed of light" % self.name)

    def RungeKutta4(self, deltaT: Union[int,float], field: 'EMField', time: Union[int,float]) -> None:
        """
        Updates the particle's position and velocity using the fourth order Runge-Kutta method.

        Parameters:
        -----------
        deltaT
            The time step to be used in the update method, measured in seconds.

        field
            The electromagnetic field that is accelerating the particle.
        
        time
            The current value of time in the simulation, in seconds.

        Raises:
        -------
        ValueError
            If the updated velocity's magnitude exceeds the speed of light in a vacuum.
        """
        particle = copy.deepcopy(self)
        field.getAcceleration([particle],time,deltaT)
        k1_v = deltaT*particle.acceleration
        k1_x = deltaT*particle.velocity

        particle2 = copy.deepcopy(self)
        particle2.position += k1_x*0.5
        particle2.velocity += k1_v*0.5
        time_2 = time + 0.5*deltaT
        field.getAcceleration([particle2],time_2,deltaT)
        k2_v = deltaT*particle2.acceleration
        k2_x = deltaT*particle2.velocity

        particle3 = copy.deepcopy(self)
        particle3.position += k2_x*0.5
        particle3.velocity += k2_v*0.5
        time_3 = time_2
        field.getAcceleration([particle3],time_3,deltaT)
        k3_v = particle3.acceleration*deltaT
        k3_x = particle3.velocity*deltaT

        particle4 = copy.deepcopy(self)
        particle4.position += k3_x
        particle4.velocity += k3_v
        time_4 = time + deltaT
        field.getAcceleration([particle4],time_4,deltaT)
        k4_v = particle4.acceleration*deltaT
        k4_x = particle4.velocity*deltaT

        self.velocity += 1/6 * (k1_v + 2*k2_v + 2*k3_v + k4_v)
        self.position += 1/6 * (k1_x + 2*k2_x + 2*k3_x + k4_x)
        if self.magnitude(self.velocity) >= const.c:
            log.logger.error("%s's speed is equal to or greater than the speed of light" % self.name)
            raise ValueError("%s's speed is equal to or greater than the speed of light" % self.name)