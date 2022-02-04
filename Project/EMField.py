import log
import numpy as np
import scipy.constants as const
import math
from typing import Union, Sequence, List

class EMField:
    """
    Class to represent an  electromagnetic field in a (synchro)cyclotron. It consists of a constant
    magnetic field and a time varying electric field of the form Acos(wt+phi). The electric field 
    only exists across a small region representing the accelerating gap in an accelerator.

    Attributes:
    -----------
    electric: list
        A 3D list containing the electric field's amplitudes and directions of the field the in x,y,z. 
        It will be converted to the time-varying electric field, the electric field has units of N/C or 
        V/m.

    magnetic: list
        A 3D numpy array containing the magnetic field's amplitudes and directions of the field the in 
        x,y,z directions. It has units of teslas.

    electricLowerBound: float
        A float representing the lower boundary in the direction of the electric field's width, in metres.
        It is the first element of the passed in list: ElectricFieldWidth.

    electricUpperBound: float
        A float representing the upper boundary in the direction of the electric field's width, in metres.
        It is the second element of the passed in list: ElectricFieldWidth.

    phase: float
        The phase shift that will be applied to the electric field. Measured in radians.
    
    frequency: float
        The frequency of the electric field, measured in radians per second. This is initially set to zero
        and set later in the main program.


    Methods:
    --------
    electric_mag
        Returns the magnitude of electric field at its peak.
    
    magnetic_mag
        Returns the magnitude of the magnetic field.
    
    set_frequency(bunch)
        Returns the angular frequnecy for the passed in particle bunch in radians per second.
    
    getAcceleration(particleBunch, time, deltaT)
        Updates the acceleration of every particle in a bunch due to the Lorentz force. It uses an anonymous
        function to represent the time-varying electric field. Once the Lorentz force has been calculated,
        using the relativistic form of Newton's second law.
    """

    def __init__(self, ElectricField: Sequence[Union[int,float]] = np.array([0,0,0], dtype=float), 
                MagneticField: Sequence[Union[int,float]] = np.array([0,0,0], dtype=float), 
                ElectricFieldWidth: Sequence[Union[int,float]] = np.array([-0.103,0.103], dtype=float),
                ElectricFieldPhase: Sequence[Union[int,float]] = 0) -> None:
        self.magnetic = np.array(MagneticField,dtype=float)
        self.electric = np.array(ElectricField,dtype=float)
        self.electricLowerBound  = float(ElectricFieldWidth[0])
        self.electricUpperBound  = float(ElectricFieldWidth[1])
        self.phase = float(ElectricFieldPhase)
        self.frequency = 0.
        log.logger.info('electromagentic field generated')

    def __repr__(self) -> None:
        return 'EM Field: \n Electric Field = {0}  \n Magnetic Field = {1}'.format(self.electric,self.magnetic)

    def electricMag(self) -> float:
        """
        Returns the amplitude of the electric field at its peak value in N/m.
        """

        return np.linalg.norm(self.electric)
    
    def magneticMag(self) -> float:
        """
        Returns the magnitude of the magnetic field in teslas.
        """

        return np.linalg.norm(self.magnetic)
    
    def setFrequency(self,bunch) -> None:
        """
        Given a particle bunch, this method will calculate the angular frequnecy of the bunch and assign
        it to frequency attribute.

        Parameters:
        -----------
        bunch
            The particle bunch whose angular frequency will be calculated.
        """

        self.frequency = abs(bunch._particleCharge)*self.magneticMag()/(bunch._particleMass*bunch.gamma())

    def getAcceleration(self, particleBunch: List['ChargedParticle'], time: Union[int,float], deltaT: Union[int,float]) -> None:
        """
        Calculates the acceleration of every particle in a bunch due to the Lorentz force and assigns the
        the acceleration to particle's acceleration attribute.

        Parameters:
        -----------
        particleBunch
            List of particles which are being accelerated by the electromagnetic field
        
        time
            The current value of time in the simulation, used to calculate the electric field strength,
            in seconds
        
        deltaT
            The time step used in the integrator, in seconds
        """  
        for particle in particleBunch:
            if self.electricLowerBound < particle.position[0] < self.electricUpperBound:
                lorentz = np.array([i*math.cos(self.frequency*time+self.phase) for i in self.electric],dtype=float)
            else:
                lorentz = np.cross(particle.velocity, self.magnetic)
            lorentz *= particle.charge
            acc = 1/(particle.mass*particle.gamma()) * (lorentz - (np.dot(particle.velocity,lorentz)*particle.velocity)/(const.c*const.c)) # relativistic acceleration equation
            particle.acceleration = acc