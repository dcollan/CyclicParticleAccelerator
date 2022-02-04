import log
import numpy as np
from Particle import Particle
import scipy.constants as const
from typing import Union, List

class ChargedParticle(Particle):
    """
    This class represents a charged particle. It inhertis from the Particle class in Particle.py

    Attributes:
    -----------
    charge: float
        The charge of particle the object is representing. Measured in coulombs.
    """
    
    def __init__(self,  Name: str = 'Point', Mass: Union[int,float] = 1.0, Charge: Union[int,float] = 1.0, Position: Union[List[float],List[float]] = np.array([0,0,0], dtype=float), 
                Velocity: Union[List[float],List[float]] = np.array([0,0,0], dtype=float), Acceleration: Union[List[float],List[float]]= np.array([0,0,0], dtype=float)) -> None:
        super().__init__(Name=Name, Mass=Mass, Position=Position, Velocity=Velocity, Acceleration=Acceleration)
        self.charge = float(Charge)
    
    def __repr__(self) -> None:
        return 'ChargedParticle: {0} \n Mass: {1} \n Charge: {2} \n Position: {3} \nVelocity: {4} \n Acceleration: {5}'.format(self.name,self.mass,self.charge,self.position, self.velocity,self.acceleration)