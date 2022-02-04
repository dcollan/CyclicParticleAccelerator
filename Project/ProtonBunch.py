import log
import scipy.constants as const
import math
from typing import Union, List
from statistics import mean
from Bunch import Bunch

class ProtonBunch(Bunch):
    """
    Class to generate a bunch of protons as Charged Particle objects, the list can be accessed 
    through the bunch attribute (derived from Bunch abstract base class). 

    Attributes:
    -----------
    AverageKinetic: float/int
        The average kinetic energy of a particle in the bunch, measured in eV.
    
    particleNumber: int, optional 
        Number of particles in the bunch (defaults to 3).
    
    positionSigma: float/int, optional
        The S.D of the particles' normal position distribution  about the origin in metres 
        (defaults to 0.01 m).
    """

    def __init__(self, AverageKinetic: Union[int,float], particleNumber: int = 3, positionSigma: Union[int,float] = 0.01) -> None:
        self._particleName = 'proton' # protected attributes as they should not change
        self._particleMass = const.m_p
        self._particleCharge = const.e
        super().__init__(particleName=self._particleName, particleMass=self._particleMass, particleCharge=self._particleCharge, 
                        AverageKinetic=AverageKinetic, particleNumber=particleNumber, positionSigma=positionSigma)