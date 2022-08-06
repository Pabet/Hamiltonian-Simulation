import math
import numpy as np
from mpmath import sec, csc, cot
from scipy.linalg import expm, sinm, cosm
"""
function for one step of split operator method
"""

def splitOperator(psi, Vprop, Tprop):
    psi *= Vprop  # 1/2 step of potential propagator
    psi_p = np.fft.fft(psi)  # shift to momentum basis
    psi_p *= Tprop  # kinetic propagator
    psi = np.fft.ifft(psi_p)  # shift back to spatial basis
    psi *= Vprop  # 1/2 step of potential propagator  return psi
    return psi
