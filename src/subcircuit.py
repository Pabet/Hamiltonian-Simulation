import math
import numpy as np
from mpmath import sec, csc, cot
from scipy.linalg import expm, sinm, cosm

# Pauli Z gate
Z = np.array([[1., 0.], [0., -1.]])

# Pauli X gate
X = np.array([[0., 1.], [1., 0.]])

# Hadamard gate
H = (1. / math.sqrt(2)) * np.array([[1., 1.], [1., -1.]])


# Rotation operator: rotate single qubit around z-axis by angle
def Rz(angle):
    return np.array([[math.cos(angle / 2), -math.sin(angle / 2)], [math.sin(angle / 2), math.cos(angle / 2)]])


# Defined by the paper Hamiltonian simulation algorithms for near-term quantum hardware - supplementary
# Figure 3 (b)
def phi(t):
    return ((1 / 4) * (3 + 2 * math.sqrt(2)) * t) ** (1 / 3)


# Pulse times according to Lemma 8 (Depth 5 Decomposition)
# t_1^a = t_1[0]
# t_1^b = t_1[1]
def t1(t, sign, c):
    first_component = (1 / 2) * math.atan(
        sign * math.sqrt(2) * csc(2 * phi(t)) * math.sqrt(math.cos(2 * t) - math.cos(4 * phi(t)))) + math.pi * c
    second_component = math.atan(-2 * math.tan(t) * cot(2 * phi(t))) + 2 * math.pi * c
    return np.array([first_component, second_component])


# t_2^a = t_2[0]
# t_2^b = t_2[1]
def t2(t, sign, c):
    first_component = math.atan(
        sign * (csc(2 * phi(t)) * math.sqrt(math.cos(2 * t) - math.cos(4 * phi(t)))) / (math.sqrt(2))) + 2 * math.pi * c
    second_component = math.atan(math.sin(t) * csc(2 * phi(t))) + 2 * math.pi * c
    return np.array([first_component, second_component])


# Pauli Y-gate decomposition
Y = Rz(math.pi / 2) * H * Z * H * Rz(-math.pi / 2)


def yzz_t1_circuit(t, sign, c, y0):
    circuit = np.kron(np.kron(np.kron(np.identity(2), Y), Z), Z)
    return np.dot(expm(1j * t1(t, sign, c)[1] * circuit), y0)


def yzz_t2_circuit(t, sign, c, y0):
    circuit = np.kron(np.kron(np.kron(np.identity(2), Y), Z), Z)
    return np.dot(expm(1j * t2(t, sign, c)[1] * circuit), y0)


def zx_circuit(t, y0):
    circuit = np.kron(np.kron(np.kron(Z, X), np.identity(2)), np.identity(2))
    return np.dot(expm(1j * phi(t) * circuit), y0)


def depth_5_decomposition(t, sign, c, y0):
    y0_1 = yzz_t1_circuit(t, sign, c, y0)
    y0_2 = zx_circuit(t, y0_1)
    y0_3 = yzz_t2_circuit(t, sign, c, y0_2)
    y0_4 = zx_circuit(t, y0_3)
    return yzz_t1_circuit(t, sign, c, y0_4)
