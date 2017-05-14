"""
Исходные данные
"""
import numpy as np
import sympy
" Объект "
T_f = 300
delta_T = 5
lx = 1.5  # m
ly = 2  # m
n = 5
lambda1 = 3.5  # mcm
lambda2 = 5.5  # mcm
k_a = 0.11

" Объектив "
D_o = 70  # mm
f_o = 110  # mm
r_0 = 30  # mcm

eta_scan = 0.5
nu_k = 30  # Hz

" ПИ "
Nx = 300
Ny = 200
n = 20
ax = 30  # mcm
ay = 30  # mcm
tau_os = 0.7

tau_eye = 0.1  # s
P = 0.95
SNR_min = 5

"""
===================
"""

delta_nu = 0.5*nu_k*Nx*Ny/(np.pi*eta_scan*n)
print(' delta_nu = ' + str(delta_nu))

""" To find"""
L_d = sympy.symbols('L')

beta = lambda f, L: f/L
L_d = float(sympy.solve(sympy.Eq(beta(f_o, L_d), 550), L_d)[0])
print(L_d)
