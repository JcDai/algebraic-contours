from sympy import symbols, Eq, solve, collect, expand
from sympy.printing import ccode
from sympy import *


def dot(a, b):
    return a[0]*b[0] + a[1]*b[1]


ui, uj, uk, ukp, vi, vj, vk, vkp = symbols('ui uj uk ukp vi vj vk vkp')
Uik = Matrix([[ui-uj, uk-uj], [vi-vj, vk-vj]])
Uik_inv = Uik.inv()
u_jk_p = Matrix([[ukp-uj], [vkp-vj]])
U_ijk = Uik_inv * u_jk_p
a1c = U_ijk[0]
a2c = 1-U_ijk[0]-U_ijk[1]
a3c = U_ijk[1]

# ac
a1c = simplify(a1c)
a2c = simplify(a2c)
a3c = simplify(a3c)

v0_pos = Matrix([ui, vi])
v1_pos = Matrix([uj, vj])
v2_pos_macro = Matrix([uk, vk])
v2_pos = (v0_pos + v1_pos + v2_pos_macro) / 3

v0_pos_prime = Matrix([uj, vj])
v1_pos_prime = Matrix([ui, vi])
v2_pos_macro_prime = Matrix([ukp, vkp])
v2_pos_prime = (v0_pos_prime + v1_pos_prime + v2_pos_macro_prime) / 3

u_01 = v1_pos - v0_pos
u_02 = v2_pos - v0_pos
u_12 = v2_pos - v1_pos

u_01_prime = v1_pos_prime - v0_pos_prime
u_02_prime = v2_pos_prime - v0_pos_prime
u_12_prime = v2_pos_prime - v1_pos_prime

m_01 = (u_02 + u_12) / 2
m_01_prime = (u_02_prime + u_12_prime) / 2

u_01_prep = Matrix([-u_01[1], u_01[0]])
u_01_prep_prime = Matrix([-u_01_prime[1], u_01_prime[0]])

K_N_p = Matrix([
    [1, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0],
    [-3, 0, 3, 0, 0, 0, 0],
    [0, -3, 0, 3, 0, 0, 0],
    [-1 / 8, -1 / 8, -7 / 8, -7 / 8, 1 / 4, 1 / 4,
     3 / 2]
])

c_hij_p = Matrix([-1. / 8., -1. / 8., -7. / 8., -
                 7. / 8., 1. / 4., 1. / 4., 3. / 2.])
c_e_m = Matrix([-3. / 2.,
                3. / 2.,
                -1. / 4.,
                1. / 4.,
                0.,])

u_01_normalized = u_01/u_01.norm()
u_01_prime_normalized = u_01_prime/u_01_prime.norm()

u_01_prep_normalized = u_01_prep / u_01_prep.norm()
u_01_prep_prime_normalized = u_01_prep_prime / u_01_prep_prime.norm()

M_N = dot(m_01, u_01_normalized) / u_01.norm() * c_e_m.transpose() * K_N_p
k_N = dot(m_01, u_01_prep_normalized)
M_N_prime = dot(m_01_prime, u_01_prime_normalized) / \
    u_01_prime.norm() * c_e_m.transpose() * K_N_p
k_N_prime = dot(m_01_prime, u_01_prep_prime_normalized)

M_N = simplify(M_N)
k_N = simplify(k_N)
M_N_prime = simplify(M_N_prime)
k_N_prime = simplify(k_N_prime)

CM = (c_hij_p - M_N.transpose())/k_N
CM_prime = (c_hij_p - M_N_prime.transpose())/k_N_prime

ag = Matrix([CM[0] + CM_prime[1],
             CM[1] + CM_prime[0],
             CM[2] + CM_prime[3],
             CM[3] + CM_prime[2],
             CM[4],
             CM[5],
             CM_prime[5],
             CM_prime[4],
             CM[6]])

ag = -ag/CM_prime[6]

ag = simplify(ag)

# am
a1m = ag[0] + ag[2] + ag[4] + ag[6]
a2m = ag[1]
a3m = ag[3]
a4m = ag[5]
a5m = ag[7]
a6m = ag[8]

# thing to check to be 0
sth = a3c * a5m + a4m
sth2 = -a1c*a5m - a1m - a3m - a6m + 1
sth3 = a2c*a5m + a2m
