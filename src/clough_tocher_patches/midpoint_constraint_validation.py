import numpy as np 
import sympy as sp 
from sympy import symbols, Poly
from Clough_Toucher_derivation import *
polys_C1_f = derive_C1_polynomials(vtx_unknowns,side_deriv_unknowns,mid_deriv_unknowns)

def perp(v): 
    return [-v[1],v[0]]


def norm2(v):
    return sp.sqrt(v[0]**2 + v[1]**2)


def compute_gm(T):
    global ch,c_e
    u01 = T[1]-T[0] 
    u10 = T[0]-T[1]
    mpt = (T[0]+T[1])/2 
    m   =  T[2]-mpt
    u01_n = u01/norm2(u01); u10_n = u10/norm2(u10); 
    gm =  (ch - np.dot(m, u01_n)*c_e/norm2(u01))/np.dot(m, perp(u01_n));
    return gm

def compute_r(T,ff,gradf):
    u01 = T[1]-T[0] 
    u10 = T[0]-T[1]
    mpt = (T[0]+T[1])/2 
    m   =  T[2]-mpt
    return [ ff(*T[0]), ff(*T[1]) , np.dot( gradf(*T[0]), u01), np.dot( gradf(*T[1]), u10), np.dot( gradf(*mpt), m)] 

npa = np.array
c_e = generate_ce(polys_C1_f)
print(c_e)
c_e = npa(c_e) 
ch = npa([0,0,0,0,1])

p0u,p0v,p1u,p1v,p2u,p2v, p2up,p2vp = symbols( 'p0u p0v p1u p1v p2u p2v p2up p2vp')
T =   list( map(npa, [ [p0u,p0v],[p1u,p1v],[p2u,p2v]]))
Tp =  list( map(npa, [ [p1u,p1v],[p0u,p0v],[p2up,p2vp]]))

a300, a030, a210, a120, a201, a021, a003, a111, a102, a012 = \
        symbols('a300 a030 a210 a120 a201 a021 a003 a111 a102 a012')
poly3 = a300 * u**3 + a030 * v**3 + a210 * u**2 * v + a120 * u * v**2 \
                + a201 * u**2  + a021 * v**2  + a003 + a111 * u * v  \
                + a102 * u  + a012 * v
f = Poly(poly3,[u,v]) 

gradf = lambda u_,v_: np.array( [f.diff(u)(u_, v_),f.diff(v)(u_, v_)]) 

print( 'checking midpoint constraint for general cubic (should be zero): ', sp.simplify( sp.factor(np.dot( compute_r(T, f,gradf ), compute_gm(T)) + np.dot( compute_r(Tp, f,gradf ), compute_gm(Tp)))))
