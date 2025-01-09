from Clough_Toucher_derivation import *
from midpoint_constraint_validation import *


#   Data from algebraic countours code, 2-triangle example

# global parametriztion uv
pp = np.array([[0, 0],
[0.87310264093377665, 0.87310264093377654],
[0.43655132046688805, -0.81779020462853524],
[-0.81779020462853447, 0.43655132046688827]
])

# edge chart uv
pp_2 = np.array([[-0.5, 0],
[0.5, 0],
[-0.7183242074230329, -0.7183242074230325],
[-0.7183242074230323,  0.7183242074230326]
])

TT = [pp[0],pp[1],pp[3]]
TT_p = [pp[1],pp[0],pp[2]]
#TT = [pp_2[0],pp_2[1],pp_2[3]]
#TT_p = [pp_2[1],pp_2[0],pp_2[2]]

# rdm for x
rx_dM = [0.07265181092348216 , 0.9869807651025245 , 0.8949478612243169, -0.9467975711355811, -0.6520840786817188]
rx_dM_prime =[  0.9869807651025245 ,0.07265181092348216 ,-0.9467975711355811,  0.8949478612243169, -0.6520840786817305]

# rdm for y
ry_dM =  [ 0.1663851475313266, -0.02981628801300628,   -0.240587473133782,   0.1218427161430667,  -0.3482987078639407]
ry_dM_prime = [-0.02981628801300628 ,  0.1663851475313266 ,  0.1218427161430667  , -0.240587473133782 ,  0.6517012921360451]

# rdm for z
rz_dM =    [0.1663851475313306, -0.0298162880130032, -0.2405874731337727,   0.121842716143081 ,  0.651701292136047]
rz_dM_prime =[ -0.0298162880130032, 0.1663851475313306,   0.121842716143081, -0.2405874731337727, -0.3482987078639431]



Tbdx = [-0.05963257602600576, 0.07265181092348216, 0.9869807651025245, 0.07847906842083788, -0.1953887824867313, 0.8949478612243169, -0.9467975711355811, -1.153506400443802, 0.8907077799964176,0.9244621997134772, -0.6520840786817188, -0.3606406155152121]
Tbdy = [-0.136568859518313, 0.1663851475313266, -0.02981628801300628, 0.1797305698418802, -0.4474739306121528, -0.240587473133782, 0.1218427161430667, -0.3515560694247197, -0.250297988696742, -0.1729945597378522, -0.3482987078639407, 0.3191569687340334 ]
Tbdz =  [0.8634311404816811, 0.1663851475313306, -0.0298162880130032, -0.8202694301581125, 0.5525260693878359, -0.2405874731337727, 0.121842716143081, 0.6484439305752829, -1.25029798869673, -0.6729945597378411, 0.651701292136047, -0.1808430312659616 ]
Tbdx_p = [0.9869807651025245, 0.07265181092348216, -0.05963257602601314, -0.9467975711355811, 0.8949478612243169, -0.195388782486742, 0.0784790684208431, 0.89070777999642, -1.153506400443815, -0.6520840786817305, 0.9244621997134806, -0.3606406155152064 ]
Tbdy_p = [-0.02981628801300628, 0.1663851475313266, 0.8634311404816849, 0.1218427161430667, -0.240587473133782, 0.5525260693878387, -0.8202694301581254, -1.250297988696766, 0.6484439305752623, 0.6517012921360451, -0.6729945597378608, -0.1808430312659596 ]
Tbdz_p = [-0.0298162880130032, 0.1663851475313306, -0.1365688595183127, 0.121842716143081, -0.2405874731337727, -0.4474739306121539, 0.1797305698418869, -0.2502979886967234, -0.3515560694247059, -0.3482987078639431, -0.1729945597378411, 0.3191569687340322 ]


r_vars = [p0,p1,G01,G10,N01]
# permuted versions because of indexing, we only need one for the first triangle as in the Tbd* arrays the shared edge indices are 1,2 not 0,1
all_vars_0 = [p0,p1,p2,G01,G10,G12,G21,G20, G02, N01,N12,N20]
all_vars_1 = [p1,p2,p0,G12,G21,G20,G02,G01, G10, N12,N20,N01]
all_vars_2 = [p2,p0,p1,G20,G02,G01,G10,G12, G21, N20,N01,N12]

# Modify the N01 = hij in the r vector assuming it was computed with Powell-Sabin construction for the derivative along the edge at the midpoint
#   hij = (m dot uij) df/du_ij/|u_ij|^2   + (m dot eij_perp) df/dn
#   df/du_ij = a*(pj-pi) + b*(dji-dij)   both for Powell-Sabin and Clough-Tocher, but  (a,b) = (2, 1/2) for P-S, and (3/2, 1/4) for C-T.
def change_N_from_quadr_to_cubic_r(r,T): 
    len_T = list( map( np.linalg.norm,[ T[1]-T[0], T[2]-T[1],T[0]-T[2]]))
    mv = T[2] - (T[0]+T[1])/2
    cub_dfde = ((3/2)*(r[1] - r[0]) + 0.25*(r[3]-r[2]))
    quad_dfde =  ((2)*(r[1] - r[0]) + 0.5*(r[3]-r[2]))    
    r[4] += (cub_dfde-quad_dfde)/(len_T[0]**2)*np.dot(mv, T[1]-T[0])

    return r
    
 # Same for all hij's in the full length 12 dof vector for each triangle. Only the one on shared edge is used!   
def change_N_from_quadr_to_cubic_full(pe,T): 
    mv = [T[2] - (T[0]+T[1])/2, T[0] - (T[1]+T[2])/2, T[1] - (T[2]+T[0])/2]
    evec = [ T[1]-T[0], T[2]-T[1],T[0]-T[2]]
    len_T = list( map( np.linalg.norm,evec))
    ind_G = [ [3,4],[5,6],[7,8]]
    ind_N = [9,10,11]
    ind_p = [0,1,2]
    nxt = lambda n: (n+1)%3

    for i in range(3):    
        cub_dfde = (1.5*(pe[ind_p[nxt(i)]] - pe[ind_p[i]]) + 0.25*(pe[ind_G[i][1]]-pe[ind_G[i][0]]))
        quad_dfde =      (2.0*(pe[ind_p[nxt(i)]] - pe[ind_p[i]]) + 0.5*( pe[ind_G[i][1]]-pe[ind_G[i][0]]))
        pe[ind_N[i]] += (cub_dfde-quad_dfde)/(len_T[i]**2)*np.dot(mv[i], evec[i])
    return pe

# Remove small coefficients.  When we construct subtriangle polynomial P from r, we do not have all dofs, and while the missing ones 
# do not affect theoretically P restricted to the shared edge, numerically there may be nonzero coefficients depending linearly on the missing dofs (d12, d21 etc)
# to make the output easier to parse, we trim all tiny terms in the expressions for the coefficients 

def sanitize_cf(c):
    cs,_ = c.as_independent(*c.free_symbols)
    if abs(cs) < 1e-12: 
        cs = 0
    for fs in c.free_symbols:
        if abs( c.coeff(fs)) >  1e-12:
            cs += fs*c.coeff(fs)
    return cs


# Restrict the polymial to an edge
def subs_edge(pp): 
    # equation of the shared edge in up,vp coords
    edge_line = {up:TT[0][0]*(1-t) + TT[1][0]*t, vp:TT[0][1]*(1-t) + TT[1][1]*t} 
    return sp.collect(sp.expand(pp.subs(edge_line)),t)
    
def sanitize_poly(pp): 
    pp_dict = pp.as_coefficients_dict(t)
    pp_s = sum( [ m*sanitize_cf( pp_dict[m]) for m in pp_dict])
    return pp_s

def test_edge_C1(poly0, poly0_p,TT):  
    # Change of coordinates going from barycentric to the parameteric (called up, vp here)          
    transf_sol =  solve( set(TT[0]*u + TT[1]*v + TT[2]*(1-u-v) - np.array([up,vp])), [u,v])
    transf_solp = solve( set(TT_p[0]*u + TT_p[1]*v + TT_p[2]*(1-u-v) - np.array([up,vp])),[u,v])

    # polynomials in up, vp
    poly0par =   sp.collect( sp.expand( poly0.subs(transf_sol), [up,vp]), [up,vp])
    poly0par_p = sp.collect( sp.expand( poly0_p.subs(transf_solp), [up,vp]), [up,vp])

    
    # polynomials and their gradients wrt up, vp restricted to the shared edge for T and T'
    poly0par_e =   sanitize_poly(subs_edge(poly0par))
    poly0par_p_e = sanitize_poly(subs_edge(poly0par_p))

    poly0par_dup = poly0par.diff(up)
    poly0par_dvp = poly0par.diff(vp)
    poly0par_dup_e =   sanitize_poly(subs_edge(poly0par_dup))
    poly0par_dvp_e =   sanitize_poly(subs_edge(poly0par_dvp))
    poly0par_p_dup = poly0par_p.diff(up)
    poly0par_p_dvp = poly0par_p.diff(vp)
    poly0par_p_dup_e =   sanitize_poly(subs_edge(poly0par_p_dup))
    poly0par_p_dvp_e =   sanitize_poly(subs_edge(poly0par_p_dvp))

    # check they match
    print('delta f on edge', sanitize_poly(poly0par_e-poly0par_p_e))
    print('delta df/du on edge', sanitize_poly(collect(poly0par_dvp_e.subs({t:1/2})-poly0par_p_dvp_e.subs({t:1/2}),t)))
    print('delta df/dv on edge', sanitize_poly(collect(poly0par_dup_e.subs({t:1/2})-poly0par_p_dup_e.subs({t:1/2}),t)))

def try_all():
    # r vectors 
    i = 0
    for rd in r_data: 
        print('testing r, ', coord[i])
        i += 1
        poly0 = polys_C1_f[0].subs({w:1-u-v}).subs( {r_vars[i]:rd[0][i] for i in range(5)})
        poly0_p = polys_C1_f[0].subs({w:1-u-v}).subs( {r_vars[i]:rd[1][i] for i in range(5)})
        test_edge_C1(poly0, poly0_p,TT)
        # verify that g_M dot r_{d,M} + g_M' dot r_{d,M}' = 0 is satisfied
        print('gm constraint', np.dot( compute_gm(TT), rd[0]) +  np.dot( compute_gm(TT_p), rd[1]) )

    # full dof vectors
    i = 0
    for fd in full_dof_data:
        print('testing full data, ', coord[i])
        i += 1
        poly0 = polys_C1_f[0].subs({w:1-u-v}).subs( {all_vars_2[i]:fd[0][i] for i in range(12)})
        poly0_p = polys_C1_f[0].subs({w:1-u-v}).subs( {all_vars_0[i]:fd[1][i] for i in range(12)})
        test_edge_C1(poly0, poly0_p,TT)
        # verify that g_M dot r_{d,M} + g_M' dot r_{d,M}' = 0 is satisfied
        print('gm constraint', np.dot( compute_gm(TT), np.array(fd[0])[[1,2,5,6,10]]) +  np.dot( compute_gm(TT_p), np.array(fd[1])[[0,1,3,4,9]]) )


# run tests
up, vp = symbols('up vp') 
# fix  hij aka Nij !!!  correct formula for cubic interpolant instead of quadratic
rx_dM = change_N_from_quadr_to_cubic_r(rx_dM,TT)
ry_dM = change_N_from_quadr_to_cubic_r(ry_dM,TT)
rz_dM = change_N_from_quadr_to_cubic_r(rz_dM,TT)
rx_dM_prime = change_N_from_quadr_to_cubic_r(rx_dM_prime,TT_p)
ry_dM_prime = change_N_from_quadr_to_cubic_r(ry_dM_prime,TT_p)
rz_dM_prime = change_N_from_quadr_to_cubic_r(rz_dM_prime,TT_p)

TT_perm = [TT[2],TT[0],TT[1]]  # need this Tbdx etc, as the indexing is different from local edge

Tbdx = change_N_from_quadr_to_cubic_full(Tbdx,TT_perm)
Tbdy = change_N_from_quadr_to_cubic_full(Tbdy,TT_perm)
Tbdz = change_N_from_quadr_to_cubic_full(Tbdz,TT_perm)
Tbdx_p = change_N_from_quadr_to_cubic_full(Tbdx_p,TT_p)
Tbdy_p = change_N_from_quadr_to_cubic_full(Tbdy_p,TT_p)
Tbdz_p = change_N_from_quadr_to_cubic_full(Tbdz_p,TT_p)

r_data = [ [rx_dM, rx_dM_prime], [ry_dM, ry_dM_prime],[rz_dM, rz_dM_prime]]
full_dof_data = [ [Tbdx, Tbdx_p],[Tbdy, Tbdy_p],[Tbdz, Tbdz_p]] 
coord = ['x','y','z']

try_all() 
