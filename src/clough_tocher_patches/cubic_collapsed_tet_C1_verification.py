from sympy import symbols, factorial, Symbol, simplify,Matrix, det, \
Rational, collect, expand, Poly, Expr, diff, solve, rcollect, binomial
from itertools import product
from math import prod
from functools import reduce
from operator import add

flatten_list = lambda l_of_l: [v for l in l_of_l for v in l]

# This is fixing a problem with sympy sobustitutions: make these work on lists, tuples and dicts recursively
def lsubs(a, subsexp):
    """Recursive substitution. If a is a list, tuple or dict, apply recursively. 
       If a has method subs(), call it. 
       If neither holds, fail. 
    """
    if hasattr(a, 'subs'):
        # a is a SymPy object with a .subs() method
        return a.subs(subsexp)
    elif isinstance(a, list):
        # a is a list, apply recursively 
        return [lsubs(elem, subsexp) for elem in a]
    elif isinstance(a, tuple):
        # a is a tuple, apply recursively 
        return tuple(lsubs(elem, subsexp) for elem in a)
    elif isinstance(a, dict):
        # a is a dict, apply to values
        return {k: lsubs(v, subsexp) for k, v in a.items()}
    else:
        # Raise an error for unsupported types
        raise TypeError(f"Unsupported type {type(a)} in lsubs")

def monomial_multiindices(num_vars, max_degree):
    # lists of integers  from 0..max_degree range with the sum <= max_degree
    return [e for e in product(range(max_degree + 1), repeat=num_vars) if sum(e) <= max_degree]

def bezier_multiindices(N, d):
    """Generate all lists of d non-negative integers that sum to N."""
    if d == 1:
        return [[N]]
    result = []
    for i in range(N + 1):
        for tail in bezier_multiindices(N - i, d - 1):
            result.append([i] + tail)
    return result

def generate_bezier_basis(N, vars):
    """Generate a list of Bernstein basis functions of total degree N in the given list of variables, 
    ordered lexicographically.
    """
    d = len(vars)
    if d == 0: 
        return [1]
    if d == 1:
        return [binomial(N, i) * vars[0]**i * (1 - vars[0])**(N - i) for i in range(N + 1)]
    basis = []
    for multiindex in bezier_multiindices(N, d+1):
        coeff = factorial(N)/prod( Rational(factorial(p)) for p in multiindex)
        basis_fun = coeff * prod(v**e for v, e in zip(vars, multiindex[:-1])) * (1 - sum(vars))**multiindex[-1]
        basis.append(basis_fun)
    return basis

# For an expression compute the maximal total degree of monomials w.r.t. list of variables vars
total_degree = lambda expr,vars: max(sum(monomial) for monomial in Poly(expr, *vars).monoms())

def bezier_coefficients(expr, vars):
    """Convert a to bezier basis given by the list of basis functions
    expr: polynomial in variables given by vars (the coefficients may depend on other variables), 
    vars: the list of variables to consider 
    """
    expr = expand(expr)
    n_vars = len(vars)
    # Determine degree from the Bernstein basis
#    max_degree = max(sum(m) for b in basis for m in b.as_poly(*vars).monoms())
    total_deg = total_degree(expr, vars)
    
    mono_exps = monomial_multiindices(n_vars, total_deg)
    mono_basis = [prod(v**e for v, e in zip(vars, exps)) for exps in mono_exps]
    bezier_basis = generate_bezier_basis(total_deg, vars)
    # Extract monomial coefficients of the expression
    poly = expr.as_poly(*vars)

    mono_coeffs = Matrix([
        poly.coeff_monomial(vars_power)
        for vars_power in mono_exps
    ])

    M = Matrix([
        [expand(b).as_poly(*vars).coeff_monomial(e) for e in mono_exps]
        for b in bezier_basis
    ])
    # Solve the linear system for Bezier
    bezier_coeffs = M.T.LUsolve(mono_coeffs)
    return bezier_coeffs.T.tolist()[0]

def node_positions_2D(alpha, beta,corners_2D):
    """Cubic node parametric positions for a triangle defined by 3 corners, 
    with C-T split with central vertex at (1-alpha-beta, beta, alpha) bary. coords.
    Linear interpolation of vertex positions on subtriangles. 
    """
    r1_3 =  Rational(1,3)
    r2_3 =  Rational(2,3)
    sp0,sp1,sp2 = tuple(map(lambda x: Matrix(x),corners_2D))
    spc =  (1-alpha-beta)*sp0 + beta*sp2 + alpha*sp1

    node_pos_subs = { 
    p0: sp0, p1: sp1, p2: sp2,
        
    p01: r2_3*sp0 + r1_3*sp1, p10: r1_3*sp0 + r2_3*sp1, 
    p12: r2_3*sp1 + r1_3*sp2, p21: r1_3*sp1 + r2_3*sp2, 
    p20: r2_3*sp2 + r1_3*sp0, p02: r1_3*sp2 + r2_3*sp0,
    
    p0c: r1_3*spc + r2_3*sp0, pc0: r2_3*spc + r1_3*sp0,
    p1c: r1_3*spc + r2_3*sp1, pc1: r2_3*spc + r1_3*sp1,
    p2c: r1_3*spc + r2_3*sp2, pc2: r2_3*spc + r1_3*sp2,
    
    pm01: (sp0 + sp1 + spc)*r1_3, 
    pm12: (sp1 + sp2 + spc)*r1_3, 
    pm20: (sp2 + sp0 + spc)*r1_3,
        
    pc: spc 
    }
    return node_pos_subs


def cpN_3D(N):
    "Symbols for Bezier control points for Bernstein basis of deg N, with 3 variables"
    indices = bezier_multiindices(N, 4)
    return [Symbol(f'q{i}{j}{k}{l}') for i, j, k, l in indices]

def lin_3D(corners):
    """Given tet vertex positions, construct cubic Bezier node positions,
        uniformly spaced, linearly interpolated between vertices
    """
    # symbolic control points for tet
    cps = cpN_3D(3)    
    bary_indices = bezier_multiindices(3, 4)
    # reduce(add, ) is used as sum(...) is less nice for matrices, requires explicit initial 
    cp_exprs = {
        cps[i]: reduce( add, [(idx[j]*Rational(1, 3)) * corners[j] for j in range(4)])
        for i, idx in enumerate(bary_indices)
    }
    return cp_exprs


def interior_C1_coeffs(alpha,beta,node_pos_subs):
    
    # coefficients for the C1 Bezier continuity conditions for 4 points: 
    # pts are parametric node positions 
    # The condition is  \sum_{i=1..4} a_i f_i = 0, where f_i are function values at nodes 
    #    0
    #  1   2
    #    3
    # Geometrically,  the coefficient of a_i is the area of the triangle opposite formed 
    # by points not including point i.  We flip the signs for a1,a2 so that the formula is uniform

    constr_sym = [
        [p0, p01, p02, p0c],    [p1, p12, p10, p1c],   [p2, p20, p21, p2c],    # p_ic
        [p0c, pm01, pm20, pc0], [p1c, pm12, pm01, pc1],[p2c, pm20, pm12, pc2], # p_ci
        [pc0, pc1, pc2, pc]
    ]

    # helper function getting constraints for one set of 4 nodes. 
    # assumes node positions pts are given as 2 x 1 matrices, converts these to lists and appends 1 
    # to get a 3 x 3 matrix for which det is the area
    def get_constr_coeffs(pts): 
        return [ simplify(det(
            Matrix([pts[i].T.tolist()[0]+[1], 
                    pts[j].T.tolist()[0]+[1], 
                    pts[k].T.tolist()[0]+[1]]))) 
        for i,j,k in [ [1,3,2], [0,2,3],[0,3,1],[0,1,2]]]

    # These constraints actually end up all permutations of the same because relevant triangles are all similar 
    # (gamma, beta, alpha), (beta, alpha, gamma), (alpha, gamma, beta)
    # The general code computing the constraints is mostly for verifciation, or in the case we need 
    # something more general 
    # TODO: ordering of coefficients is awkward check why beta and alpha are switched

    # plug in parametric values
    pts_constr =  lsubs( constr_sym, node_pos_subs)
    constr_coeffs = [get_constr_coeffs(p) for p  in pts_constr]

    # convert to substitutions
    constr_lhs = [c[3] for  c in constr_sym]
    constr_rhs = [c[:-1] for c in constr_sym]

    # for visual verification, replace  1-alpha-beta with gamma
    gamma = symbols('gamma')
    gamma_sub = {alpha + beta - 1: -gamma, 1-alpha-beta:gamma, -alpha-beta+1: gamma}

    all_subs = { constr_lhs[i]: 
        sum( 
            [simplify(-constr_rhs[i][j]*constr_coeffs[i][j]/constr_coeffs[i][3]).subs(gamma_sub) for j in range(3)]
        ) for i in range(len(constr_lhs))}
    # pc depends on pci, which depend on pic; substituting into itself twice expressese pc and pci 
    # in terms of indep. variables 
    all_exp_subs =  lsubs( lsubs(all_subs, all_subs),all_subs)
    return  all_exp_subs

# expressions for 3 x 10  control points of Clough-Tocher subtriangles  in terms of 12 independent dofs (pi, pij, pmij)
def CT_C1_patch_cp(interior_C1_subs):
    Bezier_cp_exp =  lsubs(Bezier_cps, interior_C1_subs)
    Bezier_cp_exp = list(map( lambda x: list(map( expand, x)), Bezier_cp_exp))
    return Bezier_cp_exp

def collapsed_C1_patch_cps_2D(Bezier_cp_exp, c01,c12,c20, node_pos_subs):

    # collapsed control point positions for 2D face (0,0), (0,1) (1,0)
    #  points pij, pik are collapsed to pi 
    # pmij are collapsed to a point on the edge (i,j) split in ratio cij

    sp0,sp1,sp2,spc  = lsubs([p0,p1,p2,pc],node_pos_subs)
    collapsed_cp = {
        p0: sp0, p01:sp0, p10: sp1,
        p1: sp1, p12: sp1, p21: sp2,
        p2: sp2, p20: sp2, p02: sp0,
        pm01: sp0*(1-c01)  + sp1*c01,     
        pm12: sp1*(1-c12)  + sp2*c12,
        pm20: sp0*(1-c20)  + sp2*c20,
        pc: spc
    }
    # 2D cubic Bezier patches corresponding to collapsed C-T control points in 2D
    # and the determinant of the map R2 -> R2 defined by these control points in variables (s,t)
    return collapsed_cp

apply_bezier_coeff = lambda l, vars: flatten_list(list( map( lambda x: bezier_coefficients(x,vars), l)))

# var_lists contains lists of variables; each group satisfies: 
# [s,t]: s + t <= 1, 1 >= s,t >= 0
# [alpha, beta]:  alpha + beta <=1, 1 >= alpha, beta >=0 
# c01,c12,c20:    1 >= cij >= 0, each is a one-dimensional bases
# To show that the polynomial in these variables is positive for these variable ranges
# We convert it to the Bezier basis for each set, i.e., to the tensor-product basis 
# obtained from all combinations of Bezier functions:  10 (for cubic s,t) x 28 (for order 6 in alpha, beta) x 2 x 2 x 2
#
# The list of input polynomials are converted iteratively:
# Coefficients in the basis in variables s,t depend on alpha, beta, c01, c12, c20
# Then each coefficient is converted to a list of coeffs in the Bezier basis w.r.t. alpha,beta 
# depending on c01,c12,c20, etc 
# In the end, we have a list of constant coefficients 
# The proof of positivity of the initial poloynomial list is recursive: 
# Let [p^k_0 ... p^k_{n_k-1}] be a list of polinomials on step k, in variables var_k, 
# with coefficients depending on variables union( var_{k+1}, .. var_K) 
# Define [p^{k+1}_0 .. p^{k+1]_{n_{k+1}-1}] be a new list of polynomials, obtained by 
# computing Bezier coefficients of all p^k_i w.r.t. var_k, and concatenating the resulting list. 
# p^k_i are nonegative for all variable values iff p^{k+1}_i are nonnegative. 
# repeating the process K times, we get to a list of polynomials that does not depend on 
# any variables, i.e., are constants, which we can check for non-negativity. 

# compute bezier coefficients of each polynomial on a list w.r.t. vars, and then flatten the resulting 
# list of lists, for iterative application 

# determinants as polynomials in s, t, alpha, beta, gamma, c01, c12, c20

def is_2D_collapsed_patch_bijective(Bezier_cp_exp, collapsed_cp, alpha, beta,c01,c12,c20):
    B3 = generate_bezier_basis(3, [s, t])
    # Polynomial patches on subtriangles, expanded in terms of C-T dofs
    patch_exp = [
        sum(B3[i] * Bezier_cp_exp[j][i] for i in range(len(Bezier_cp_exp[j])))
        for j in range(3)
    ]
    # substitute collapsed dofs for the standard macro-triangle with center point (alpha, beta)
    patch_coll = [
        patch_exp[i].subs(collapsed_cp)
        for i in range(3)
    ]
    
    detJ = [
        collect( (expand(
            diff(patch[0], s) * diff(patch[1], t) -
            diff(patch[0], t) * diff(patch[1], s))), [s,t]
        )
        for patch in patch_coll
    ]
        
    var_lists = [[s,t],[alpha,beta],[c01],[c12],[c20]]
    gamma = symbols('gamma')
    bcoeffs = lsubs(detJ, {gamma:1-alpha-beta})
    for v in var_lists: 
        bcoeffs = apply_bezier_coeff(bcoeffs,v)
    return all( [c >= 0 for c in bcoeffs])

def collapsed_C1_patch_cps_3D(Bezier_cp_exp, collapsed_cp_2D): 
    """ Bezier control points of 3 3D patch obtained by a C-T split of one face of a tet and splitting the tet into 3.
        The indep control points on the face are collapsed as in collapsed_C1_patch_cps_2D, 
        All other cps are set to linearly interpolated node coords.
    """
    collapsed_cp_exp = [
        [list(lsubs(cp, collapsed_cp_2D)) for cp in bcp]
        for bcp in Bezier_cp_exp
    ]

    # Vertices of tetrahedra obtained by splitting the standard tet with lexicographic vertex enumeration 
    # into three subtets, by inserting vertex  vP = [0, pc]  on the face A,B,C, and connecting it to A,B,C,D
    # vertex enumeration:  subfaces of (A,B,C) are at the end, with the central point vP first
    # The order is 

    sp0,sp1,sp2,spc = lsubs([p0,p1,p2,pc],collapsed_cp_2D)
    # the first 3 vertices of the tet are assumed to be consistent with
    # the collapsed_cp triangle vertices p0,p1,p2
    # sit in the plane x = 0
    vA = Matrix([[0, sp0[0],sp0[1]]])
    vB = Matrix([[0, sp1[0],sp1[1]]])
    vC = Matrix([[0, sp2[0],sp2[1]]])
    # the fourth vertex is at distance 1                
    vD = Matrix([[1, 0, 0]])
    # the face center vertex also comes from triangle
    vP = Matrix([[0,spc[0],spc[1]]])

    tet_corners = [
        [vD, vP, vB, vA],
        [vD, vP, vC, vB],
        [vD, vP, vA, vC]
    ]
    
    cp_syms = cpN_3D(3)  # 15 control point symbols for a tet
    
    # this generates substitution expressions replacing symbolic c.p.s
    # for the 3 subpatches face (0,0,0), (0,0,1), (0,1,0) of the tet 
    # with the collapsed c.p.s  from the 2d trinagle converted to 3d by 
    # appending zero. !!! This assumes tet ordering with these points 
    # enumerated first in the same lexicographic order for each subpatch, 
    # as simply 10 first c.p.s for the tet are replaced in order
    
    collapsed_cp_exp_subs_3D = [
        {cp_syms[i]: Matrix( [[0] + collapsed_cp_exp[k][i]]) for i in range(10)}
        for k in range(3)]

    lin_3D_cps_sub = list(map( lin_3D, tet_corners))
    
    collapsed_cp_3D = [
        lsubs(
            lsubs(cpN_3D(3), collapsed_cp_exp_subs_3D[k]),
            lin_3D_cps_sub[k]
        )
        for k in range(3)
    ]
#    gamma = symbols('gamma')
#    collapsed_cp_3D = lsubs(collapsed_cp_3D, {gamma: 1 - alpha - beta})
    return collapsed_cp_3D

def get_collapsed_C1_patch_cps_3D(alpha, beta, c01,c12,c20, corners_2D):

    node_pos_subs = node_positions_2D(alpha,beta,corners_2D)
    # C1 constraints converted to substituions of 7 dependent points pic, pci, pc expressed 
    # in terms of 12 (local) indep.  pi, pij, pmij
    interior_C1_subs = interior_C1_coeffs(alpha,beta,node_pos_subs)

    gamma = symbols('gamma')
    # 3 x 10  polynomial subpatch control polints expressed in terms of indep points
    Bezier_cp_exp =  lsubs( CT_C1_patch_cp(interior_C1_subs), {gamma: 1-alpha-beta})

    # Collapsed indep 2D control points, pij -> pi,  pmij -> (1-cij)*pi + cij*pj, returned as substition
    collapsed_cp = collapsed_C1_patch_cps_2D(Bezier_cp_exp, c01,c12,c20, node_pos_subs)

#  Verify if the 2D patch obtained by plugging in collapsed_cp values into the C1 patch expressions 
    # is bijectve by viewing it as a polynomial in s,t, (alpha, beta), cij, 
    # and converting to the Bezier form, and verifying that all coeffs are positive. 
    # This check is insufficient was an intermediate stage to writing the code for the 3D check
    # print(  is_2D_collapsed_patch_bijective(Bezier_cp_exp, collapsed_cp, alpha, beta, c01, c12, c20))
    
    #  Construct 3D collapsed points by combining linearly spaced ones away from the face (0,0,0),(0,0,1),(0,1,0) 
    # with collapsed ones substituted on that face. 

    collapsed_cp_3D = collapsed_C1_patch_cps_3D(Bezier_cp_exp, collapsed_cp)
    return collapsed_cp_3D

def is_3D_collapsed_patch_bijective(collapsed_cp_3D,var_lists):
    # 3 3D patches for each subtet in variables var_lists[0] (should be length 3, defining map R3->R3 
    # with coefficients as functions depending on var_lists[1] ... 
    BN = generate_bezier_basis(3,var_lists[0])
    patch_coll_3D = [
        simplify(reduce(add, [Matrix(collapsed_cp_3D[k][i]) * BN[i] for i in range(len(BN))]))
        for k in range(3)]
    # Compute 3×3 Jacobian matrix of each  patch_coll_3D[k] w.r.t. [s, t, r] and its determinants
    vars = [s, t, r]
    J_3D = [
        Matrix([[diff(patch_coll_3D[k][i], vars[j]) for j in range(3)] for i in range(3)])
        for k in range(3)
    ]
    detJ_3D = [det(J) for J in J_3D]
    
    # Similar to the above test for the 2D triangle, but for the 3D tet 
  #  var_lists = [[s,t,r],[alpha,beta],[c01],[c12],[c20]]
   # gamma = symbols('gamma')
   # bcoeffs = lsubs(detJ_3D, {gamma:1-alpha-beta})
    bcoeffs = detJ_3D
    for v in var_lists: 
        bcoeffs = apply_bezier_coeff(bcoeffs,v)
    return all( [c >= 0 for c in bcoeffs])

if __name__ == "__main__":
    #Symbolic global variables
    # variables used for 2D and 3D Bezier functions
    s,t,r = symbols('s t r')

    # Symbols for independent control points of a C-T patch
    p0, p1, p2, p01, p10, p12, p21, p20, p02, pm01, pm12, pm20 = symbols(
        'p0 p1 p2 p01 p10 p12 p21 p20 p02 pm01 pm12 pm20'
    )

    # Symbols for dependent control points of a C-T patch
    p0c, p1c, p2c, pc0, pc1, pc2, pc = symbols('p0c p1c p2c pc0 pc1 pc2 pc')

    # symbolic control points of the cubic Bezier subtriangles 
    # enumerated in the  lexicographic order of powers of Bezier basis functions
    # i.e.  [0 0 3], [0 1 2],[0 2 1], [0 3 0], [1 0 2], [1 1 1], [1 2 0], [2 0 1],[2 1 0], [3,0,0] 
    # assuming vertex order (p_i p_{i+1}, p_c) for each

    Bezier_cps = [
        [p0, p01, p10, p1, p0c, pm01, p1c, pc0, pc1, pc],
        [p1, p12, p21, p2, p1c, pm12, p2c, pc1, pc2, pc],
        [p2, p20, p02, p0, p2c, pm20, p0c, pc2, pc0, pc]
    ]

    # symbols for collapsed configuration parameters, gamma = 1-alpha-beta 
    alpha,beta,gamma = symbols('alpha beta gamma')
    c01, c12, c20 = symbols('c01 c12 c20')

    # List of variables used to convert the determinant of the Jacobian to the Bezier basis
    # The first list is the variables for the map R3 -> R3, the rest are the independent variables
    var_lists = [[s,t,r],[alpha,beta],[c01],[c12],[c20]]

    # Node positions on subtriangles, uniformly linearly interpolated from subtriangle corners 
    # With central vertex at (alpha, beta)
    corners_2D =  [ [0, 0],[0, 1], [1, 0]]

    collapsed_cp_3D = get_collapsed_C1_patch_cps_3D(alpha, beta, c01,c12,c20, corners_2D)
    # Verify  that the 3D maps given by the 3 subpatches are bijective
    print("Verifying that the 3D patches on Clough-Tocher split of a tetrahedron with collapsed control points is bijective...")  
    print("Result:", is_3D_collapsed_patch_bijective(collapsed_cp_3D,var_lists)) 