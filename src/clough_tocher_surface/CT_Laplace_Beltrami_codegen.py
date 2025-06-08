
import sympy as sp
from sympy import symbols, factorial,Function, Matrix,Symbol, Poly, diff, simplify,collect, \
sqrt, cosh, sinh, cos, sin, pi, expand, solve, lambdify
from itertools import product
from sympy.printing.c import C99CodePrinter
from typing import List, Union
from pathlib import Path
import sys

def dot3(a, b):
    return sum(ai * bi for ai, bi in zip(a, b))

# Global definitions 
s, t = symbols('s t')
r = 1 - s - t

# Generate node indices for the cubic spline in barycentric form (i+j+k=3)
node_ind = [ [i, j, 3 - i - j] for i in range(4) for j in range(4 - i) ]
# Define the Bernstein basis functions
def bernstein_term(x):
    i, j, k = x
    return factorial(3) / (factorial(i)*factorial(j)*factorial(k)) * s**i * t**j * r**k
B = [bernstein_term(x) for x in node_ind]

# First derivatives of the Bernstein basis (2 x 10 matrix)
B1 = Matrix([
    [diff(b, s) for b in B],
    [diff(b, t) for b in B]
])

# Second derivatives of the Bernstein basis (3 x 10 matrix)
B2 = Matrix([
    [diff(b, s, s) for b in B],
    [diff(b, t, t) for b in B],
    [diff(b, s, t) for b in B]
])

# Symbols for the coefficients of the function to be solved for
cp = [symbols(f'q{i}{j}{k}') for i, j, k in node_ind]

# Symbols for the coefficients of the 3D surface used for metric
cp3d = [
    [symbols(f'mx{i}{j}{k}'), symbols(f'my{i}{j}{k}'), symbols(f'mz{i}{j}{k}')]
    for i, j, k in node_ind
]

# Linear positions of control points in 3D (projected to a triangle)
linear_cp = [[i/3, j/3, 0] for i, j, k in node_ind]


# Quadrature rules to be used for numerical integration of the element matrices 
# Lifted from polyfem, only two rules -- there are more there

list_to_pairs = lambda quad_pts_raw: [(quad_pts_raw[2*i], quad_pts_raw[2*i + 1]) for i in range(len(quad_pts_raw)//2)]


quad_pts_raw_12 = [
    0.06751786707392436, 0.8700998678316848,
    0.32150249385201563, 0.6232720494910644,
    0.660949196186798, 0.30472650086810704,
    0.277716166976405, 0.20644149866999495,
    0.06238226509439084, 0.06751786707392436,
    0.05522545665692, 0.32150249385201563,
    0.03432430294509488, 0.660949196186798,
    0.5158423343536001, 0.277716166976405,
    0.8700998678316848, 0.06238226509439084,
    0.6232720494910644, 0.05522545665692,
    0.30472650086810704, 0.03432430294509488,
    0.20644149866999495, 0.5158423343536001
]

quad_pts_12 = list_to_pairs(quad_pts_raw_12)

weights_12 = [
    0.053034056314869, 0.08776281742889622, 0.05755008556995056, 0.13498637401961758,
    0.053034056314869, 0.08776281742889622, 0.05755008556995056, 0.13498637401961758,
    0.053034056314869, 0.08776281742889622, 0.05755008556995056, 0.13498637401961758
]

quad_pts_raw_16 = [
    0.3333333333333333, 0.3333333333333333,
    0.4592925882927229, 0.08141482341455419,
    0.05054722831703103, 0.898905543365938,
    0.1705693077517601, 0.6588613844964798,
    0.08141482341455419, 0.4592925882927229,
    0.898905543365938, 0.05054722831703103,
    0.6588613844964798, 0.1705693077517601,
    0.4592925882927229, 0.4592925882927229,
    0.05054722831703103, 0.05054722831703103,
    0.1705693077517601, 0.1705693077517601,
    0.7284923929554041, 0.26311282963463867,
    0.008394777409957211, 0.7284923929554041,
    0.26311282963463867, 0.008394777409957211,
    0.008394777409957211, 0.26311282963463867,
    0.7284923929554041, 0.008394777409957211,
    0.26311282963463867, 0.7284923929554041
]

quad_pts_16 = list_to_pairs(quad_pts_raw_16)

# Higher order quadrature
weights_16 = [
    0.1443156076777862,
    0.09509163426728497,
    0.03245849762319813,
    0.1032173705347184,
    0.09509163426728497,
    0.03245849762319813,
    0.1032173705347184,
    0.09509163426728497,
    0.03245849762319813,
    0.1032173705347184,
    0.027230314174434864,
    0.027230314174434864,
    0.027230314174434864,
    0.027230314174434864,
    0.027230314174434864,
    0.027230314174434864
]


def compute_derivatives(cp3d_vecs):
    """
    cp3d_vecs: list of 3D vectors (each is a list of 3 sympy symbols), length 10
    Returns: tuple (first_derivatives, second_derivatives)
             each is a list of 3D vectors, length 2 and 3 respectively
    """
    # Matrix dimensions: B1 is 2 x 10, B2 is 3 x 10
    # cp3d_vecs must be reshaped as 3 x 10 matrix: one for x, y, z separately
    cp_matrix = [Matrix([vec[i] for vec in cp3d_vecs]) for i in range(3)]  # list of 3 column vectors (len 10)

    # Multiply basis derivative matrices with x/y/z coefficient vectors
    f1 = [ [cp_matrix[i].dot(row) for i in range(3)] for row in B1.tolist() ]  # 2 x 3
    f2 = [ [cp_matrix[i].dot(row) for i in range(3)] for row in B2.tolist() ]  # 3 x 3

    return f1, f2

def compute_EFG(f1, f2):
    """
    f1: list of two 3D vectors [df/ds, df/dt]
    f2: list of three 3D vectors [d²f/ds², d²f/dt², d²f/dsdt]
    Returns: [ [E,F,G], [Es,Fs,Gs], [Et,Ft,Gt] ]
    """
    fs, ft = f1
    fss, ftt, fst = f2

    E  = dot3(fs, fs)
    F  = dot3(fs, ft)
    G  = dot3(ft, ft)

    Es = 2 * dot3(fss, fs)
    Fs = dot3(fss, ft) + dot3(fs, fst)
    Gs = 2 * dot3(fst, ft)

    Et = 2 * dot3(fst, fs)
    Ft = dot3(fst, ft) + dot3(fs, ftt)
    Gt = 2 * dot3(ftt, ft)

    return [[E, F, G], [Es, Fs, Gs], [Et, Ft, Gt]]

def generate_compute_LB_function():
    s, t = symbols('s t')
    g = Function('g')(s, t)
    E = Function('E')(s, t)
    F = Function('F')(s, t)
    G = Function('G')(s, t)

    # First derivatives of g
    g_s = Symbol('g_s')
    g_t = Symbol('g_t')
    g_ss = Symbol('g_ss')
    g_tt = Symbol('g_tt')
    g_st = Symbol('g_st')

    # Partial derivatives of E, F, G
    Es = Symbol('Es')
    Fs = Symbol('Fs')
    Gs = Symbol('Gs')
    Et = Symbol('Et')
    Ft = Symbol('Ft')
    Gt = Symbol('Gt')

    # Area term
    We = E*G - F**2
    sqrt_We = sqrt(We)

    # Laplace-Beltrami operator in symbolic form
    Delta_g = (1 / sqrt_We) * (
        diff((F * diff(g, t) - G * diff(g, s)) / sqrt_We, s) +
        diff((F * diff(g, s) - E * diff(g, t)) / sqrt_We, t)
    )
    
    # Substitution rules for simplifying Delta_g
    subs_dict = {
        diff(g, s): g_s,
        diff(g, t): g_t,
        diff(g, s, s): g_ss,
        diff(g, t, t): g_tt,
        diff(g, s, t): g_st,
        diff(g, t, s): g_st,
        E: Symbol('E'), F: Symbol('F'), G: Symbol('G'),
        diff(E, s): Es, diff(F, s): Fs, diff(G, s): Gs,
        diff(E, t): Et, diff(F, t): Ft, diff(G, t): Gt
    }
    
    Delta_g_subs = Delta_g.subs(subs_dict)
    Delta_g_subs = simplify(Delta_g_subs)

    # Expand and collect in terms of function derivatives
    fun_vars = [g_s, g_t, g_ss, g_tt, g_st]
    Delta_g_collected =  Poly(expand(Delta_g_subs), fun_vars)
    
    # Extract coefficients
    LB_coeffs = [expand(Delta_g_collected.coeff_monomial(var)) for var in fun_vars]
    
    # Construct the function of the array EFG
    def LB(EFG):
        # EFG is a 3x3 list [[E, F, G], [Es, Fs, Gs], [Et, Ft, Gt]]
        E_dict = {
            Symbol('E'): EFG[0][0], Symbol('F'): EFG[0][1], Symbol('G'): EFG[0][2],
            Es: EFG[1][0], Fs: EFG[1][1], Gs: EFG[1][2],
            Et: EFG[2][0], Ft: EFG[2][1], Gt: EFG[2][2]
        }

        return [c.subs(E_dict) for c in LB_coeffs]

    return LB


def apply_LB(LBc, g1, g2):
    """
    LBc: list of 5 Laplace-Beltrami coefficients
    g1: list of two first derivatives [g_s, g_t]
    g2: list of three second derivatives [g_ss, g_tt, g_st]
    """
    if type(g1[0]) == list:     
        return LBc[0]*Matrix(g1[0]) + LBc[1]*Matrix(g1[1]) + LBc[2]*Matrix(g2[0]) + LBc[3]*Matrix(g2[1]) + LBc[4]*Matrix(g2[2])
    else:
        return LBc[0]*g1[0] + LBc[1]*g1[1] + LBc[2]*g2[0] + LBc[3]*g2[1] + LBc[4]*g2[2]


def compute_LB_of_basis(LBc):
    """
    LBc: list of 5 Laplace-Beltrami coefficients
    Returns: list of 10 expressions (one per basis function)
    """
    return [
        apply_LB(LBc,
                 [B1[0, i], B1[1, i]],  # first derivatives of basis function i
                 [B2[0, i], B2[1, i], B2[2, i]])  # second derivatives of basis function i
        for i in range(10)
    ]

def compute_elem_matrix(cp3d, quad_pts, weights):
    """
    cp3d: list of 10 control points in 3D, each a list of [x, y, z] (length-3)
    quad_pts: list of (s, t) pairs for quadrature points
    weights: list of quadrature weights

    Returns: 10×10 element stiffness matrix (SymPy Matrix)
    """

    A = sp.zeros(10, 10)
    LB_coeff_basis = []
    W = []

    # For each quadrature point
    for m, (s_val, t_val) in enumerate(quad_pts):
        subs_dict = {sp.symbols('s'): s_val, sp.symbols('t'): t_val}

        # Evaluate first and second derivatives at (s, t)
        f1, f2 = compute_derivatives(cp3d)
        f1_eval = [[comp.subs(subs_dict) for comp in vec] for vec in f1]
        f2_eval = [[comp.subs(subs_dict) for comp in vec] for vec in f2]

        # Compute fundamental form and its derivatives
        EFG = compute_EFG(f1_eval, f2_eval)

        E, F, G = EFG[0]
        We_val = E * G - F**2
        W.append(We_val)
        
        compute_LB_coeffs = generate_compute_LB_function()
        # Compute LB coefficients and basis values at this quad point
        LBc = compute_LB_coeffs(EFG)
        LBc_basis = [val.subs(subs_dict) for val in compute_LB_of_basis(LBc)]
        LB_coeff_basis.append(LBc_basis)

    # Assemble 10x10 element matrix
    for i in range(10):
        for j in range(10):
            A[i, j] = sum(
                sp.sqrt(W[m]) * LB_coeff_basis[m][i] * LB_coeff_basis[m][j] * weights[m]
                for m in range(len(weights))
            )
    return A

# Functions needed for testing 

def to_bezier_bary_3D(expr_vec):
    """
    Given a 3D vector-valued expression [f1, f2, f3] as functions of s,t,
    return 10 Bezier control points.
    """
    global cp, B, node_ind
    bezier_cp = []
    for i in range(3):  # x, y, z components
        # Construct symbolic patch
        patch_expr = sum(B[j]*cp[j] for j in range(10))
        # Collect coefficients wrt monomials in s, t
        eqns = Poly( expand(patch_expr-expr_vec[i]), [s,t]).coeffs()
        sol = solve(eqns, cp)
        # Extract control points for this component
        component_cp = [sol[c] for c in cp]
        bezier_cp.append(component_cp)

    # Transpose to get 10 control points each with [x, y, z]
    return [ [bezier_cp[i][j] for i in range(3)] for j in range(10) ]


import mpmath
mpmath.mp.dps = 16  # digits of precision
mpmath.mp.eps = 1e-6  
from mpmath import quad
def test_elem_matrix(metric_expr, fun_expr, quad_pts, weights):
    """
    Compare energy integral computed using stiffness matrix with direct symbolic integration.
    
    metric_expr: list [x(s,t), y(s,t), z(s,t)]
    fun_expr: scalar function g(s,t)
    quad_pts: list of (s, t) tuples
    weights: corresponding list of weights
    """
    s, t = sp.symbols('s t')

    # Compute derivatives of the metric and function expressions
    metric_deriv = compute_derivatives_closed( metric_expr)
    fun_deriv = compute_derivatives_closed([fun_expr, 0, 0])

    # Only use first component of function derivatives (i.e. scalar function)
    fun_f1 = [vec[0] for vec in fun_deriv[0]]
    fun_f2 = [vec[0] for vec in fun_deriv[1]]

    # Compute E, F, G and We
    EFG = compute_EFG(metric_deriv[0], metric_deriv[1])
    E, F, G = EFG[0]
    We = E * G - F**2

    # Laplace-Beltrami applied to function
    compute_LB_coeffs = generate_compute_LB_function()
    LBc = compute_LB_coeffs(EFG)
    LB_val = apply_LB(LBc, fun_f1, fun_f2)
    LB_squared = (LB_val**2 * sp.sqrt(We))
    LB_squared_fun = lambdify((t, s), LB_squared, modules='mpmath')

    # integrating numerically instead of symbolically (again sympy not powerful enough) but using adaptive mpmath quad
    int_adapt_quad = quad(lambda t: quad(lambda s: LB_squared_fun(t, s), [0, 1 - t]), [0, 1])

    metric_cp = to_bezier_bary_3D(metric_expr)  # should return 10 control points
    fun_cp_vec = to_bezier_bary_3D([fun_expr, 0, 0])  # function as 3D vector field (only x-component used)
    fun_cp = [vec[0] for vec in fun_cp_vec]

    # Fixed quadrature-based integral: cp^T A cp / 2
    A = compute_elem_matrix(metric_cp, quad_pts, weights)
    fun_cp_col = sp.Matrix(fun_cp).reshape(10, 1)
    int_quad = (fun_cp_col.T * A * fun_cp_col)[0, 0] / 2

    # Output results
    print("Energy from stiffness matrix:", int_quad)
    print("Energy from adaptive quadrature integration of the integrand function:", int_adapt_quad)
    rel_diff = (int_quad - int_adapt_quad) / max(abs(int_quad), abs(int_adapt_quad))
    print("Relative difference:", rel_diff)

    return int_quad, int_adapt_quad, rel_diff

# For testing on analytically defined surfaces 
def compute_derivatives_closed(expr):
    # Returns ([df/ds, df/dt], [d²f/ds², d²f/dt², d²f/dsdt])
    f1 = [ [diff(comp, s) for comp in expr],
           [diff(comp, t) for comp in expr] ]
    f2 = [ [diff(comp, s, s) for comp in expr],
           [diff(comp, t, t) for comp in expr],
           [diff(comp, s, t) for comp in expr] ]
    return f1, f2

# Apply LB operator and evaluate squared norm
def eval_LB_norm(surface_expr, label, expected):
    f1, f2 = compute_derivatives_closed(surface_expr)
    EFG = compute_EFG(f1, f2)
    compute_LB_coeffs = generate_compute_LB_function()
    LBc = compute_LB_coeffs(EFG)
    LBv = apply_LB(LBc, f1, f2)
    LB_sq_norm = simplify(dot3(LBv, LBv))
    print(f"{label}, should be {expected}:", LB_sq_norm)


# Testing on simple analytic surfaces
def test_simple_analytic():
    compute_LB_coeffs = generate_compute_LB_function()
    
    # Parametric surfaces
    s, t = symbols('s t')
    
    sphere = [s, t, sqrt(1 - s**2 - t**2)]
    catenoid = [cosh(t)*cos(s), cosh(t)*sin(s), t]
    epper = [(1/3)*s*(1 - s**2/3 + t**2), (1/3)*t*(1 - t**2/3 + s**2), (1/3)*(s**2 - t**2)]
    cylinder = [cos(pi/4)*cos(s) - sin(pi/4)*t,
                sin(s),
                cos(pi/4)*t + sin(pi/4)*cos(s)]
    
    
    eval_LB_norm(sphere, "Sphere", "4")
    eval_LB_norm(cylinder, "Rotated cylinder", "1")
    eval_LB_norm(catenoid, "Catenoid", "0")
    eval_LB_norm(epper, "Epper", "0")

# Unduloid is a more complex surface, so we will test it separately
# This should also be an analytic test, but sympy simplify is too slow, so just testing at specific points

from sympy.functions.special.elliptic_integrals import elliptic_f, elliptic_e
from sympy import Rational

def make_unduloid():
    c, a = 2, 1
    k = sqrt(c**2 - a**2)/c
    mu = 2 / Rational(a + c)
    m = (c**2 - a**2) / Rational(2)
    n = (c**2 + a**2) / Rational(2)
    phi = mu * s / Rational(2) - pi/Rational(4)

    return [
        a * elliptic_f(phi, k) + c * elliptic_e(phi, k),
        sqrt(m*sin(mu*s) + n) * cos(t),
        sqrt(m*sin(mu*s) + n) * sin(t)
    ]

def test_unduloid(): 
    unduloid = make_unduloid()
    unduloid_f1, unduloid_f2 = compute_derivatives_closed(unduloid)
    EFG_unduloid = compute_EFG(unduloid_f1, unduloid_f2)
    compute_LB_coeffs = generate_compute_LB_function()
    LBc_unduloid = compute_LB_coeffs(EFG_unduloid)
    LBv_unduloid = apply_LB(LBc_unduloid, unduloid_f1, unduloid_f2)
    H2 = (dot3(LBv_unduloid, LBv_unduloid))
    
    x,y = symbols('x y')
    subs_trig = { 
        cos(s/3 + pi/4): expand(cos(s/3 + pi/4),trig=True),
        sin(s/3 + pi/4): expand(sin(s/3 + pi/4),trug=True)
    }
    
    subs_exprs = {
        cos(t): y, 
        sin(t):sqrt(1-y**2),
        cos(s/3): x,
        sin(s/3): sqrt(1 - x**2),
        cos(2*s/3): 2*x**2 - 1,
        sin(2*s/3): 2*x*sqrt(1 - x**2),
    }
    print( H2.subs({s:0.5, t:0.1}).evalf(), 4/9)
    print( H2.subs({s:0.01, t:0.2}).evalf(), 4/9)
    #H2_simplified = 
    #print("Unduloid, should be 4/9:", H2_simplified)
    #print(H2_simplified)
    #print(H2_simplified)

def elem_matrix_tests():
    metric_expr = [ (s + 1)**2, (t + 1)**3, s**3 + t**2 ]
    fun_expr = s**3 + 2*t - s*t
    test_elem_matrix(metric_expr, fun_expr, quad_pts_12, weights_12)

# Code generation
def list_to_c_init(name, lst, dims):
    """
    Convert a Python list to a C-style array declaration and initialization string.

    Parameters:
    - name: variable name
    - lst: list (or nested list) of floats
    - dims: shape of the array
    Returns:
    - A C-style string like: double var[10][3] = { ... };
    """
    # Convert list to string with C curly-brace formatting
    def list_to_braced_string(l):
        if isinstance(l, (list, tuple)):
            return "{" + ", ".join(list_to_braced_string(e) for e in l) + "}"
        else:
            return str(float(sp.N(l)))  # Convert to float for numerical output

    array_str = list_to_braced_string(lst)

    # Generate declaration
    decl = f"double {name}" + "".join([f"[{d}]" for d in dims])
    return f"{decl} = {array_str};"

# This is used to construct tests in test_CT_Laplace_Beltrami
def print_cps_for_c(metric_expr, fun_expr):
    """
    Generate and print C-initializer strings for:
    - metric control points (mcp[10][3])
    - scalar function control points (fcp[10])
    Also runs symbolic-numeric test.

    Parameters:
    - metric_expr: list [x(s,t), y(s,t), z(s,t)]
    - fun_expr: scalar expression g(s,t)
    """
    # Compute metric control points
    mcp = to_bezier_bary_3D(metric_expr)
    mcp_str = list_to_c_init("mcp", mcp, [10, 3])

    # Compute function control points (x-component only)
    fcp_vec = to_bezier_bary_3D([fun_expr, 0, 0])
    fcp = [vec[0] for vec in fcp_vec]
    fcp_str = list_to_c_init("fcp", fcp, [10])
    print(mcp_str)
    print(fcp_str)


def generate_c_function_from_expressions(
    expr: Union[List[sp.Expr], List[List[sp.Expr]]],
    func_name: str,
    result_array: str,
    result_dims: List[int],
    coeff_arrays: List[sp.MatrixSymbol] = None,
    coeff_scalars: List[sp.Symbol] = None, 
    use_cse: bool = True,
) -> str:
    """ Generate a C function from a list or list of lists of sympy expressions. 
        The function computes an output C array (1d or 2d) with elements assigned based on 
        experssions.  
    expr: either a list of list of lists of expressions, with free symbols either from 
        coeff_arrays or scalar_vars. If it is a list of lists, all sublists should have the same lengths.
    func_name:  the name of the C function to produce. 
    result_array: the name of the array to return the expression values in 
    resul_dims: a list of length 1 or 2, dimensions of the array [TODO: infer from input]
    coeff_arrays: list of  matrix free symbols used in expressions, as MatrixSymbol
          these are converted to function arguments.  Matrices of size 1 x n become 1D arrays 
          otherwise 2D arrays
    coeff_scalars: list of scalar free variables, become double function arguments. 
    use_cse:  use common expression elimination
    """
    
    coeff_arrays = coeff_arrays or []
    coeff_scalars = coeff_scalars or []
    shape_map = {sym: sym.shape for sym in coeff_arrays}

    # Flatten expressions
    is_nested = isinstance(expr[0], list)
    flat_exprs = [e for row in expr for e in row] if is_nested else expr
    nrows = len(expr) if is_nested else len(flat_exprs)
    ncols = len(expr[0]) if is_nested else 1


    # Recursively replace integer powers with repeated multiplication
    def recursive_pow_replace(expr):
        if isinstance(expr, sp.Pow) and expr.exp.is_Integer and expr.exp > 1:
            res = sp.Mul(*[recursive_pow_replace(expr.base)] * int(expr.exp), evaluate=False)
            return res
        elif expr.args:
            # Recursively apply to all subexpressions
            args = [recursive_pow_replace(arg) for arg in expr.args]
            try:
                return expr.func(*args, evaluate=False)
            except TypeError:
                return expr.func(*args)
        else:
            return expr


    # Apply CSE and replace Pow
    if use_cse:
        cse_repl, reduced_exprs = sp.cse(flat_exprs)
        cse_repl = [(sym, recursive_pow_replace(ex)) for sym, ex in cse_repl]
        reduced_exprs = [recursive_pow_replace(ex) for ex in reduced_exprs]
    else:
        cse_repl = []
        reduced_exprs = [recursive_pow_replace(ex) for ex in flat_exprs]

    class MatrixSymbolPrinter(C99CodePrinter):
        def _print_MatrixElement(self, expr):
            base = self._print(expr.parent)
            i = self._print(expr.i)
            j = self._print(expr.j)
            shape = shape_map.get(expr.parent)
            if shape[0] == 1:
                return f"{base}[{j}]"
            if shape[1] == 1:
                return f"{base}[{i}]"
            else:
                return f"{base}[{i}][{j}]"

    printer = MatrixSymbolPrinter()

    # Function signature
    lines = [f"void {func_name}("]
    for sym in coeff_arrays:
        name = sym.name
        shape = sym.shape
        if  shape[0] == 1:
            lines.append(f"    double {name}[{shape[1]}],")
        elif shape[1] == 1:
            lines.append(f"    double {name}[{shape[0]}],")
        else:
            lines.append(f"    double {name}[{shape[0]}][{shape[1]}],")
    for sym in coeff_scalars: 
        name = sym.name
        lines.append(f" double {name},")
    if len(result_dims) == 1:
        lines.append(f"    double {result_array}[{result_dims[0]}]")
    else:
        lines.append(f"    double {result_array}[{result_dims[0]}][{result_dims[1]}]")
    lines.append(") {")

    # CSE temporaries
    for sym, expr in cse_repl:
        lines.append(f"    double {printer.doprint(sym)} = {printer.doprint(expr)};")

    # Final output assignment
    for i, expr in enumerate(reduced_exprs):
        lhs = (
            f"{result_array}[{i}]"
            if len(result_dims) == 1
            else f"{result_array}[{i // ncols}][{i % ncols}]"
        )
        lines.append(f"    {lhs} = {printer.doprint(expr)};")

    lines.append("}")
    return "\n".join(lines)

def generate_Laplace_Beltrami_element_matrix_C(fname, quad_pts, weights):
    scp3d = sp.MatrixSymbol('cp3d',10,3)
    exprs1, exprs2 = compute_derivatives([[scp3d[i,j] for j in range(3)] for i in range(10)])
    
    compute_first_derivatives_C_str = generate_c_function_from_expressions(
        expr=exprs1,
        func_name='compute_first_derivatives_C',
        result_array='f1',
        result_dims=[2,3],
        coeff_arrays=[scp3d],
        coeff_scalars=[s,t]
    )
    
    compute_second_derivatives_C_str = generate_c_function_from_expressions(
        expr=exprs2,
        func_name='compute_second_derivatives_C',
        result_array='f2',
        result_dims=[3,3],
        coeff_arrays=[scp3d],
        coeff_scalars=[s,t]
    )
    
    sf1 = sp.MatrixSymbol('f1',2,3)
    sf2 = sp.MatrixSymbol('f2',3,3)
    exprs = compute_EFG([ [sf1[i,j] for j in range(3)] for i in range(2)], [ [sf2[i,j] for j in range(3)] for i in range(3)]) 
    
    compute_EFG_C_str = generate_c_function_from_expressions(
        expr=exprs,
        func_name='compute_EFG_C',
        result_array='EFG',
        result_dims=[3,3],
        coeff_arrays=[sf1,sf2],
        coeff_scalars=None
    )
    
    sEFG = sp.MatrixSymbol('EFG',3,3)
    
    compute_LB_coeffs = generate_compute_LB_function()
    exprs = compute_LB_coeffs([ [sEFG[i,j] for j in range(3)] for i in range(3)])
    
    compute_LB_coeffs_C_str = generate_c_function_from_expressions(
        expr=exprs,
        func_name='compute_LB_coeffs_C',
        result_array='LB_coeffs',
        result_dims=[5],
        coeff_arrays={sEFG},
        coeff_scalars=None
    )
    
    sLBc = sp.MatrixSymbol('LBc',1,5)
    exprs = compute_LB_of_basis( [sLBc[0,j] for j in range(5)])
    
    compute_LB_of_basis_C_str = generate_c_function_from_expressions(
        expr=exprs,
        func_name='compute_LB_of_basis_C',
        result_array='LB_of_basis',
        result_dims=[10],
        coeff_arrays={sLBc},
        coeff_scalars=[s,t]
    )
    
    def write_autogenerated_file_header(f):
        f.write("/* !!!! Autogenerated file, do not modify! */\n\n")
    
    quad_pts_str = list_to_c_init("quad_pts", quad_pts, [len(weights), 2])
    weights_str = list_to_c_init("weights", weights, [len(weights)])
        
    with open(fname, "w") as f:
       # f = sys.stdout
        write_autogenerated_file_header(f)
        f.write(f"#define QUAD_DIM {len(weights)}\n\n")
        f.write(quad_pts_str + "\n\n")
        f.write(weights_str + "\n\n")
    
    compute_LB_of_basis_cp_C_str = """
    void compute_LB_of_basis_cp_C(double cp_3d[10][3], double s, double t, double* W, double LB_of_basis[10]) { 
       double f1[2][3]; 
       double f2[3][3]; 
       double EFG[3][3]; 
       double LB_coeffs[5]; 
    
       compute_first_derivatives_C(cp_3d,s,t,f1); 
       compute_second_derivatives_C(cp_3d,s,t,f2); 
       compute_EFG_C(f1,f2,EFG); 
       compute_LB_coeffs_C(EFG,LB_coeffs); 
       compute_LB_of_basis_C(LB_coeffs, s,t, LB_of_basis);
       *W = EFG[0][0]*EFG[0][2]-EFG[0][1]*EFG[0][1];
    }
    """
    
    compute_elem_matrix_C_str = """
    #include <math.h>
    void compute_elem_matrix_C(double cp_3d[10][3], int quad_dim, double quad_pts[][2], double weights[], double A[10][10]) {
       int i,j, m;
       double LB_of_basis[quad_dim][10]; 
       double W[quad_dim];
       for( m=0; m < quad_dim; m++) {
          compute_LB_of_basis_cp_C(cp_3d, quad_pts[m][0], quad_pts[m][1], &(W[m]), LB_of_basis[m] ); 
       }
       
       for( i = 0; i < 10; i++) { 
          for( j = 0; j < 10; j++) { 
              A[i][j] = 0.0;
              for( m = 0; m < quad_dim; m++) { 
                  A[i][j] += LB_of_basis[m][i]*LB_of_basis[m][j]*weights[m]*sqrt(W[m]);
              } 
          } 
       }
    }
    """
    
    with open(fname, "a") as f:
       # f = sys.stdout
        f.write(compute_first_derivatives_C_str  + "\n\n")
        f.write(compute_second_derivatives_C_str  + "\n\n")
        f.write(compute_EFG_C_str + "\n\n")
        f.write(compute_LB_coeffs_C_str + "\n\n")
        f.write(compute_LB_of_basis_C_str+"\n\n")
        f.write(compute_LB_of_basis_cp_C_str + "\n\n")
        f.write(compute_elem_matrix_C_str + "\n\n")


if __name__ == "__main__":
    # Run tests 
    print("Running tests on Python element matrix code")
    test_simple_analytic()
    #test_unduloid()
    elem_matrix_tests()
    # Generate C code

    fname = Path("Clough_Tocher_Laplace_Beltrami.c") 
    print("Generating C code in", fname)
    generate_Laplace_Beltrami_element_matrix_C(fname, quad_pts_12, weights_12)





