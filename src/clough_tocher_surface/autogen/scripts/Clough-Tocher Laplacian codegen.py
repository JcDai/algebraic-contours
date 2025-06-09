import sympy as sp
from sympy import Matrix, symbols, diff, factorial, simplify, integrate, diag
#from sympy.printing.c import C99CodePrinter
#from typing import List, Union
from pathlib import Path
import sys
from c_codegen_helper import *

# Symbols
s, t = symbols('s t')
u, v = symbols('u v')
a11, a12, a21, a22 = symbols('a11 a12 a21 a22')
r = 1 - s - t

# Node indices for cubic Bernstein basis on triangle
node_ind = [[i, j, 3 - i - j] for i in range(4) for j in range(4 - i)]

# Bernstein basis functions
B = [
    factorial(3) // (factorial(x[0]) * factorial(x[1]) * factorial(x[2])) *
    s**x[0] * t**x[1] * r**x[2]
    for x in node_ind
]

# Second derivatives of basis functions: B_ss, B_tt, B_st (3 x 10 matrix)
B2 = Matrix([
    [diff(b, s, s) for b in B],
    [diff(b, t, t) for b in B],
    [diff(b, s, t) for b in B]
])

# Symbolic control points
cp = [sp.symbols(f'q{x[0]}{x[1]}{x[2]}') for x in node_ind]

# Precompute symbolic integrals for stiffness matrix
BM = {}
for l in range(3):
    for m in range(3):
        for i in range(10):
            for j in range(10):
                integrand = B2[l, i] * B2[m, j]
                val = integrate(integrate(integrand, (s, 0, 1 - t)), (t, 0, 1))
                BM[(l, m, i, j)] = simplify(val)


def compute_T(u_coords, v_coords):
    UV_to_bary = Matrix([
        [u_coords[1] - u_coords[0], u_coords[2] - u_coords[0]],
        [v_coords[1] - v_coords[0], v_coords[2] - v_coords[0]]
    ]).inv()
    pts = [[0, 0], [1, 0], [0, 1], [1/Rational(3), 1/Rational(3)]]
    subtris = [[pts[i - 1] for i in tri] for tri in [[1, 2, 4], [2, 3, 4], [3, 1, 4]]]
    maps = []
    for tri in subtris:
        M = Matrix([
            [tri[1][0] - tri[0][0], tri[2][0] - tri[0][0]],
            [tri[1][1] - tri[0][1], tri[2][1] - tri[0][1]]
        ]).inv()
        maps.append(M @ UV_to_bary)
    return maps

def compute_R(T):
    a11, a12 = T[0, 0], T[0, 1]
    a21, a22 = T[1, 0], T[1, 1]
    return Matrix([
        [a11**2, a21**2, 2*a11*a21],
        [a12**2, a22**2, 2*a12*a22],
        [a11*a12, a21*a22, a11*a22 + a12*a21]
    ])


def compute_AT(M):
    return Matrix(10, 10, lambda i, j: sum(BM[(l, m, i, j)] * M[l, m] for l in range(3) for m in range(3)))

def to_bezier(expr, T, uv0):
    Tinv = T.inv()
    s_t_expr = Tinv @ Matrix([[s],[t]]) + Matrix([[uv0[0]],[uv0[1]]])
    expr_st = sp.expand(expr.subs({u: s_t_expr[0,0], v: s_t_expr[1,0]}))
    patch_expr = sum(B[i] * cp[i] for i in range(10))
    coeffs = sp.Poly(patch_expr - expr_st, s, t).coeffs()
    sol = sp.solve(coeffs, cp)
    return [sol[c] if c in sol else 0 for c in cp]


def test_compute_R():
    T = Matrix([[a11, a12], [a21, a22]])
    R = compute_R(T)
    patch = sum(B[i] * cp[i] for i in range(10))
    p_ss = diff(patch, s, s)
    p_tt = diff(patch, t, t)
    p_st = diff(patch, s, t)
    st = T @ Matrix([u, v])
    patch_uv = patch.subs({s: st[0], t: st[1]})
    lhs = Matrix([diff(patch_uv, u, u), diff(patch_uv, v, v), diff(patch_uv, u, v)])
    rhs = R @ Matrix([p_ss, p_tt, p_st]).subs({s: st[0], t: st[1]})
    return simplify(lhs - rhs) == Matrix([0, 0, 0])


def test_E_symb():
    T = Matrix([[a11, a12], [a21, a22]])
    R = compute_R(T)
    st = T @ Matrix([u, v])
    patch = sum(B[i] * cp[i] for i in range(10))
    patch_uv = patch.subs({s: st[0], t: st[1]})
    Euv = diff(patch_uv, u, u)**2 + 2*diff(patch_uv, u, v)**2 + diff(patch_uv, v, v)**2
    Tinv = T.inv()
    uv = Tinv @ Matrix([s, t])
    Ma = simplify(R.T @ diag(1, 1, 2) @ R)
    Est = Euv.subs({u: uv[0], v: uv[1]})
    Est_poly = sp.Poly(Est, [s,t])
    E_int_s = integrate(Euv.subs({u: uv[0], v: uv[1]}), (s, 0, 1 - t))
    E_int = integrate(E_int_s, (t,0,1))
    AT = compute_AT(Ma)
    cp_vec = Matrix(cp)
    energy_cp = cp_vec.T @ AT @ cp_vec
    return (sp.expand(energy_cp[0,0] - E_int) == 0)


from sympy import Rational

def test_compute_T():
    u_coords = [sp.Symbol(f"u{i+1}") for i in range(3)]
    v_coords = [sp.Symbol(f"v{i+1}") for i in range(3)]
    bary_maps = compute_T(u_coords, v_coords)

    centroid = [(u_coords[0] + u_coords[1] + u_coords[2]) * Rational(1, 3),
                (v_coords[0] + v_coords[1] + v_coords[2]) * Rational(1, 3)]


    subtri_vectors = [
        [
            [u_coords[1] - u_coords[0], v_coords[1] - v_coords[0]],
            [centroid[0] - u_coords[0], centroid[1] - v_coords[0]]
        ],
        [
            [u_coords[2] - u_coords[1], v_coords[2] - v_coords[1]],
            [centroid[0] - u_coords[1], centroid[1] - v_coords[1]]
        ],
        [
            [u_coords[0] - u_coords[2], v_coords[0] - v_coords[2]],
            [centroid[0] - u_coords[2], centroid[1] - v_coords[2]]
        ]
    ]


    for i in range(3):
        T = bary_maps[i]
        for j, vec in enumerate(subtri_vectors[i]):
            transformed = T @ Matrix(vec)
            target = Matrix([1, 0]) if j == 0 else Matrix([0, 1])
            if not all(simplify(transformed[k] - target[k]) == 0 for k in range(2)):
                return False
    return True

def test_to_bezier():
    a = {(i, j): sp.Symbol(f'a{i}{j}') for j in range(4) for i in range(j + 1)}
    expr_test = sum(a[i, j] * u**i * v**(j - i) for j in range(4) for i in range(j + 1))

    T_test = Matrix([[a11, a12], [a21, a22]])
    cp_test = to_bezier(expr_test, T_test, [0, 0])

    patch_expr = sum(B[i] * cp_test[i] for i in range(10))
    st = T_test @ Matrix([u, v])
    patch_uv = patch_expr.subs({s: st[0], t: st[1]})

    residual = simplify(patch_uv - expr_test)
    return residual == 0

def run_test(name, func):
    result = func()
    print(f"{name}: {'success' if result else 'fail'}")

def run_all_tests():
    run_test("test_compute_T", test_compute_T)
    run_test("test_compute_R", test_compute_R)
#    run_test("test_E_symb", test_E_symb)
    run_test("test_to_bezier", test_to_bezier)


def testE_complete(expr_test, corners_test_subs):
    u1, u2, u3 = sp.symbols('u1 u2 u3')
    v1, v2, v3 = sp.symbols('v1 v2 v3')

    ut = [corners_test_subs[u1], corners_test_subs[u2], corners_test_subs[u3]]
    vt = [corners_test_subs[v1], corners_test_subs[v2], corners_test_subs[v3]]
    corners = [[ut[i], vt[i]] for i in range(3)]

    def TrH(e):
        return diff(e, u, u)**2 + diff(e, v, v)**2 + 2 * diff(e, u, v)**2

    if ut[2] != ut[0]:
        line13 = (u - ut[0]) * (vt[2] - vt[0]) / (ut[2] - ut[0]) + vt[0]
    else:
        line13 = vt[0]

    line12 = (u - ut[0]) * (vt[1] - vt[0]) / (ut[1] - ut[0]) + vt[0]
    line23 = (u - ut[2]) * (vt[1] - vt[2]) / (ut[1] - ut[2]) + vt[2]


    E_direct_v1 = sp.expand(integrate(TrH(expr_test), (v, line12, line13)))
    E_direct_v2 = sp.expand(integrate(TrH(expr_test), (v, line12, line23)))
    E_direct = (
        integrate(E_direct_v1, (u, ut[0], ut[2])) +
        integrate(E_direct_v2, (u, ut[2], ut[1]))
    )
    Ttest = compute_T(ut, vt)
    bps = [to_bezier(expr_test, Ttest[i], corners[i]) for i in range(3)]
    R_test = [compute_R(Ttest[i]) for i in range(3)]
    M_test = [simplify(R.T @ diag(1, 1, 2) @ R) for R in R_test]
    
    E_comp = Matrix([[0]])
    for i in range(3):
        bp = Matrix(bps[i])
        AT = compute_AT(M_test[i])
        E_comp += (bp.T @ AT @ bp) / Ttest[i].det()

    res = simplify(E_comp[0] - E_direct)
    return res == 0

def run_all_E_tests():
    expr1 = u**3 - u*v**2 + 2*u**2 + 4*u
    subs1 = {symbols('u1'): -1, symbols('v1'): Rational(1, 2),
             symbols('u2'): 2, symbols('v2'): Rational(-1, 2),
             symbols('u3'): Rational(1, 2), symbols('v3'): 3}
    print(f"testE_complete case 1: {'success' if testE_complete(expr1, subs1) else 'fail'}")
    
    # Test case 2
    expr2 = sp.S.Zero
    subs2 = {symbols('u1'): 0, symbols('v1'): 0,
             symbols('u2'): 1, symbols('v2'): 0,
             symbols('u3'): 0, symbols('v3'): 1}
    print(f"testE_complete case 2: {'success' if testE_complete(expr2, subs2) else 'fail'}")
    
    # Test case 3
    expr3 = u**3 - u*v**2 + 2*u**2 + 4*u
    subs3 = {symbols('u1'): -1, symbols('v1'): 0,
             symbols('u2'): 1,  symbols('v2'): 0,
             symbols('u3'): 0,  symbols('v3'): 1}
    print(f"testE_complete case 3: {'success' if testE_complete(expr3, subs3) else 'fail'}")



def generate_Laplacian_matrix_C_code(fname):
    BMlist = [ 
        [
            [
                [
                    BM[(i,j,k,m)] for m in range(10)
                ]  for k in range(10) 
            ]  for j in range(3) 
        ] for i in range(3) 
    ]
    compute_BM_C_str = generate_c_function_from_expressions(
        BMlist, 
        func_name="compute_BM",
        result_array="BM",
        result_dims=[3,3,10,10],
        coeff_arrays=[],
        coeff_scalars=[])
    
    a11, a12, a21, a22 = symbols('a11 a12 a21 a22')
    T_symb = Matrix([[a11, a12], [a21, a22]])
    R = compute_R(T_symb)
    Ma = (R.T @ diag(1, 1, 2) @ R).applyfunc(simplify)
    
    compute_M_C_str = generate_c_function_from_expressions(Ma.tolist(), 
        func_name="compute_M",
        result_array="Mcoeffs",
        result_dims=[3,3],
        coeff_arrays=[],
        coeff_scalars=[a11,a12,a21,a22])
    
    su = sp.MatrixSymbol('u',1,3)
    sv = sp.MatrixSymbol('v',1,3)
    UV_2_bary_subtri = compute_T(su,sv)
    
    UV_2_bary_subtri_C_str = generate_c_function_from_expressions(
        [UV_2_bary_subtri[i].tolist() for i in range(3)], 
        func_name="compute_UV_2_bary_subtri",
        result_array="UV_2_bary_subtri",
        result_dims=[3,2,2],
        coeff_arrays=[su,sv],
        coeff_scalars=[])
    
    compute_AT_C_str = """\
    void compute_AT(double M[3][3], double BM[3][3][10][10], double AT[10][10]) {
        int i, j, l, m;
        for (i = 0; i < 10; i++)
            for (j = 0; j < 10; j++) {
                AT[i][j] = 0.0;
                for (l = 0; l < 3; l++)
                    for (m = 0; m < 3; m++)
                        AT[i][j] += BM[l][m][i][j] * M[l][m];
            }
    }
    """
    compute_AT_from_uv_C_str = """\
    void compute_AT_from_uv(double u[3], double v[3], int i, double BM[3][3][10][10], double AT[10][10]) {
        double T[3][2][2];
        double M[3][3];
        assert(i >= 0 && i < 3);
        compute_UV_2_bary_subtri(u, v, T);
        compute_M(T[i][0][0], T[i][0][1], T[i][1][0], T[i][1][1], M);
        compute_AT(M, BM, AT);
    }
    """
    
    with open(fname, "w") as outf:
        outf.write("/* !!!! Autogenerated file, do not modify! */\n")
        outf.write("#include <assert.h>\n\n")
        outf.write(compute_BM_C_str+"\n")
        outf.write(compute_M_C_str+"\n\n")
        outf.write(UV_2_bary_subtri_C_str+"\n\n")
        outf.write(compute_AT_C_str+"\n\n")
        outf.write(compute_AT_from_uv_C_str+"\n\n")


if __name__ == "__main__":
    run_all_tests()
    run_all_E_tests()
    fname = "Clough_Tocher_Laplacian.c"
    generate_Laplacian_matrix_C_code(fname)
