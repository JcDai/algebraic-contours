from sympy import Matrix, Rational, Symbol, sqrt, symbols, solve, collect, factorial, expand, diff, simplify

def get_pijm_coeffs(T, Tp):
    # Compute key geometric vectors
    m01 = T[2] - (T[0] + T[1])*Rational(1,2)
    m01p = Tp[2] - (Tp[0] + Tp[1])*Rational(1,2)
    e01 = T[1] - T[0]
    e01_n = e01 / sqrt(e01.dot(e01))
    e01_np = -e01_n
    eperp = Matrix([-e01[1], e01[0]])
    eperp_n = Matrix([-e01_n[1], e01_n[0]])
    eperp_np = -eperp_n
    cme = Matrix([[
        Rational(-3, 8), Rational(-3, 8),
        Rational(-9, 8), Rational(-9, 8),
        Rational(3, 4), Rational(3, 4),
        Rational(3, 2)
    ]])
    KN = Matrix([
        [1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [-3, 0, 3, 0, 0, 0, 0],
        [0, -3, 0, 3, 0, 0, 0],
        cme.tolist()[0]
    ])
    ce = Matrix([[Rational(-3, 2), Rational(3, 2), Rational(-1, 4), Rational(1, 4), 0]])
    len_e01 = sqrt(e01.dot(e01))
    MN = m01.dot(e01_n) * (ce * KN) / len_e01
    k = m01.dot(eperp_n)

    MNp = m01p.dot(e01_np) * (ce * KN) / len_e01
    kp = m01p.dot(eperp_np)

    pFN = Matrix([[1,0,0, 0,0,0, 0,0,0], [0,1,0, 0,0,0, 0,0,0], [0,0,1, 0,0,0, 0,0,0], 
                 [0,0,0, 1,0,0, 0,0,0], [0,0,0, 0,1,0, 0,0,0], [0,0,0, 0,0,1, 0,0,0], 
                 [0,0,0, 0,0,0, 0,0,1]])
    pFNp = Matrix([
        [0,1,0, 0,0,0, 0,0,0],
        [1,0,0, 0,0,0, 0,0,0],
        [0,0,0, 1,0,0, 0,0,0],
        [0,0,1, 0,0,0, 0,0,0],
        [0,0,0, 0,0,0, 0,1,0],
        [0,0,0, 0,0,0, 1,0,0],
        [0,0,0, 0,0,0, 0,0,1]
    ])
    pFNrp = Matrix([
        [0,1,0, 0,0,0, 0,0,0],
        [1,0,0, 0,0,0, 0,0,0],
        [0,0,0, 1,0,0, 0,0,0],
        [0,0,1, 0,0,0, 0,0,0],
        [0,0,0, 0,0,0, 0,1,0],
        [0,0,0, 0,0,0, 1,0,0]
    ])

    CM = (cme - MN) / k
    CMp = (cme - MNp) / kp
    CMrp = CMp[:6]     
    alpha_p = CMp[6] 

    pmij_p_coeffs = -((CM * pFN + Matrix(CMrp).T * pFNrp) / alpha_p)

    return pmij_p_coeffs


def get_transform_subs(T):
    a00, a01, a10, a11, b0, b1 = symbols('a00 a01 a10 a11 b0 b1')
    u, v = symbols('u v')
    A = Matrix([[a00, a01], [a10, a11]])
    b = Matrix([b0, b1])

    eqs = []
    eqs += (A * T[0] + b - Matrix([1, 0])).tolist()
    eqs += (A * T[1] + b - Matrix([0, 1])).tolist()
    eqs += (A * T[2] + b - Matrix([0, 0])).tolist()
    eqs = [item for sublist in eqs for item in sublist]  # flatten

    # Solve the system for the affine transform coefficients
    sol = solve(eqs, [a00, a01, a10, a11, b0, b1], dict=True)
    if not sol:
        raise ValueError("Transformation could not be determined from triangle T.")
    sol = sol[0]

    # Substitute back into A and b
    As = A.subs(sol)
    bs = b.subs(sol)

    # Return substitution for u and v to global coordinates
    global_coords = As * Matrix([u, v]) + bs
    return {u: global_coords[0], v: global_coords[1]}

# Bernstein basis
u, v = symbols('u v')
pind = [[i, j, 3 - i - j] for i in range(4) for j in range(4 - i)]
B = [
        factorial(3) / (factorial(i) * factorial(j) * factorial(k)) *
        u**i * v**j * (1 - u - v)**k
        for i, j, k in pind
]

def get_bezier_pts(poly, T):    
    Tglob_uv = get_transform_subs(T)
    
    cp = [Symbol(f'q{i}{j}{k}') for i, j, k in pind]
    patch = sum(B[i] * cp[i] for i in range(len(cp)))
    patch_sub = patch.subs(Tglob_uv)

    eqs = []
    for i in range(4):
        for j in range(4 - i):
            coeff_poly = poly.expand().coeff(u, i).coeff(v, j)
            coeff_patch = patch_sub.expand().coeff(u, i).coeff(v, j)
            eqs.append(coeff_poly - coeff_patch)

    sol = solve(eqs, cp, dict=True)
    return [sol[0][c] for c in cp]

if 1: #__name__ == '__main__':
        Ttest = list(map(lambda x: Matrix(x),[[0,0],[0,1],[1,0]]))
        Tptest =list(map(lambda x: Matrix(x),[[0,1],[0,0],[-1,0]]))
        print( get_pijm_coeffs(Ttest, Tptest).tolist()[0])
        testbp = get_bezier_pts(u**2, Ttest)
        Tglob_uv = get_transform_subs(Ttest)
        print(simplify( (sum(B[i] * testbp[i] for i in range(len(testbp)))).subs(Tglob_uv)))