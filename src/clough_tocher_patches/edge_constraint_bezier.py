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
    print(CM)
    CMp = (cme - MNp) / kp
    print(CMp)
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
    return {s: global_coords[0], t: global_coords[1]}

# Bernstein basis
s, t = symbols('s t')
pind = [[i, j, 3 - i - j] for i in range(4) for j in range(4 - i)]
B = [
        factorial(3) / (factorial(i) * factorial(j) * factorial(k)) *
        s**i * t**j * (1 - s - t)**k
        for i, j, k in pind
]
# global variables common for two triangles
u, v = symbols('u v')

def get_bezier_pts(poly, T):    
    Tglob_st = get_transform_subs(T)
    
    cp = [Symbol(f'q{i}{j}{k}') for i, j, k in pind]
    patch = sum(B[i] * cp[i] for i in range(len(cp)))
    patch_sub = patch.subs(Tglob_st).expand()

    eqs = []
    for i in range(4):
        for j in range(4 - i):
            coeff_poly = poly.expand().coeff(u, i).coeff(v, j)
            coeff_patch = patch_sub.expand().coeff(u, i).coeff(v, j)
            eqs.append(coeff_poly - coeff_patch)

    sol = solve(eqs, cp, dict=True)

    return [sol[0][c] for c in cp]



if __name__ == '__main__':
        Ttest = list(map(lambda x: Matrix(x),[[Rational(1,2),0],[0,1],[1,0]]))
        Tptest =list(map(lambda x: Matrix(x),[[0,1],[Rational(1,2),0],[-1,0]]))
        print( get_pijm_coeffs(Ttest, Tptest).tolist()[0])

        testpoly = u**2 - v**3
        bezier_pts = get_bezier_pts(testpoly, Ttest)
        Tglob_st = get_transform_subs(Ttest)
        print(simplify( (sum(B[i] * bezier_pts[i] for i in range(len(bezier_pts)))).subs(Tglob_st)))

        # name remappings for control points for 2 patches
        q300, q210, q120, q030, q201, q111, q021 = symbols('q300 q210 q120 q030 q201 q111 q021')
        cp = [Symbol(f"q{i}{j}{k}") for i,j,k in pind]

        pic, pmij, pjc = symbols('pic pmij pjc')
        piv, pij, pji, pjv = symbols('piv pij pji pjv')
        pic_p, pmij_p, pjc_p = symbols('pic_p pmij_p pjc_p')

        # name used in N -> qijk in Tp
        patch2N = {
            pic: q201, pmij: q111, pjc: q021,
            piv: q300, pij: q210, pji: q120, pjv: q030
        }
        # name used in Np -> qijk in Tp
        patch2Np = {
            pic_p: q201, pmij_p: q111, pjc_p: q021,
             piv: q300, pij: q210, pji: q120, pjv: q030
        } 
        # qijk name -> number in cp  
        bpt2ind = {cp[i]: i for i in range(len(cp))}

        bezier_pts = get_bezier_pts(testpoly, Ttest)
        bezier_pts_p = get_bezier_pts(testpoly, Tptest)

        Nfull = [piv, pjv, pij, pji, pic, pjc, pic_p, pjc_p, pmij]
        # extract actual values from bezier arrays using remapping 
        # name in Nfull -> qijk name -> index in cp
        Nfull_s = \
            [bezier_pts[bpt2ind[patch2N[Nfull[i]]]] for i in range(6)] + \
            [bezier_pts_p[bpt2ind[patch2Np[Nfull[i]]]] for i in range(6, 8)] + \
            [bezier_pts[bpt2ind[patch2N[Nfull[8]]]]]
        
        # extract pijm_p similarly
        pijm_p_val = bezier_pts_p[bpt2ind[patch2Np[pmij_p]]]
        cfs =  get_pijm_coeffs(Ttest, Tptest).tolist()[0]
        dot_val = sum(cfs[i] * Nfull_s[i] for i in range(len(cfs)))
        print('checking constraint, should be zero: ', dot_val - pijm_p_val)


        