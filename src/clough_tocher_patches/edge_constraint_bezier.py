from sympy import Matrix, Rational, sqrt, symbols, solve, collect, factorial, expand, diff, simplify

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

if 1: #__name__ == '__main__':
        Ttest = list(map(lambda x: Matrix(x),[[0,0],[0,1],[1,0]]))
        Tptest =list(map(lambda x: Matrix(x),[[0,1],[0,0],[-1,0]]))
        print( get_pijm_coeffs(Ttest, Tptest).tolist()[0])