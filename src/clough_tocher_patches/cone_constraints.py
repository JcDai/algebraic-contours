from sympy import symbols, Eq, solve, collect, expand
from sympy.printing import ccode

def generate_cone_constraint_coeffs():
    x_i, y_i, z_i, xj, yj, zj, xji, yji, zji = symbols('x_i y_i z_i xj yj zj xji yji zji')
    xcj, xcj_p, ycj, ycj_p, zcj, zcj_p = symbols('xcj xcj_p ycj ycj_p zcj zcj_p')
    xmij, xmij_p, ymij, ymij_p, zmij, zmij_p = symbols('xmij xmij_p ymij ymij_p zmij zmij_p')
    a1c, a2c, a3c, a1m, a2m, a3m, a4m, a5m, a6m = symbols('a1c a2c a3c a1m a2m a3m a4m a5m a6m')
    nix, niy, niz = symbols('nix niy niz')

    dep_vars = [xji, xmij, xmij_p, ymij_p, zmij_p, xcj, xcj_p, ycj_p, zcj_p]
    vars_list = [x_i, y_i, z_i, xj, yj, zj, yji, zji, ycj, zcj, ymij, zmij]

    constraints = [
        Eq(xcj_p, a1c*xji + a2c*xj + a3c*xcj),
        Eq(ycj_p, a1c*yji + a2c*yj + a3c*ycj),
        Eq(zcj_p, a1c*zji + a2c*zj + a3c*zcj),
        Eq(xmij_p, a1m*x_i + a2m*xj + a3m*xji + a4m*xcj + a5m*xcj_p + a6m*xmij),
        Eq(ymij_p, a1m*y_i + a2m*yj + a3m*yji + a4m*ycj + a5m*ycj_p + a6m*ymij),
        Eq(zmij_p, a1m*z_i + a2m*zj + a3m*zji + a4m*zcj + a5m*zcj_p + a6m*zmij),
        Eq(nix*(xji - x_i) + niy*(yji-y_i) + niz*(zji-z_i), 0),
        Eq(nix*(xmij - x_i) + niy*(ymij-y_i) + niz*(zmij-z_i), 0),
        Eq(nix*(xmij_p - x_i) + niy*(ymij_p-y_i) + niz*(zmij_p-z_i), 0)
    ]

    solution = solve(constraints, dep_vars, dict=True)[0]

    for var in dep_vars:
        expr = expand(solution[var])
        coeffs = [collect(expr, v, evaluate=False).get(v, 0) for v in vars_list]
        coeffs_str = ", ".join(ccode(c) for c in coeffs)
        print(f"double {str(var)}[] = {{ {coeffs_str} }};")

if __name__ == '__main__':
    generate_cone_constraint_coeffs()
