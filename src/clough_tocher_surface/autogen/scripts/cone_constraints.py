from sympy import symbols, Eq, solve, collect, expand
from sympy.printing import ccode


def generate_cone_constraint_coeffs_x():
    x_i, y_i, z_i, xj, yj, zj, xji, yji, zji = symbols(
        'x_i y_i z_i xj yj zj xji yji zji')
    xcj, xcj_p, ycj, ycj_p, zcj, zcj_p = symbols(
        'xcj xcj_p ycj ycj_p zcj zcj_p')
    xmij, xmij_p, ymij, ymij_p, zmij, zmij_p = symbols(
        'xmij xmij_p ymij ymij_p zmij zmij_p')
    a1c, a2c, a3c, a1m, a2m, a3m, a4m, a5m, a6m = symbols(
        'a1c a2c a3c a1m a2m a3m a4m a5m a6m')
    nix, niy, niz = symbols('nix niy niz')

    dep_vars = [xji, xmij, xmij_p, ymij_p, zmij_p, xcj_p, ycj_p, zcj_p]
    vars_list = [x_i, y_i, z_i, xj, yj, zj,
                 yji, zji, xcj, ycj, zcj, ymij, zmij]

    constraints = [
        Eq(xcj_p, a1c*xji + a2c*xj + a3c*xcj),
        Eq(ycj_p, a1c*yji + a2c*yj + a3c*ycj),
        Eq(zcj_p, a1c*zji + a2c*zj + a3c*zcj),
        Eq(xmij_p, a1m*x_i + a2m*xj + a3m*xji + a4m*xcj + a5m*xcj_p + a6m*xmij),
        Eq(ymij_p, a1m*y_i + a2m*yj + a3m*yji + a4m*ycj + a5m*ycj_p + a6m*ymij),
        Eq(zmij_p, a1m*z_i + a2m*zj + a3m*zji + a4m*zcj + a5m*zcj_p + a6m*zmij),
        Eq(nix*(xji - x_i) + niy*(yji-y_i) + niz*(zji-z_i), 0),
        Eq(nix*(xmij - x_i) + niy*(ymij-y_i) + niz*(zmij-z_i), 0),
        # Eq(nix*(xmij_p - x_i) + niy*(ymij_p-y_i) + niz*(zmij_p-z_i), 0)
    ]

    solution = solve(constraints, dep_vars, dict=True)[0]

    for i, var in enumerate(dep_vars):
        expr = expand(solution[var])
        coeffs = [collect(expr, v, evaluate=False).get(v, 0)
                  for v in vars_list]
        coeffs_str = ", ".join(ccode(c) for c in coeffs)
        print(f"// {str(var)}")
        print(f"res[{i}] = {{{{  {coeffs_str} }}}};")


def generate_cone_constraint_coeffs_y():
    x_i, y_i, z_i, xj, yj, zj, xji, yji, zji = symbols(
        'x_i y_i z_i xj yj zj xji yji zji')
    xcj, xcj_p, ycj, ycj_p, zcj, zcj_p = symbols(
        'xcj xcj_p ycj ycj_p zcj zcj_p')
    xmij, xmij_p, ymij, ymij_p, zmij, zmij_p = symbols(
        'xmij xmij_p ymij ymij_p zmij zmij_p')
    a1c, a2c, a3c, a1m, a2m, a3m, a4m, a5m, a6m = symbols(
        'a1c a2c a3c a1m a2m a3m a4m a5m a6m')
    nix, niy, niz = symbols('nix niy niz')

    dep_vars = [yji, ymij, xmij_p, ymij_p, zmij_p, xcj_p, ycj_p, zcj_p]
    vars_list = [x_i, y_i, z_i, xj, yj, zj,
                 xji, zji, xcj, ycj, zcj, xmij, zmij]

    constraints = [
        Eq(xcj_p, a1c*xji + a2c*xj + a3c*xcj),
        Eq(ycj_p, a1c*yji + a2c*yj + a3c*ycj),
        Eq(zcj_p, a1c*zji + a2c*zj + a3c*zcj),
        Eq(xmij_p, a1m*x_i + a2m*xj + a3m*xji + a4m*xcj + a5m*xcj_p + a6m*xmij),
        Eq(ymij_p, a1m*y_i + a2m*yj + a3m*yji + a4m*ycj + a5m*ycj_p + a6m*ymij),
        Eq(zmij_p, a1m*z_i + a2m*zj + a3m*zji + a4m*zcj + a5m*zcj_p + a6m*zmij),
        Eq(nix*(xji - x_i) + niy*(yji-y_i) + niz*(zji-z_i), 0),
        Eq(nix*(xmij - x_i) + niy*(ymij-y_i) + niz*(zmij-z_i), 0),
        # Eq(nix*(xmij_p - x_i) + niy*(ymij_p-y_i) + niz*(zmij_p-z_i), 0)
    ]

    solution = solve(constraints, dep_vars, dict=True)[0]

    for i, var in enumerate(dep_vars):
        expr = expand(solution[var])
        coeffs = [collect(expr, v, evaluate=False).get(v, 0)
                  for v in vars_list]
        coeffs_str = ", ".join(ccode(c) for c in coeffs)
        print(f"// {str(var)}")
        print(f"res[{i}] = {{{{  {coeffs_str} }}}};")


def generate_cone_constraint_coeffs_z():
    x_i, y_i, z_i, xj, yj, zj, xji, yji, zji = symbols(
        'x_i y_i z_i xj yj zj xji yji zji')
    xcj, xcj_p, ycj, ycj_p, zcj, zcj_p = symbols(
        'xcj xcj_p ycj ycj_p zcj zcj_p')
    xmij, xmij_p, ymij, ymij_p, zmij, zmij_p = symbols(
        'xmij xmij_p ymij ymij_p zmij zmij_p')
    a1c, a2c, a3c, a1m, a2m, a3m, a4m, a5m, a6m = symbols(
        'a1c a2c a3c a1m a2m a3m a4m a5m a6m')
    nix, niy, niz = symbols('nix niy niz')

    dep_vars = [zji, zmij, xmij_p, ymij_p, zmij_p,  xcj_p, ycj_p, zcj_p]
    vars_list = [x_i, y_i, z_i, xj, yj, zj,
                 xji, yji, xcj, ycj, zcj, xmij, ymij]

    constraints = [
        Eq(xcj_p, a1c*xji + a2c*xj + a3c*xcj),
        Eq(ycj_p, a1c*yji + a2c*yj + a3c*ycj),
        Eq(zcj_p, a1c*zji + a2c*zj + a3c*zcj),
        Eq(xmij_p, a1m*x_i + a2m*xj + a3m*xji + a4m*xcj + a5m*xcj_p + a6m*xmij),
        Eq(ymij_p, a1m*y_i + a2m*yj + a3m*yji + a4m*ycj + a5m*ycj_p + a6m*ymij),
        Eq(zmij_p, a1m*z_i + a2m*zj + a3m*zji + a4m*zcj + a5m*zcj_p + a6m*zmij),
        Eq(nix*(xji - x_i) + niy*(yji-y_i) + niz*(zji-z_i), 0),
        Eq(nix*(xmij - x_i) + niy*(ymij-y_i) + niz*(zmij-z_i), 0),
        # Eq(nix*(xmij_p - x_i) + niy*(ymij_p-y_i) + niz*(zmij_p-z_i), 0)
    ]

    solution = solve(constraints, dep_vars, dict=True)[0]

    for i, var in enumerate(dep_vars):
        expr = expand(solution[var])
        coeffs = [collect(expr, v, evaluate=False).get(v, 0)
                  for v in vars_list]
        coeffs_str = ", ".join(ccode(c) for c in coeffs)
        print(f"// {str(var)}")
        print(f"res[{i}] = {{{{  {coeffs_str} }}}};")


def generate_cone_constraint_coeffs_x_rev():
    # pj is the cone

    x_i, y_i, z_i, xj, yj, zj, xij, yij, zij = symbols(
        'x_i y_i z_i xj yj zj xij yij zij')
    xci, xci_p, yci, yci_p, zci, zci_p = symbols(
        'xci xci_p yci yci_p zci zci_p')
    xmij, xmij_p, ymij, ymij_p, zmij, zmij_p = symbols(
        'xmij xmij_p ymij ymij_p zmij zmij_p')
    a1c, a2c, a3c, a1m, a2m, a3m, a4m, a5m, a6m = symbols(
        'a1c a2c a3c a1m a2m a3m a4m a5m a6m')
    nix, niy, niz = symbols('nix niy niz')

    dep_vars = [xij, xmij, xmij_p, ymij_p, zmij_p,  xci_p, yci_p, zci_p]
    vars_list = [x_i, y_i, z_i, xj, yj, zj,
                 yij, zij, xci, yci, zci, ymij, zmij]

    constraints = [
        Eq(xci_p, a1c*xij + a2c*x_i + a3c*xci),
        Eq(yci_p, a1c*yij + a2c*y_i + a3c*yci),
        Eq(zci_p, a1c*zij + a2c*z_i + a3c*zci),
        Eq(xmij_p, a1m*x_i + a2m*xj + a3m*xij + a4m*xci + a5m*xci_p + a6m*xmij),
        Eq(ymij_p, a1m*y_i + a2m*yj + a3m*yij + a4m*yci + a5m*yci_p + a6m*ymij),
        Eq(zmij_p, a1m*z_i + a2m*zj + a3m*zij + a4m*zci + a5m*zci_p + a6m*zmij),
        Eq(nix*(xij - xj) + niy*(yij-yj) + niz*(zij-zj), 0),
        Eq(nix*(xmij - xj) + niy*(ymij-yj) + niz*(zmij-zj), 0),
        # Eq(nix*(xmij_p - xj) + niy*(ymij_p-yj) + niz*(zmij_p-zj), 0)
    ]

    solution = solve(constraints, dep_vars, dict=True)[0]

    for i, var in enumerate(dep_vars):
        expr = expand(solution[var])
        coeffs = [collect(expr, v, evaluate=False).get(v, 0)
                  for v in vars_list]
        coeffs_str = ", ".join(ccode(c) for c in coeffs)
        print(f"// {str(var)}")
        print(f"res[{i}] = {{{{  {coeffs_str} }}}};")


def generate_cone_constraint_coeffs_y_rev():
    # pj is the cone

    x_i, y_i, z_i, xj, yj, zj, xij, yij, zij = symbols(
        'x_i y_i z_i xj yj zj xij yij zij')
    xci, xci_p, yci, yci_p, zci, zci_p = symbols(
        'xci xci_p yci yci_p zci zci_p')
    xmij, xmij_p, ymij, ymij_p, zmij, zmij_p = symbols(
        'xmij xmij_p ymij ymij_p zmij zmij_p')
    a1c, a2c, a3c, a1m, a2m, a3m, a4m, a5m, a6m = symbols(
        'a1c a2c a3c a1m a2m a3m a4m a5m a6m')
    nix, niy, niz = symbols('nix niy niz')

    dep_vars = [yij, ymij, xmij_p, ymij_p, zmij_p,  xci_p, yci_p, zci_p]
    vars_list = [x_i, y_i, z_i, xj, yj, zj,
                 xij, zij, xci, yci, zci, xmij, zmij]

    constraints = [
        Eq(xci_p, a1c*xij + a2c*x_i + a3c*xci),
        Eq(yci_p, a1c*yij + a2c*y_i + a3c*yci),
        Eq(zci_p, a1c*zij + a2c*z_i + a3c*zci),
        Eq(xmij_p, a1m*x_i + a2m*xj + a3m*xij + a4m*xci + a5m*xci_p + a6m*xmij),
        Eq(ymij_p, a1m*y_i + a2m*yj + a3m*yij + a4m*yci + a5m*yci_p + a6m*ymij),
        Eq(zmij_p, a1m*z_i + a2m*zj + a3m*zij + a4m*zci + a5m*zci_p + a6m*zmij),
        Eq(nix*(xij - xj) + niy*(yij-yj) + niz*(zij-zj), 0),
        Eq(nix*(xmij - xj) + niy*(ymij-yj) + niz*(zmij-zj), 0),
        # Eq(nix*(xmij_p - xj) + niy*(ymij_p-yj) + niz*(zmij_p-zj), 0)
    ]

    solution = solve(constraints, dep_vars, dict=True)[0]

    for i, var in enumerate(dep_vars):
        expr = expand(solution[var])
        coeffs = [collect(expr, v, evaluate=False).get(v, 0)
                  for v in vars_list]
        coeffs_str = ", ".join(ccode(c) for c in coeffs)
        print(f"// {str(var)}")
        print(f"res[{i}] = {{{{  {coeffs_str} }}}};")


def generate_cone_constraint_coeffs_z_rev():
    # pj is the cone

    x_i, y_i, z_i, xj, yj, zj, xij, yij, zij = symbols(
        'x_i y_i z_i xj yj zj xij yij zij')
    xci, xci_p, yci, yci_p, zci, zci_p = symbols(
        'xci xci_p yci yci_p zci zci_p')
    xmij, xmij_p, ymij, ymij_p, zmij, zmij_p = symbols(
        'xmij xmij_p ymij ymij_p zmij zmij_p')
    a1c, a2c, a3c, a1m, a2m, a3m, a4m, a5m, a6m = symbols(
        'a1c a2c a3c a1m a2m a3m a4m a5m a6m')
    nix, niy, niz = symbols('nix niy niz')

    dep_vars = [zij, zmij, xmij_p, ymij_p, zmij_p,  xci_p, yci_p, zci_p]
    vars_list = [x_i, y_i, z_i, xj, yj, zj,
                 xij, yij, xci, yci, zci, xmij, ymij]

    constraints = [
        Eq(xci_p, a1c*xij + a2c*x_i + a3c*xci),
        Eq(yci_p, a1c*yij + a2c*y_i + a3c*yci),
        Eq(zci_p, a1c*zij + a2c*z_i + a3c*zci),
        Eq(xmij_p, a1m*x_i + a2m*xj + a3m*xij + a4m*xci + a5m*xci_p + a6m*xmij),
        Eq(ymij_p, a1m*y_i + a2m*yj + a3m*yij + a4m*yci + a5m*yci_p + a6m*ymij),
        Eq(zmij_p, a1m*z_i + a2m*zj + a3m*zij + a4m*zci + a5m*zci_p + a6m*zmij),
        Eq(nix*(xij - xj) + niy*(yij-yj) + niz*(zij-zj), 0),
        Eq(nix*(xmij - xj) + niy*(ymij-yj) + niz*(zmij-zj), 0),
        # Eq(nix*(xmij_p - xj) + niy*(ymij_p-yj) + niz*(zmij_p-zj), 0)
    ]

    solution = solve(constraints, dep_vars, dict=True)[0]

    for i, var in enumerate(dep_vars):
        expr = expand(solution[var])
        coeffs = [collect(expr, v, evaluate=False).get(v, 0)
                  for v in vars_list]
        coeffs_str = ", ".join(ccode(c) for c in coeffs)
        print(f"// {str(var)}")
        print(f"res[{i}] = {{{{  {coeffs_str} }}}};")


if __name__ == '__main__':
    generate_cone_constraint_coeffs_x()
