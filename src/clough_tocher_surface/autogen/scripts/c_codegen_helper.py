from sympy.printing.c import C99CodePrinter
from typing import List, Union
import sympy as sp

def generate_c_function_from_expressions(
    expr: Union[sp.Expr, List],
    func_name: str,
    result_array: str,
    result_dims: List[int],
    coeff_arrays: List[sp.MatrixSymbol] = None,
    coeff_scalars: List[sp.Symbol] = None,
    use_cse: bool = True,
) -> str:
    """
    Generate a C function from a nested list of sympy expressions representing a tensor.
    Supports tensors of arbitrary rank (depth of nesting).

    expr: Nested list of expressions.
    func_name: Name of the generated C function.
    result_array: Name of the output array variable in C.
    result_dims: Shape of the output array (matches shape of expr).
    coeff_arrays: Matrix symbols used in expressions.
    coeff_scalars: Scalar symbols used in expressions.
    use_cse: Whether to apply common subexpression elimination.
    """
    coeff_arrays = coeff_arrays or []
    coeff_scalars = coeff_scalars or []
    shape_map = {sym: sym.shape for sym in coeff_arrays}

    def flatten_exprs(nested):
        flat = []
        index_map = []

        def recurse(subtree, index_prefix=[]):
            if isinstance(subtree, list):
                for i, child in enumerate(subtree):
                    recurse(child, index_prefix + [i])
            else:
                flat.append(subtree)
                index_map.append(index_prefix)

        recurse(nested)
        return flat, index_map

    flat_exprs, index_map = flatten_exprs(expr)

    def recursive_pow_replace(expr):
        if isinstance(expr, sp.Pow) and expr.exp.is_Integer and expr.exp > 1:
            res = sp.Mul(*[recursive_pow_replace(expr.base)] * int(expr.exp), evaluate=False)
            return res
        elif expr.args:
            args = [recursive_pow_replace(arg) for arg in expr.args]
            try:
                return expr.func(*args, evaluate=False)
            except TypeError:
                return expr.func(*args)
        else:
            return expr

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
        if shape[0] == 1:
            lines.append(f"    double {name}[{shape[1]}],")
        elif shape[1] == 1:
            lines.append(f"    double {name}[{shape[0]}],")
        else:
            lines.append(f"    double {name}[{shape[0]}][{shape[1]}],")
    for sym in coeff_scalars:
        lines.append(f"    double {sym.name},")
    
    dims = ''.join(f'[{d}]' for d in result_dims)
    lines.append(f"    double {result_array}{dims}")
    lines.append(") {")

    for sym, ex in cse_repl:
        lines.append(f"    double {printer.doprint(sym)} = {printer.doprint(ex)};")

    for idx, expr in zip(index_map, reduced_exprs):
        lhs = f"{result_array}" + ''.join(f"[{i}]" for i in idx)
        lines.append(f"    {lhs} = {printer.doprint(expr)};")

    lines.append("}")
    return "\n".join(lines)
