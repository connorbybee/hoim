import dimod
import dwavebinarycsp as dbc
from utils import binary_set


def csp_to_cup(csp):
    cup = dbc.ConstraintSatisfactionProblem(dimod.BINARY)
    for c in csp.constraints:
        k = len(c.variables)
        bs = {tuple(b) for b in binary_set(k)}
        bs -= c.configurations
        cup.add_constraint(bs,
                           c.variables)
    return cup