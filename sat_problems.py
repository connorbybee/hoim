import dwavebinarycsp as dbcsp
from os.path import join

def load_sat(sat_dir):
    with open(sat_dir, 'r') as fp:
        # csp = dbcsp.cnf.load_cnf(fp)
        csp = load_cnf(fp)

    return csp

import re

import dimod

from dwavebinarycsp import ConstraintSatisfactionProblem

_PROBLEM_REGEX = r'^p cnf (\d+)\s*(\d+)'
_CLAUSE_REGEX = r'^-?[0-9]\d*(?:\W-?[1-9]\d*)*\W0$'


def load_cnf(fp):


    fp = iter(fp)  # handle lists/tuples/etc

    csp = ConstraintSatisfactionProblem(dimod.BINARY)

    # first look for the problem
    num_clauses = num_variables = 0
    problem_pattern = re.compile(_PROBLEM_REGEX)
    for line in fp:
        matches = problem_pattern.findall(line)
        if matches:
            if len(matches) > 1:
                raise ValueError
            nv, nc = matches[0]
            num_variables, num_clauses = int(nv), int(nc)
            break

    # now parse the clauses, picking up where we left off looking for the header
    clause_pattern = re.compile(_CLAUSE_REGEX)
    for line in fp:
        line = line.strip()
        if clause_pattern.match(line) is not None:
            clause = [int(v) for v in line.split(' ')[:-1]]  # line ends with a trailing 0

            # -1 is the notation for NOT(1)
            variables = [abs(v) for v in clause]

            f = _cnf_or(clause)

            csp.add_constraint(f, variables)

    for v in range(1, num_variables+1):
        csp.add_variable(v)
    for v in csp.variables:
        if v > num_variables:
            msg = ("given .cnf file's header defines variables [1, {}] and {} clauses "
                   "but constraints a reference to variable {}").format(num_variables, num_clauses, v)
            raise ValueError(msg)

    if len(csp) != num_clauses:
        msg = ("given .cnf file's header defines {} "
               "clauses but the file contains {}").format(num_clauses, len(csp))
        raise ValueError(msg)

    return csp


def _cnf_or(clause):
    def f(*args):
        return any(v == int(c > 0) for v, c in zip(args, clause))
    return f


def reduce_cnf(fp):
    fp = iter(fp)  # handle lists/tuples/etc

    csp = ConstraintSatisfactionProblem(dimod.BINARY)

    # first look for the problem
    num_clauses = num_variables = 0
    problem_pattern = re.compile(_PROBLEM_REGEX)
    for line in fp:
        matches = problem_pattern.findall(line)
        if matches:
            if len(matches) > 1:
                raise ValueError
            nv, nc = matches[0]
            num_variables, num_clauses = int(nv), int(nc)
            break

    # now parse the clauses, picking up where we left off looking for the header
    clause_pattern = re.compile(_CLAUSE_REGEX)

    for line in fp:
        line = line.strip()
        if clause_pattern.match(line) is not None:
            clause = [int(v) for v in line.split(' ')[:-1]]  # line ends with a trailing 0

            # -1 is the notation for NOT(1)
            variables = [abs(v) for v in clause]

            f = _cnf_or(clause)

            csp.add_constraint(f, variables)

    for v in range(1, num_variables + 1):
        csp.add_variable(v)
    for v in csp.variables:
        if v > num_variables:
            msg = ("given .cnf file's header defines variables [1, {}] and {} clauses "
                   "but constraints a reference to variable {}").format(num_variables, num_clauses, v)
            raise ValueError(msg)

    if len(csp) != num_clauses:
        msg = ("given .cnf file's header defines {} "
               "clauses but the file contains {}").format(num_clauses, len(csp))
        raise ValueError(msg)

    return csp


def reduce_sat(sat_dir):
    with open(sat_dir, 'r') as fp:
        # csp = dbcsp.cnf.load_cnf(fp)
        csp = reduce_cnf(fp)

    return csp


if __name__ == '__main__':
    problems_dir = '/home/connor/repositories/hoim/sat'
    problem = 'uf20-01.cnf'

    reduce_sat(join(problems_dir, problem))