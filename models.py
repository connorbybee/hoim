import dimod
import dwavebinarycsp
import dwavebinarycsp as dbc
import numpy
import numpy as np
from jax import numpy as jnp, jit, grad
from energy import config_energy, ising_energy, kuramoto_config_energy
from utils import quantize_binary
from csp import csp_to_cup


class Model:
    def check(self, state):
        pass


def check_states(sample_states: numpy.ndarray,
                 model: Model):
    # Samples: time x variable
    valid = [model.check(s) for s in sample_states]
    return valid


class HuboModel:
    def __init__(self,
                 csp: dwavebinarycsp.ConstraintSatisfactionProblem,
                 unsat=False,
                 holomorphic=True,
                 kuramoto=False,
                 ):
        self.csp = csp
        self.kuramoto = kuramoto
        if unsat:
            self.cup = csp_to_cup(csp)
            self.configs = [np.array([np.array(c) for c in const.configurations])
                            for const in self.cup.constraints]
        else:
            self.configs = [np.array([np.array(c) for c in const.configurations])
                            for const in csp.constraints]
        self.variables = list(csp.variables)
        self.var_order = list(csp.variables)
        self.num_z = len(csp.variables)
        self.const_var_inds = [np.array([self.var_order.index(v) for v in const.variables], dtype=int) for const in csp.constraints]
        self.description = {
            'csp_size': len(csp.variables),
            'model_size': len(csp.variables),
            'num_constraints': len(self.csp.constraints),
            'kuramoto': kuramoto,
            'unsat': unsat
        }
        self.var_inds = np.array(list(range(len(self.variables))), dtype=int)
        self.holomorphic = holomorphic
        self.ind_mats = {}
        self.config_mats = {}
        self.num_configs = {}
        self.const_inds = {}
        for c, (var_inds, config) in enumerate(zip(self.const_var_inds, self.configs)):
            k = len(var_inds)
            l = len(config)
            if k in self.ind_mats:
                if l in self.ind_mats[k]:
                    self.num_configs[k][l] += 1
                    self.ind_mats[k][l] = jnp.concatenate([self.ind_mats[k][l], jnp.tile(var_inds, (len(config), 1))[None, :, :]])
                    self.config_mats[k][l] = jnp.concatenate([self.config_mats[k][l], config[None, :, :]])
                else:
                    self.num_configs[k][l] = 1
                    self.ind_mats[k][l] = jnp.tile(var_inds, (len(config), 1))[None, :, :]
                    self.config_mats[k][l] = config[None, :, :]
            else:
                self.num_configs[k] = {}
                self.ind_mats[k] = {}
                self.config_mats[k] = {}
                self.num_configs[k][l] = 1
                self.ind_mats[k][l] = jnp.tile(var_inds, (len(config), 1))[None, :, :]
                self.config_mats[k][l] = config[None, :, :]

        if kuramoto:
            e_func = kuramoto_config_energy
            self.holomorphic = False
        else:
            e_func = config_energy
        if unsat:
            # @jit
            def const_energy(z):
                c = jnp.concatenate([-e_func(z.take(self.ind_mats[k][l]), self.config_mats[k][l]).sum(-1)
                                     for k in self.ind_mats for l in self.ind_mats[k]])
                return c
        else:
            # @jit
            def const_energy(z):
                c = jnp.concatenate([1 + e_func(z.take(self.ind_mats[k][l]), self.config_mats[k][l]).sum(-1)
                                     for k in self.ind_mats for l in self.ind_mats[k]])
                return c

        # @jit
        def energy(z):
            e = 0
            c = const_energy(z)
            e = c.sum()
            return e


        # @jit
        def coupling_func(z):
            dz = -grad(energy, holomorphic=self.holomorphic)(z)#.conj()
            return dz

        # @jit
        def venergy(z):
            e = const_energy(z).sum()
            return e

        self.energy = venergy
        self.const_energy = const_energy
        self.coupling_func = coupling_func

    def check(self, state):
        if self.kuramoto:
            state = jnp.exp(1j * state)
        x = quantize_binary(state)[self.var_inds]
        x = {v: int(s) for v, s in zip(self.variables, x)}
        return self.csp.check(x)



class QuadraticModel(Model):
    def __init__(self,
                 csp: dwavebinarycsp.ConstraintSatisfactionProblem,
                 min_gap: float,
                 kuramoto=False,
                 type=''):
        self.csp = csp  # type: dwavebinarycsp.ConstraintSatisfactionProblem
        self.bqm = csp_to_bqm(csp, min_gap)  # type: dimod.BinaryQuadraticModel
        # print(self.bqm.num_interactions, self.bqm.num_variables)
        self.variables = list(self.bqm.variables)
        self.b_lqo = bqm_to_numpy(self.bqm, dimod.BINARY)
        self.s_lqo = bqm_to_numpy(self.bqm, dimod.SPIN)
        self.bqm_ind_map = {v: i for i, v in enumerate(list(self.bqm.variables))}
        self.var_inds = np.array([self.bqm_ind_map[v] for v in list(self.csp.variables)])
        self.type = type
        self.description = {
            'csp_size': len(csp.variables),
            'model_size': len(self.bqm.variables),
            'min_gap': min_gap,
            'num_interactions': self.bqm.num_interactions,
            'kuramoto': kuramoto,

        }
        self.kuramoto = kuramoto

        if kuramoto:
            def e(z):
                return kuramoto_ising_energy(z, *self.s_lqo)

            def coupling_func(z):
                # Compute energy for csp
                return kuramoto_ising_grad(z, *self.s_lqo)
        else:
            def e(z):
                return ising_energy(z, *self.s_lqo)

            def coupling_func(z):
                # Compute energy for csp
                return -grad(e, holomorphic=True)(z).conj()
        self.energy = e
        self.coupling_func = coupling_func

    def check_states(self, states: numpy.ndarray):
        return [self.csp.check({
            variable: value for variable, value in zip(self.csp.variables, s)
        }) for s in states]

    def check(self, state):
        x = quantize_binary(state)[self.var_inds]
        state = {v: int(s) for v, s in zip(self.csp.variables, x)}
        try:
            check = self.csp.check(state)
        except Exception as e:
            print(e, state, self.variables, self.csp.variables)
        return check


def csp_to_bqm(csp, min_gap=2):
    return dbc.stitch(csp, min_classical_gap=min_gap, max_graph_size=16)


def bqm_to_numpy(bqm: dimod.BinaryQuadraticModel, vartype: dimod.Vartype):
    variables = list(bqm.variables)
    var_map = {v: i for i, v in enumerate(variables)}
    if vartype == dimod.BINARY:
        linear, quadratic, offset = bqm.linear, bqm.quadratic, bqm.offset
    elif vartype == dimod.SPIN:
        linear, quadratic, offset = bqm.spin.linear, bqm.spin.quadratic, bqm.spin.offset

    linear_n = np.array([linear[v] for v in variables])

    quadratic_n = np.zeros(2 * [len(variables)])

    for (v1, v2), val in quadratic.items():
        quadratic_n[var_map[v1], var_map[v2]] = val

    quadratic_n = 0.5 * (quadratic_n + quadratic_n.T)
    return linear_n, quadratic_n, offset