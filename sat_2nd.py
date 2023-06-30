from oscillator import OscillatorSimulation, quantize

# jax.config.update('jax_platform_name', 'cpu')
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = "False"

from jax.interpreters import xla
from jax import vmap
from jax import numpy as jnp
from models import check_states, HuboModel, QuadraticModel
from utils import write_results, quantize_spin, get_cases
from tqdm import tqdm
from os.path import join
from sat_problems import load_sat


def exp(dir, csp, num_samples, min_gap, num_cycles, rho, d, coupling_const, shil, omega, num_steps,
        annealing, kuramoto, normalize, seed, num_problem_vars, **kwargs):

    # define model
    qm = QuadraticModel(csp, min_gap, kuramoto=kuramoto)
    hm = HuboModel(csp, 'polynomial_energy', kuramoto=kuramoto)
    osc_sim = OscillatorSimulation(num_cycles=num_cycles, rho=rho, d=d, seed=seed,
                                   coupling_const=coupling_const, shil=shil, omega=omega,
                                   num_oscillators=len(qm.variables),
                                   coupling_func=qm.coupling_func, normalize=normalize,
                                   num_steps=num_steps, annealing=annealing,
                                   kuramoto=kuramoto,

                                   )
    run_vm = vmap(osc_sim.run)
    seeds = jnp.arange(num_samples)
    samples = run_vm(seeds)
    num_steps = samples.shape[1]
    samples = jnp.concatenate(samples)
    description = osc_sim.description
    description.update(qm.description)
    description.update(hm.description)
    description['problem'] = kwargs['problem']
    description['num_constraints'] = len(csp.constraints)
    valid = check_states(samples[:, qm.var_inds], hm)
    valid = jnp.array(valid, dtype=int)

    data = {}

    data['times'] = jnp.concatenate([osc_sim.times for _ in range(num_samples)])
    data['step'] = jnp.concatenate([jnp.arange(num_steps) for _ in range(num_samples)])
    data['sample'] = jnp.concatenate([jnp.array([i] * num_steps) for i in range(num_samples)])
    data.update({'valid': valid})
    if kuramoto:
        energy = jnp.array([hm.energy(sample[qm.var_inds]) for sample in samples])
    else:
        energy = jnp.array([hm.energy(quantize_spin(sample[qm.var_inds])) for sample in samples])

    data['energy'] = energy
    data.update({f'problem_variable_{v}': d for v, d in zip(qm.variables, samples.T[:num_problem_vars])})
    minimum, maximum = param_range(qm.s_lqo[:-1])
    description.update({'param_min': float(minimum)})
    description.update({'param_max': float(maximum)})
    description.update({'param_range': float(maximum - minimum)})
    write_results(dir, description, data)


def param_range(params):
    minimum, maximum = 0, 0
    for param in params:
        minimum = jnp.minimum(minimum, jnp.min(param))
        maximum = jnp.maximum(maximum, jnp.max(param))
    return minimum, maximum


if __name__ == '__main__':
    fixed_params = {
        'num_steps': 2**6,
        'num_samples': 2 ** 6,
        'num_problem_vars': 2 ** 3
    }
    
    # Probelm params
    results_dir = 'results/sat_2nd'
    problems_dir = './sat'
    n_instances = 1
    instance_start = 1
    problems = [f'uf{p}-0{i}.cnf' for p in [20] for i in range(instance_start, instance_start + n_instances)]
    csps = [load_sat(join(problems_dir, problem)) for problem in problems]
    
    # form combinations of params
    # params can be enetered as a list of values, e.g., [1, 2, 3]
    # params can be entered together with keys as tuples and values as list of list, e.g., ('a', 'b'): [[1, 2], [3, 4]]
    params = {
        'dir': [results_dir],
        'seed': [0],
        'num_cycles': [256],
        ('rho', 'd'): [[1, -1],],
        ('coupling_const', 'shil'): [
             [1, 1],
        ],
        'annealing': [True],
        'omega': [0],
        'min_gap': [1],
        ('problem', 'csp'): list(zip(problems, csps)),
        'kuramoto': [False],
        'normalize': [True]

    }
    cases = get_cases(params)
    pbar = tqdm(cases)
    for case in pbar:
        pbar.set_description(str(case))
        exp(**fixed_params, **case)

