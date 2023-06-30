# jax.config.update('jax_platform_name', 'cpu')
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = "False"
from jax import vmap
from jax import numpy as jnp
from jax.experimental.ode import odeint
import itertools
from utils import write_results, quantize_spin, get_cases
from models import check_states, HuboModel as model
from oscillator import OscillatorSimulation, quantize
import dwavebinarycsp as dbcsp
from tqdm import tqdm
from os.path import join
from sat_problems import load_sat

def exp(dir, csp, num_samples, num_cycles, rho, d, coupling_const, shil, omega,
        num_steps, unsat, annealing, num_problem_vars,
        kuramoto, normalize, seed, **kwargs):

    # define model
    m = model(csp, unsat=unsat, kuramoto=kuramoto)
    osc_sim = OscillatorSimulation(num_cycles=num_cycles, rho=rho, d=d, seed=seed,
                                   coupling_const=coupling_const, shil=shil, omega=omega,
                                   num_oscillators=len(m.variables),
                                   coupling_func=m.coupling_func,
                                   num_steps=num_steps,
                                   annealing=annealing,
                                   kuramoto=kuramoto, normalize=normalize
                                   )
    run_vm = vmap(osc_sim.run)
    seeds = jnp.arange(num_samples)
    samples = run_vm(seeds)
    num_steps = samples.shape[1]
    samples = jnp.concatenate(samples)
    description = osc_sim.description
    description.update(m.description)
    description['num_constraints'] = len(csp.constraints)
    description['problem'] = kwargs['problem']
    valid = check_states(samples[:, :m.num_z], m)
    valid = jnp.array(valid, dtype=int)
    data = {}
    data.update({f'problem_variable_{v}': d for v, d in zip(m.variables, samples.T[:num_problem_vars])})
    data['times'] = jnp.concatenate([osc_sim.times for _ in range(num_samples)])
    data['step'] = jnp.concatenate([jnp.arange(num_steps) for _ in range(num_samples)])
    data['sample'] = jnp.concatenate([jnp.array([i] * num_steps) for i in range(num_samples)])
    data.update({'valid': valid})
    if kuramoto:
        energy = jnp.array([m.energy(sample) for sample in samples])
    else:
        energy = jnp.array([m.energy(quantize_spin(sample)) for sample in samples])

    data['energy'] = energy
    constraint_energies = jnp.array([m.const_energy(s) for s in samples[:, :m.num_z]])
    data.update({'seed': jnp.concatenate([jnp.array([s] * num_steps) for s in seeds])})
    write_results(dir, description, data)


if __name__ == '__main__':
    fixed_params = {
        'num_steps': 2**6,
        'num_samples': 2**6,
        'num_problem_vars': 2**3
    }

    # Problem params
    results_dir = 'results/sat_3rd'
    problems_dir = '/home/connor/repositories/hoim/sat'

    n_instances = 2
    instance_start = 1
    problems = [f'uf{p}-0{i}.cnf' for p in [20] for i in range(instance_start, instance_start + n_instances)]
    
    csps = [load_sat(join(problems_dir, problem)) for problem in problems]

    params = {
        'dir': [results_dir],
        'seed': [0],
        'num_cycles': [2**i for i in range(8, 9)],
        ('rho', 'd'): [[1, -1]],
        ('coupling_const', 'shil'): [
            [1, 1],
        ],
        'omega': [0],
        'unsat': [True],
        ('problem', 'csp'): list(zip(problems, csps)),
        'annealing': [True],
        'kuramoto': [False],
        'normalize': [True],
    }

    cases = get_cases(params)
    pbar = tqdm(cases)
    for case in pbar:
        pbar.set_description(str(case))
        exp(**fixed_params, **case)

