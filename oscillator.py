import numpy as np
from jax import numpy as jnp, jit, random
from jax.experimental.ode import odeint


class OscillatorSimulation:
    def __init__(self,
                 num_oscillators: int,
                 coupling_func,
                 num_cycles,
                 kuramoto,
                 normalize,
                 annealing,
                 rho,
                 d,
                 coupling_const,
                 shil,
                 num_steps,
                 seed,
                 omega=0,
                 ):

        self.grad_f = coupling_func
        self.n = num_oscillators  # number of oscillators
        self.num_cycles = num_cycles
        self.t_end = 2 * jnp.pi * num_cycles
        times = jnp.linspace(0, self.t_end, num=num_steps)
        self.times = times
        self.seed = seed
        self.num_steps = num_steps
        self.kuramoto = kuramoto
        self.normalize = normalize

        self.description = {
            'rho': rho,
            'd': d,
            'coupling_const': coupling_const,
            'shil': shil,
            'annealing': annealing,
            'omega': omega,
            'num_osc': self.n,
            'num_steps': len(self.times),
            'seed': seed,
            'num_cycles': num_cycles,
            'kuramoto': self.kuramoto,
            'normalize': self.normalize
        }

        if self.normalize:
            if self.kuramoto:
                raise ValueError('cannot normalize kuramoto model')
            coupling_func_ = coupling_func
            coupling_func = lambda z: coupling_func_(z / jnp.abs(z))

        if annealing:
            @jit
            def f(z, t):
                dz = (rho + 1j * omega) * z + d * z * (jnp.abs(z) ** 2)  # Non-linear oscillator, van-der pol equation
                t = t / jnp.complex64(self.t_end)  # fraction of simulation
                dz += t * shil * jnp.conj(z)  # sub-harmonic injection locking
                dz += coupling_const * coupling_func(z)  # coupling function
                return dz
        else:
            @jit
            def f(z, t):
                dz = (rho + 1j * omega) * z + d * z * (jnp.abs(z) ** 2)  # Non-linear oscillator, van-der pol equation
                dz += shil * jnp.conj(z)  # sub-harmonic injection locking
                dz += coupling_const * coupling_func(z)  # coupling function
                return dz

        if self.kuramoto:
            @jit
            def f(phi, t):
                t = t / self.t_end
                dphi = -t * shil * jnp.sin(2 * phi)
                dphi += coupling_const * coupling_func(phi)
                return dphi

        self.f = f  # complex-valued model

    def run(self, seed):
        key = random.PRNGKey(seed)
        key, nkey = random.split(key)
        phase = random.uniform(
            key=nkey, minval=-1, maxval=1, dtype=float, shape=[self.n]
        )
        z_init = jnp.exp(1j * jnp.pi * phase)

        if self.kuramoto:
            z_init = jnp.angle(z_init)
        
        sol = odeint(self.f, z_init, self.times, rtol=1.4e-8, atol=1.4e-8)

        z = sol

        return z


@jit
def quantize(z):
    return jnp.where(np.real(z) > 0, jnp.ones_like(z), -jnp.ones_like(z))
