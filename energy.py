import numpy
from jax import jit, numpy as jnp


@jit
def config_energy(variables: numpy.ndarray,
                      configurations: numpy.ndarray):
    """
    Example: XOR
    config = [[1, 0, -1], [-1, 1, 0]]
    zeros = [[0, 1, 0], [0, 0, 1]]
    variables = [-1+0j, 1+0j, 1+0j]
    config_energy = 1 - 1/3 * prod((1 + config * variables) / 2)

    :param variables: complex / bipolar variables
    :param configurations: Spin configurations
    :return:
    """
    # variables = variables / (jnp.abs(variables))
    energy = - ((1 + (2*configurations-1) * variables) / 2).prod(-1)
    return energy


# @jit
def kuramoto_config_energy(phi: numpy.ndarray,
                           configurations: numpy.ndarray):
    """
    Example: XOR
    config = [[1, 0, -1], [-1, 1, 0]]
    zeros = [[0, 1, 0], [0, 0, 1]]
    variables = [-1+0j, 1+0j, 1+0j]
    config_energy = 1 - 1/3 * prod((1 + config * variables) / 2)

    :param phi: complex / bipolar variables
    :param configurations: Spin configurations
    :return:
    """

    energy = -((jnp.cos((jnp.pi*(1-configurations) - phi)) + 1) / 2).prod(-1)

    return energy

@jit
def ising_energy(z, linear, quadratic, offset):
    # z samples x dim
    z = z / (jnp.abs(z))
    return z.conj() @ quadratic @ z + z @ linear + offset
