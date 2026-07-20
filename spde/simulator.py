import numpy as np
from .tamed_ns import TamedNS


def run_simulation(
    initial_state=None,
    T=80,
    dim=100
):
    """
    Run SPDE evolution.

    Parameters
    ----------
    initial_state : ndarray or None
        Initial market state.
        If None, use Gaussian random state.

    T : int
        Evolution steps.

    dim : int
        State dimension.
    """

    model = TamedNS()

    if initial_state is None:
        u = np.random.randn(dim)
    else:
        u = np.asarray(initial_state, dtype=float).copy()

    trajectory = [u.copy()]

    for _ in range(T):

        u = model.step(u)

        trajectory.append(u.copy())

    return np.asarray(trajectory)