import numpy as np


def build_market_state(
    returns,
    window=100
):
    """
    Build rolling market states.

    Each state is a rolling window
    of historical returns.
    """

    states = []

    for i in range(window, len(returns)):

        state = returns[i-window:i]

        state = (
            state
            - np.mean(state)
        ) / (
            np.std(state)+1e-8
        )

        states.append(state)

    return np.asarray(states)