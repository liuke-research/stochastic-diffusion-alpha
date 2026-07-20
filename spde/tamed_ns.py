import numpy as np


class TamedNS:

    def __init__(self, dt=0.002, nu=0.1):

        self.dt = dt
        self.nu = nu


    def laplacian(self, u):

        return (
            np.roll(u, 1)
            +
            np.roll(u, -1)
            -
            2*u
        )


    def step(self, u):

        diffusion = self.nu * self.laplacian(u)

        grad = np.gradient(u)

        nonlinear = u * grad


        # Tamed nonlinear term
        nonlinear = nonlinear / (
            1 + self.dt * np.abs(nonlinear)
        )


        # Energy damping
        energy = np.mean(u ** 2)

        damping = 0.05 * u * energy


        u_new = u + self.dt * (
            diffusion
            -
            nonlinear
            -
            damping
        )


        # Projection stabilization
        norm = np.linalg.norm(u_new)

        if norm > 100:

            u_new = (
                u_new / norm * 100
            )


        return u_new