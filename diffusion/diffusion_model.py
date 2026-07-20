import numpy as np

class Diffusion:
    def transform(self, x, steps=5):
        x = x.copy()
        for _ in range(steps):
            x = (x + np.roll(x,1) + np.roll(x,-1)) / 3
        return x