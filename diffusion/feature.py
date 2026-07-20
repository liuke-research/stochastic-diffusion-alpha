import numpy as np

from .diffusion_model import Diffusion



class FeatureBuilder:


    def __init__(self):

        self.diffusion = Diffusion()



    def build_signal(self, x0, xT):

        """
        Build SPDE alpha features.

        Parameters
        ----------
        x0:
            Initial market state

        xT:
            Evolved SPDE state


        Returns
        -------
        alpha score
        """

        # diffusion smooth state

        smooth = self.diffusion.transform(x0)



        # residual after diffusion

        residual = xT - smooth



        # =========================
        # 1. Diffusion energy
        # =========================

        energy = np.mean(
            residual ** 2
        )



        # =========================
        # 2. State transition magnitude
        # =========================

        state_shift = np.linalg.norm(
            xT - x0
        )



        # =========================
        # 3. Market roughness
        # =========================

        roughness = np.mean(
            np.abs(
                np.gradient(xT)
            )
        )



        # =========================
        # SPDE Alpha Score
        # =========================

        alpha = (
            0.5 * energy
            +
            0.3 * state_shift
            +
            0.2 * roughness
        )



        return alpha