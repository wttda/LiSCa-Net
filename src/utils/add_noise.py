import numpy as np
from .util import normalize_img
from easydict import EasyDict as edict


class NoiseSimulator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.scene_name, self.n_case = cfg.datasets.scene_name, cfg.noise.case
        self.noise_cfg = cfg.noise.get(self.n_case).get(self.scene_name)
        if not self.noise_cfg:
            raise ValueError(f"No configuration found for {self.n_case}")

        # Parse noise types
        self.n_types = [nt.strip() for nt in cfg.noise.get(self.n_case).noise_type.split('+')]

        # Initialize noise parameters
        self.int_rng = self.noise_cfg.int_rng
        self.impulse_params = self._impulse_params()
        self.stripe_params = self._stripe_params()
        self.deadline_params = self._deadline_params()
        self.bd = 0

    def _impulse_params(self):
        """Get impulse noise configuration.
        Returns dict with:
            bd_ratio: ratio of bands to which impulse noise is added
            amounts: list of impulse noise amount values to select from for each band
            s_vs_p: salt vs pepper ratio
        """
        if 'impulse' not in self.n_types:
            return edict()
        defaults = edict({'bd_ratio': 0.3, 'amounts': [0.1, 0.3, 0.5, 0.7], 's_vs_p': 0.5})
        params = edict()
        for key, default in defaults.items():
            params[key] = self.noise_cfg.impulse.get(key, default)
        return params

    def _stripe_params(self):
        """Get stripe noise configuration.
        Returns dict with:
            bd_ratio: ratio of bands to which stripe noise is added
            stripe_type: type of stripe noise, either 'vertical' or 'diagonal'
            vertical_ratio_rng: range of stripe ratio values for vertical stripes (used when stripe_type is 'vertical')
            diagonal_num_rng: range of stripe count values for diagonal stripes (used when stripe_type is 'diagonal')
            stripe_int: stripe intensity ranging from 0 to 1
        """
        if 'stripe' not in self.n_types:
            return edict()
        defaults = edict({'bd_ratio': 0.3,
                          'stripe_type': 'vertical',
                          'vertical_ratio_rng': [ 0.05, 0.15 ],
                          'diagonal_num_rng': [ 5, 10 ],
                          'stripe_int': 0.5})
        params = edict()
        for key, default in defaults.items():
            params[key] = self.noise_cfg.stripe.get(key, default)
        return params

    def _deadline_params(self):
        """Get deadline noise configuration.
        Returns dict with:
            bd_ratio: ratio of bands to which deadline noise is added
            deadline_type: type of deadline noise, either 'vertical' or 'diagonal'
            vertical_ratio_rng: range of deadline ratio values for vertical deadlines
            diagonal_num_rng: range of deadline count values for diagonal deadlines
            wider_deadline_width: width of the wider deadline lines
        """
        if 'deadline' not in self.n_types:
            return edict()
        defaults = edict({'bd_ratio': 0.3,
                          'deadline_type': 'diagonal',
                          'vertical_ratio_rng': [ 0.02, 0.1 ],
                          'diagonal_num_rng': [ 3, 5 ],
                          'wider_deadline_width': 8})
        params = edict()
        for key, default in defaults.items():
            params[key] = self.noise_cfg.deadline.get(key, default)
        return params

    def simulate(self, c_hsi):
        """Simulate mixed noise on a HSI.

        Parameters:
            c_hsi (np.ndarray): Clean HSI with shape (row, column, band).

        Returns (tuple): (clean HSI, noisy HSI).
        """
        return self._add_noise(c_hsi)

    def _add_noise(self, c_hsi):
        # Normalize image
        c_hsi = np.array([normalize_img(c_hsi[:, :, i]) for i in range(c_hsi.shape[-1])])
        c_hsi = c_hsi.transpose(1, 2, 0)
        n_hsi = c_hsi.copy()
        self.rows, self.cols, self.nbd = c_hsi.shape
        self.npx = self.rows * self.cols

        # Add non-IID Gaussian noise
        if 'non_iid_gaussian' in self.n_types:
            sigmas = self._generate_sigma()
            n_hsi += np.random.randn(*n_hsi.shape) * sigmas[np.newaxis, np.newaxis, :]

        # Add stripe noise
        if 'stripe' in self.n_types:
            n_hsi = self._add_stripe_noise(n_hsi)

        # Add deadline noise
        if 'deadline' in self.n_types:
            n_hsi = self._add_deadline_noise(n_hsi)

        # Add impulse noise
        if 'impulse' in self.n_types:
            n_hsi = self._add_impulse_noise(n_hsi)

        return c_hsi, n_hsi

    def _generate_sigma(self):
        """Generate random noise standard deviations (sigma) for each band."""
        min_sigma, max_sigma = self.int_rng
        assert min_sigma < max_sigma
        return min_sigma + (max_sigma - min_sigma) * np.random.rand(self.nbd)

    def _add_stripe_noise(self, hsi):
        """Add stripe noise to randomly selected bands."""
        params = self.stripe_params

        stripe_bd_num = int(round(self.nbd * params.bd_ratio))
        if params.stripe_type == 'vertical':
            assert params.vertical_ratio_rng[0] < params.vertical_ratio_rng[1]
            stripe_counts = np.random.randint(
                np.floor(params.vertical_ratio_rng[0] * self.cols),
                np.floor(params.vertical_ratio_rng[1] * self.cols),
                size=stripe_bd_num
            )
            for i, n in zip(np.random.permutation(self.nbd)[:stripe_bd_num], stripe_counts):
                # Add vertical stripes
                loc = np.random.permutation(range(self.cols))
                loc = loc[:n]
                stripe = np.random.uniform(0, 1, size=n) * 2 * params.stripe_int - params.stripe_int
                hsi[:, loc, i] -= stripe[np.newaxis, :]
        elif params.stripe_type == 'diagonal':
            assert params.diagonal_num_rng[0] < params.diagonal_num_rng[1]
            stripe_counts = np.random.randint(
                params.diagonal_num_rng[0], params.diagonal_num_rng[1], size=stripe_bd_num)
            for idx, self.bd in enumerate(np.random.permutation(self.nbd)[:stripe_bd_num]):
                # Add diagonal stripes
                for _ in range(stripe_counts[idx]):
                    self._apply_diag_stripe(hsi, params.stripe_int, width=1)
        else:
            raise ValueError(f"Unknown stripe type: {params.stripe_type}")

        return hsi

    def _apply_diag_stripe(self, hsi, intensity, width):
        """Apply diagonal stripe noise in random direction."""
        stripe = np.random.uniform(0, 1) * 2 * intensity - intensity
        if np.random.rand() > 0.5:
            # Add stripe diagonally from top-left to bottom-right
            offset = 0
            for r in range(np.random.randint(self.rows), self.rows-1):
                offset += 1
                c = offset
                if c >= self.cols:
                    break
                hsi[r, c:min(c+width, self.cols), self.bd] -= stripe
        else:
            # Add stripe diagonally from top-right to bottom-left
            offset = 0
            for c in range(np.random.randint(self.cols), self.cols-1):
                offset += 1
                r = offset
                if r >= self.rows:
                    break
                hsi[r:min(r+width, self.rows), c, self.bd] -= stripe

    def _add_deadline_noise(self, hsi):
        """Add deadline noise to randomly selected bands."""
        params = self.deadline_params

        deadline_bd_num = int(round(self.nbd * params.bd_ratio))
        if params.deadline_type == 'vertical':
            assert params.vertical_ratio_rng[0] < params.vertical_ratio_rng[1]
            deadline_counts = np.random.randint(
                np.floor(params.vertical_ratio_rng[0] * self.cols),
                np.floor(params.vertical_ratio_rng[1] * self.cols),
                size=deadline_bd_num
            )
            for i, n in zip(np.random.permutation(self.nbd)[:deadline_bd_num], deadline_counts):
                # Add vertical deadlines
                loc = np.random.permutation(range(self.cols))
                hsi[:, loc[:n], i] = 1
        elif params.deadline_type == 'diagonal':
            assert params.diagonal_num_rng[0] < params.diagonal_num_rng[1]
            deadline_counts = np.random.randint(
                params.diagonal_num_rng[0], params.diagonal_num_rng[1], size=deadline_bd_num)
            for idx, self.bd in enumerate(np.random.permutation(self.nbd)[:deadline_bd_num]):
                # Add diagonal stripes
                for _ in range(deadline_counts[idx]):
                    self._apply_diag_deadline(hsi, width=1)
                self._apply_diag_deadline(hsi, width=params.wider_deadline_width)
        else:
            raise ValueError(f"Unknown stripe type: {params.stripe_type}")

        return hsi

    def _apply_diag_deadline(self, hsi, width):
        """Apply diagonal deadline in random direction."""
        if np.random.rand() > 0.5:
            # Add deadline diagonally from top-left to bottom-right
            offset = 0
            for r in range(np.random.randint(self.rows), self.rows-1):
                offset += 1
                c = offset
                if c >= self.cols:
                    break
                hsi[r, c:min(c+width, self.cols), self.bd] = 1
        else:
            # Add stripe deadline from top-right to bottom-left
            offset = 0
            for c in range(np.random.randint(self.cols), self.cols-1):
                offset += 1
                r = offset
                if r >= self.rows:
                    break
                hsi[r:min(r+width, self.rows), c, self.bd] = 1

    def _add_impulse_noise(self, hsi):
        """Add impulse noise to randomly selected bands."""
        params = self.impulse_params
        impulse_bd_num = int(round(self.nbd * params.bd_ratio))
        bands = np.random.permutation(self.nbd)[:impulse_bd_num]
        amounts = np.array(params.amounts)
        for i, amount in zip(bands, amounts[np.random.randint(0, len(amounts), len(bands))]):
            band = hsi[:, :, i]
            flipped = np.random.choice([True, False], size=band.shape, p=[amount, 1 - amount])
            salted = np.random.choice([True, False], size=band.shape, p=[params.s_vs_p, 1 - params.s_vs_p])
            band[flipped & salted] = 1.0
            band[flipped & ~salted] = 0.0
            hsi[:, :, i] = band
        return hsi
