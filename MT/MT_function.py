import numpy as np
from numba import njit, prange

@njit(parallel=True, fastmath=True)
def forward(model, frequencies, Obsres, Obsphs,noise_level=0):
    noise_level = noise_level / 100.0 
    mu = 4 * np.pi * 1e-7

    n_models = model.shape[0]
    n_params = model.shape[1]
    n_freq = frequencies.size

    n_layer = (n_params + 1) // 2

    misfits = np.zeros(n_models)
    all_apparent_res = np.zeros((n_models, n_freq))
    all_phases = np.zeros((n_models, n_freq))

    for i in prange(n_models):

        resistivities = model[i, :n_layer]
        thicknesses = model[i, n_layer:]

        Z = np.zeros(n_layer, dtype=np.complex128)

        for k in range(n_freq):
            w = 2 * np.pi * frequencies[k]

            # Basement
            Z[-1] = np.sqrt(1j * w * mu * resistivities[-1])

            # Upward recursion
            for j in range(n_layer - 2, -1, -1):
                rho = resistivities[j]
                h = thicknesses[j]

                dj = np.sqrt(1j * w * mu / rho)
                wj = rho * dj
                ej = np.exp(-2 * dj * h)

                rj = (wj - Z[j+1]) / (wj + Z[j+1])
                Z[j] = wj * (1 - rj * ej) / (1 + rj * ej)

            Z0_pure = Z[0]
            
            zre_noisy = Z0_pure.real + (Z0_pure.real * noise_level * np.random.randn())
            zim_noisy = Z0_pure.imag + (Z0_pure.imag * noise_level * np.random.randn())
            
            rhoa = ( (zre_noisy**2 + zim_noisy**2) ) / (mu * w)
            phase = np.atan2(zim_noisy, zre_noisy)

            all_apparent_res[i, k] = rhoa
            all_phases[i, k] = phase * 180.0 / np.pi

        # Misfit
        s = 0.0
        for k in range(n_freq):
            dr = np.log10(Obsres[k] / all_apparent_res[i, k])
            dp = (Obsphs[k] - all_phases[i, k]) * np.pi / 180.0
            s += np.abs(dr) + np.abs(dp)

        misfits[i] = s

    return misfits, all_apparent_res, all_phases
