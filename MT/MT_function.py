<<<<<<< HEAD
import numpy as NP
=======
"""import numpy as NP
>>>>>>> e362550 (Final push all files)
#from numba import jit

#@jit(nopython=True, fastmath=True)

def forward(model, frequencies, Obsres, Obsphs):

    mu = 4 * NP.pi * 1e-7
    
    # Pastikan model adalah array 2D
    model = NP.atleast_2d(model)
    n_models, n_params = model.shape
    
    # Hitung jumlah layer dari jumlah parameter
    # Format: [rho1, rho2, ..., rhoN, h1, h2, ..., h(N-1)]
    # Total param = N + (N-1) = 2N - 1
    # Maka N = (n_params + 1) // 2
    n_layer = (n_params + 1) // 2
    
    # Inisialisasi array untuk hasil
    misfits = NP.zeros(n_models)
    all_apparent_res = NP.zeros((n_models, len(frequencies)))
    all_phases = NP.zeros((n_models, len(frequencies)))

    
    # Loop untuk setiap model
    for i in range(n_models):
        # Ekstrak resistivitas dan ketebalan
        resistivities = model[i, :n_layer]
        thicknesses = model[i, n_layer:]
        
        apparent_resistivities = []
        phases = []
        
        # Loop untuk setiap frekuensi
        for frequency in frequencies:
            w = 2 * NP.pi * frequency
            Z = [0] * n_layer
            
            # Basement impedance
            Z[-1] = NP.sqrt(1j * w * mu * resistivities[-1])
            
            # Upward recursion
            for j in range(n_layer - 2, -1, -1):
                rho = resistivities[j]
                h = thicknesses[j]
                
                dj = NP.sqrt(1j * w * mu / rho)
                wj = rho * dj
                ej = NP.exp(-2 * dj * h)
                
                rj = (wj - Z[j+1]) / (wj + Z[j+1])
                Z[j] = wj * (1 - rj * ej) / (1 + rj * ej)
            
            # Apparent resistivity & phase
            Z0 = Z[0]
            rhoa = (abs(Z0)**2) / (mu * w)
            phase = NP.atan2(Z0.imag, Z0.real)
            
            apparent_resistivities.append(rhoa)
            phases.append(phase)
        
        apparent_resistivities = NP.array(apparent_resistivities)
        phases = NP.rad2deg(NP.array(phases))

        all_apparent_res[i, :] = apparent_resistivities  # Baris i untuk model i
        all_phases[i, :] = phases  # Baris i untuk model i

        
        misfit = NP.sqrt(
            NP.sum(
                (NP.log10(Obsres / apparent_resistivities))**2 +
                ((NP.deg2rad(Obsphs)) - (NP.deg2rad(phases)))**2
            )/len(frequencies)
        )
        misfits[i] = misfit*100

        
    return misfits,all_apparent_res,all_phases

<<<<<<< HEAD
=======
"""

import numpy as np
from numba import njit, prange

@njit(parallel=True, fastmath=True)
def forward(model, frequencies, Obsres, Obsphs):

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

            Z0 = Z[0]
            rhoa = (np.abs(Z0)**2) / (mu * w)
            phase = np.atan2(Z0.imag, Z0.real)

            all_apparent_res[i, k] = rhoa
            all_phases[i, k] = phase * 180.0 / np.pi

        # Misfit
        s = 0.0
        for k in range(n_freq):
            dr = np.log10(Obsres[k] / all_apparent_res[i, k])
            dp = (Obsphs[k] - all_phases[i, k]) * np.pi / 180.0
            s += dr*dr + dp*dp

        misfits[i] = np.sqrt(s / n_freq) * 100.0

    return misfits, all_apparent_res, all_phases
>>>>>>> e362550 (Final push all files)
