from numba import njit, prange
import numpy as np

@njit(parallel=True, fastmath=True)
def sampling_jit(ns, nd, nr, models, npop, misfits):
    new_models = np.zeros((ns, nd))
    m = models[:npop]

    best_ids = np.argsort(misfits[:npop])[:nr]
    walk_length = ns // nr

    for k in prange(nr):
        vk = m[best_ids[k]]
        start = k * walk_length
        end = start + walk_length

        d2 = np.zeros(npop)
        for j in range(npop):
            for d in range(nd):
                diff = m[j, d] - vk[d]
                d2[j] += diff * diff

        for i in range(start, end):
            xA = vk.copy()
            axes = np.random.permutation(nd)

            for ax in axes:
                dk2 = d2[best_ids[k]]
                li = 0.0
                ui = 1.0

                for j in range(npop):
                    b = vk[ax] - m[j, ax]
                    if b != 0.0:
                        a = dk2 - d2[j]
                        xji = 0.5 * (vk[ax] + m[j, ax] + a / b)
                        if xji < xA[ax] and xji > li:
                            li = xji
                        if xji > xA[ax] and xji < ui:
                            ui = xji

                xA[ax] = (ui - li) * np.random.random() + li

            new_models[i] = xA

    return new_models
