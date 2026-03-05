import numpy as NP
from config import seed

random_generator = NP.random.RandomState(seed)
random_sampling = NP.random.RandomState()

def dim_models(lower_bounds, upper_bounds, models):
    return lower_bounds + (upper_bounds - lower_bounds) * models

def generate_random_models(n, nd):
    return random_generator.random_sample((n, nd))

def get_bests_indices(nr, misfits, np_current):
    """Mengambil indeks asli dari model-model terbaik."""
    return NP.argsort(misfits[:np_current])[:nr]

def sampling(ns, nd, nr, models, np_current, misfits):
    new_models = NP.zeros((ns, nd))
    
    best_indices = get_bests_indices(nr, misfits, np_current)
    m = models[:np_current, :]
    
    global_best_idx = best_indices[0]
    
    idx_new = 0
    for k in best_indices:

        walk_length = int(NP.floor(ns / nr))
        if k == global_best_idx:
            walk_length += int(ns % nr)
            
        vk = m[k].copy()
        
        for step in range(walk_length):
            xA = vk.copy()             
            d2 = NP.sum((m - xA) ** 2, axis=1)            
            axes = random_generator.permutation(nd)
            
            for i in axes:
                dk2 = d2[k]                
                vji = m[:, i]
                a = (dk2 - d2)
                b = (vk[i] - vji)                
                xji = 0.5 * (vk[i] + vji + NP.divide(a, b, out=NP.zeros_like(a), where=b != 0))

                li = NP.nanmax(NP.hstack((0, xji[xji < xA[i]])))
                ui = NP.nanmin(NP.hstack((1, xji[xji > xA[i]])))
                
                xA_old_i = xA[i]
                xA[i] = (ui - li) * random_sampling.random_sample() + li
                
                d2 = d2 - (m[:, i] - xA_old_i)**2 + (m[:, i] - xA[i])**2
            
            if idx_new < ns:
                new_models[idx_new] = xA.copy()
                idx_new += 1
                
    return new_models