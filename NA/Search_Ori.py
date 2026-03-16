import numpy as NP
from config import seed
random_generator = NP.random.RandomState(seed)
random_sampling = NP.random.RandomState()

def dim_models(lower_bounds, upper_bounds, models):
    return lower_bounds + (upper_bounds - lower_bounds) * models

def generate_random_models(n, nd):
    return random_generator.random_sample((n, nd))

def sampling(ns, nd, nr, models, np_current, misfits):
    new_models = NP.zeros((ns, nd))
    
    best_indices = NP.argsort(misfits[:np_current])[:nr]
    m = models[:np_current, :]
    
    idx_new = 0
    for k in best_indices:
        walk_length = int(NP.floor(ns / nr))
        if k == best_indices[0]: 
            walk_length += int(ns % nr)
            
        vk = m[k].copy() 
       
        xA = vk.copy() 
        
        
        d2_full = NP.sum((m - xA) ** 2, axis=1)
        
        for step in range(walk_length):
            axes = random_generator.permutation(nd)
            
            for i in axes:
                d2_curr_axis = (m[:, i] - xA[i])**2
                d2_perp = d2_full - d2_curr_axis
                dk2_perp = d2_perp[k]
                
                vji = m[:, i]
                vki = vk[i]
                
                a = (dk2_perp - d2_perp)
                b = (vki - vji)
                xji = 0.5 * (vki + vji + NP.divide(a, b, out=NP.zeros_like(a), where=b != 0))

                li = NP.nanmax(NP.hstack((0.0, xji[xji < xA[i]])))
                ui = NP.nanmin(NP.hstack((1.0, xji[xji > xA[i]])))
                
                xA_old_i = xA[i]
                xA[i] = li + (ui - li) * random_sampling.random_sample()
                
                d2_full = d2_full - (m[:, i] - xA_old_i)**2 + (m[:, i] - xA[i])**2
            
            if idx_new < ns:
                new_models[idx_new] = xA.copy()
                idx_new += 1
                
    return new_models