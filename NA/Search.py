import numpy as np
from numba import njit, prange
from config import seed


random_generator = np.random.RandomState(seed)

def dim_models(lower_bounds, upper_bounds, models):
    return lower_bounds + (upper_bounds - lower_bounds) * models

def generate_random_models(n, nd):
    return random_generator.random_sample((n, nd))


@njit(parallel=True, fastmath=True)
def sampling_jit(ns, nd, nr, models, np_current, misfits):
    new_models = np.zeros((ns, nd))
    
    m = models[:np_current]
    best_indices = np.argsort(misfits[:np_current])[:nr]
    
    walk_lengths = np.full(nr, ns // nr, dtype=np.int32)
    walk_lengths[0] += ns % nr 
    
    offsets = np.zeros(nr, dtype=np.int32)
    for i in range(1, nr):
        offsets[i] = offsets[i-1] + walk_lengths[i-1]

    for r in prange(nr):
        k = best_indices[r]
        vk = m[k] 
        current_walk_len = walk_lengths[r]
        current_offset = offsets[r]
        
        xA = vk.copy() 
        
        
        d2_full = np.zeros(np_current)
        for j in range(np_current):
            dist_sq = 0.0
            for d in range(nd):
                diff = m[j, d] - xA[d]
                dist_sq += diff * diff
            d2_full[j] = dist_sq
        
        for step in range(current_walk_len):

            for i in range(nd):
                vki = vk[i]
                vji = m[:, i]

                # ---  (Eq 18 Paper) ---
                d2_perp = np.zeros(np_current)
                for j in range(np_current):
                    d2_perp[j] = d2_full[j] - (m[j, i] - xA[i])**2
                
                dk2_perp = d2_perp[k] 
                
                li = 0.0
                ui = 1.0
                
                for j in range(np_current):
                    denominator = vki - vji[j]
                    
                    if denominator != 0.0:
                        numerator = dk2_perp - d2_perp[j]
                        # ---  (Eq 19 Paper) ---
                        xji = 0.5 * (vki + vji[j] + (numerator / denominator))
                        
                        # ---  (Eq 20 & 21 Paper) ---
                        if xji < xA[i]:
                            if xji > li: li = xji
                        elif xji > xA[i]:
                            if xji < ui: ui = xji
                
                xA_old_i = xA[i]
                xA[i] = li + (ui - li) * np.random.random()
                
                # --- RECURSIVE DISTANCE UPDATE 
                for j in range(np_current):
                    old_diff = m[j, i] - xA_old_i
                    new_diff = m[j, i] - xA[i]
                    d2_full[j] = d2_full[j] - (old_diff**2) + (new_diff**2)
            
            new_models[current_offset + step] = xA.copy()
            
    return new_models