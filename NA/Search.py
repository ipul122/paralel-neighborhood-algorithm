import numpy as NP
from config import seed

def dim_models(lower_bounds, upper_bounds, models):
    return lower_bounds + (upper_bounds - lower_bounds) * models
        
random_generator = NP.random.RandomState(seed)           #############
random_sampling = NP.random.RandomState()           #############
def generate_random_models(n,nd):
    return random_generator.random_sample((n, nd))
        
def get_bests_models(nr,misfits,models,np):
    best_models_ids = NP.argsort(misfits[:np])[:nr]
    return models[best_models_ids]

def sampling(ns,nd,nr,models,np,misfits):
    new_models = NP.zeros((ns, nd))
    bests_so_far = get_bests_models(nr,misfits,models,np)
    m = models[:np, :]
    idx = 0
    
    for k, vk in enumerate(bests_so_far):
        walk_length = int(NP.floor(ns / nr))

        if k == 0:
            walk_length += int(NP.floor(ns % nr))
    
        d2 = NP.sum((m - vk) ** 2, axis=1)                  # model - best model
        d2_prev_axis = 0.                                   # 0 for first axis
        for step in range(walk_length):                     # walk around best model
            xA = vk.copy()                                  # start from the best model            
            axes = random_generator.permutation(nd)         # random order of axes (random parameters)
            for id_ax, i in enumerate(axes):                     # loop over axes
                d2_current_axis = (m[:, i] - xA[i]) ** 2        # distance squared along current axis
                d2 += d2_prev_axis - d2_current_axis      # update distance squared
                dk2 = d2[k]                          # distance squared to the best model
                
                vji = m[:, i]                     # models along current axis
                a = (dk2 - d2)                   # numerator
                b = (vk[i] - vji)                # denominator
                
                xji = 0.5 *(vk[i] + vji + NP.divide(a, b, out=NP.zeros_like(a), where=b!=0))  # new positions along current axis

                li = NP.nanmax(                                 # lower bound
                            NP.hstack((0, xji[xji < xA[i]])))
                ui = NP.nanmin(                                 # upper bound    
                            NP.hstack((1, xji[xji > xA[i]])))
                    
                xA[i] = (ui - li) * random_sampling.random_sample() + li  # new model along current axis within bounds
                d2_prev_axis = d2_current_axis              # update previous distance squared
                    
            new_models[idx] = xA.copy()                     # store new model
            idx += 1            
                    
    return new_models         