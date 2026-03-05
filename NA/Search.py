import numpy as np
from numba import njit, prange
from config import seed


# Inisialisasi generator acak
random_generator = np.random.RandomState(seed)

def dim_models(lower_bounds, upper_bounds, models):
    return lower_bounds + (upper_bounds - lower_bounds) * models

def generate_random_models(n, nd):
    return random_generator.random_sample((n, nd))

@njit(parallel=True, fastmath=True)
def sampling_jit(ns, nd, nr, models, np_current, misfits):
    new_models = np.zeros((ns, nd))
    
    # 1. Ambil indeks model terbaik (sorting di Numba)
    m = models[:np_current]
    best_indices = np.argsort(misfits[:np_current])[:nr]
    
    # Pre-calculating walk_length untuk tiap best_indices
    # Agar bisa dijalankan dalam prange (paralel)
    walk_lengths = np.full(nr, ns // nr, dtype=np.int32)
    walk_lengths[0] += ns % nr # Sisa pembagian diberikan ke peringkat 1
    
    # Tentukan offset indeks untuk menyimpan hasil di new_models
    offsets = np.zeros(nr, dtype=np.int32)
    for i in range(1, nr):
        offsets[i] = offsets[i-1] + walk_lengths[i-1]

    # 2. Parallel loop untuk setiap sel model terbaik
    for r in prange(nr):
        k = best_indices[r]
        vk = m[k]
        current_walk_len = walk_lengths[r]
        current_offset = offsets[r]
        
        for step in range(current_walk_len):
            xA = vk.copy()
            
            # Hitung d2 awal: jarak xA ke semua model m
            d2 = np.zeros(np_current)
            for j in range(np_current):
                dist_sq = 0.0
                for d in range(nd):
                    diff = m[j, d] - xA[d]
                    dist_sq += diff * diff
                d2[j] = dist_sq
            
            # Acak urutan sumbu
            axes = np.random.permutation(nd)
            
            for i in axes:
                # dk2 adalah jarak model referensi k terhadap sumbu i
                dk2 = d2[k]
                
                li = 0.0
                ui = 1.0
                
                # Cari batas Voronoi (Persamaan 9, 10, 11)
                for j in range(np_current):
                    denominator = vk[i] - m[j, i]
                    
                    if denominator != 0.0:
                        numerator = dk2 - d2[j]
                        xji = 0.5 * (vk[i] + m[j, i] + (numerator / denominator))
                        
                        # Update li dan ui
                        if xji < xA[i]:
                            if xji > li: li = xji
                        elif xji > xA[i]:
                            if xji < ui: ui = xji
                
                # Ambil posisi baru
                xA_old_i = xA[i]
                xA[i] = li + (ui - li) * np.random.random()
                
                # 5. Update d2 secara rekursif (Persamaan 12)
                # Linear cost O(np_current)
                for j in range(np_current):
                    old_diff = m[j, i] - xA_old_i
                    new_diff = m[j, i] - xA[i]
                    d2[j] = d2[j] - (old_diff * old_diff) + (new_diff * new_diff)
            
            # Simpan hasil ke array output
            new_models[current_offset + step] = xA
            
    return new_models