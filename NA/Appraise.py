from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from typing import Tuple
from numba import njit, prange
from tqdm.auto import tqdm

from ._mcintegrals import MCIntegrals


@njit(parallel=True, fastmath=True)
def numba_compute_distances(xA: np.ndarray, initial_ensemble: np.ndarray, Cm: np.ndarray, axis: int):
    Ne = initial_ensemble.shape[0]
    Nd = initial_ensemble.shape[1]
    
    d2 = np.zeros(Ne)
    dk2 = np.zeros(Ne)
    
    for j in prange(Ne):
        dist_total = 0.0
        dist_perp = 0.0
        for i in range(Nd):
            diff_sq = (xA[i] - initial_ensemble[j, i]) ** 2
            scaled_dist = diff_sq * Cm[i]
            dist_total += scaled_dist
            if i != axis:
                dist_perp += scaled_dist
        d2[j] = dist_total
        dk2[j] = dist_perp
        
    k = np.argmin(d2)
    return k, dk2


@njit(fastmath=True)
def numba_get_axis_intersections(axis: int, k: int, di2: np.ndarray, initial_ensemble: np.ndarray, Cm: np.ndarray, down: bool):
    intersections = []
    cells = []
    Ne = initial_ensemble.shape[0]
    
    current_k = k
    while True:
        vki = initial_ensemble[current_k, axis]
        best_xji = np.inf if down else -np.inf
        k_new = -1
        
        for j in range(Ne):
            if j == current_k:
                continue
                
            vji = initial_ensemble[j, axis]
            b = Cm[axis] * (vki - vji)
            
            if b == 0.0:
                continue
                
            a = di2[current_k] - di2[j]
            xji = 0.5 * (vki + vji + a / b)
            
            if down:
                if vki > vji:
                    if k_new == -1 or xji > best_xji:
                        best_xji = xji
                        k_new = j
            else:
                if vki < vji:
                    if k_new == -1 or xji < best_xji:
                        best_xji = xji
                        k_new = j
        
        if k_new != -1:
            intersections.append(best_xji)
            cells.append(k_new)
            current_k = k_new
        else:
            break
            
    return intersections, cells

@njit(fastmath=True)
def numba_identify_cell(xp: float, intersections: np.ndarray, cells: np.ndarray) -> int:
    closest_intersection = np.argmin(np.abs(intersections - xp))
    if xp < intersections[closest_intersection]:
        cell_id = closest_intersection
    else:
        cell_id = closest_intersection + 1
    return cells[cell_id]

@njit(fastmath=True)
def numba_random_step(axis: int, intersections: np.ndarray, cells: np.ndarray, lower: np.ndarray, upper: np.ndarray, log_ppd: np.ndarray):
    while True:
        xpi = np.random.uniform(lower[axis], upper[axis])
        k = numba_identify_cell(xpi, intersections, cells)

        r = np.random.uniform(0.0, 1.0)
        logPxpi = log_ppd[k]
        
        max_val = -np.inf
        for c in cells:
            if log_ppd[c] > max_val:
                max_val = log_ppd[c]
                
        if np.log(r) < logPxpi - max_val:
            return xpi

@njit(fastmath=True)
def numba_execute_walk_batch(batch_size: int, nd: int, xA: np.ndarray, initial_ensemble: np.ndarray, Cm: np.ndarray, lower: np.ndarray, upper: np.ndarray, log_ppd: np.ndarray):
    samples_out = np.zeros((batch_size, nd))
    
    for s in range(batch_size):
        for i in range(nd):
            k, dk2 = numba_compute_distances(xA, initial_ensemble, Cm, i)
            
            down_i, down_c = numba_get_axis_intersections(i, k, dk2, initial_ensemble, Cm, True)
            up_i, up_c = numba_get_axis_intersections(i, k, dk2, initial_ensemble, Cm, False)
            
            n_down = len(down_i)
            n_up = len(up_i)
            total_elements = n_down + n_up
            
            intersections_arr = np.zeros(total_elements)
            cells_arr = np.zeros(total_elements + 1, dtype=np.int64)
            
            for idx in range(n_down):
                intersections_arr[idx] = down_i[n_down - 1 - idx]
                cells_arr[idx] = down_c[n_down - 1 - idx]
                
            cells_arr[n_down] = k
            
            for idx in range(n_up):
                intersections_arr[n_down + idx] = up_i[idx]
                cells_arr[n_down + 1 + idx] = up_c[idx]
                
            xpi = numba_random_step(i, intersections_arr, cells_arr, lower, upper, log_ppd)
            xA[i] = xpi
            
        samples_out[s] = xA.copy()
        
    return samples_out, xA

class NAAppraiser:
    def __init__(
        self,
        n_resample: int,
        n_walkers: int = 1,
        initial_ensemble: NDArray | None = None,
        log_ppd: NDArray | None = None,
        bounds: Tuple[Tuple[float, float], ...] | None = None,
        verbose: bool = True,
        seed: int | None = None,
    ):
        self.initial_ensemble = initial_ensemble
        self.log_ppd = log_ppd
        self.bounds = bounds
        self.nd = len(bounds) 
        self.lower = np.array([b[0] for b in bounds])
        self.upper = np.array([b[1] for b in bounds])
        self.Cm = 1 / (self.upper - self.lower) ** 2
        self.verbose = verbose
        self.Ne = len(initial_ensemble)
        self.nr = n_resample

        if seed is not None:
            np.random.seed(seed)

    def run(self, save: bool = True, start_fraction: float = 0.5, callback=None) -> None:
        start_cell = np.argmax(self.log_ppd)
        accumulator = MCIntegrals(self.nd, save)
        
        xA = self.initial_ensemble[start_cell].copy()
        
        batch_size = 10 
        total_samples_collected = 0
                  
        with tqdm(total=self.nr, desc="Resampling", disable=not self.verbose) as pbar:
            while total_samples_collected < self.nr:
                current_batch = min(batch_size, self.nr - total_samples_collected)
                
                batch_samples, xA = numba_execute_walk_batch(
                    current_batch, self.nd, xA, self.initial_ensemble, 
                    self.Cm, self.lower, self.upper, self.log_ppd
                )
                
                for s_idx in range(current_batch):
                    accumulator.accumulate(batch_samples[s_idx])
                    
                total_samples_collected += current_batch
                
                pbar.update(current_batch)
                
                if callback is not None:
                    callback(total_samples_collected, self.nr)

        self.mean = accumulator.mean()
        self.sample_mean_error = accumulator.sample_mean_error()
        self.covariance = accumulator.covariance()
        self.sample_covariance_error = accumulator.sample_covariance_error()
        if save and accumulator.samples is not None:
            self.samples = np.stack(accumulator.samples)