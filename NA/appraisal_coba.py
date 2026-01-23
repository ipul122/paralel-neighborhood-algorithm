from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from typing import Tuple
import warnings
from tqdm import tqdm

from ._mcintegrals import MCIntegrals

def run_appraisal(
    n_resample: int,
    initial_ensemble: NDArray,
    log_ppd: NDArray,
    bounds: Tuple[Tuple[float, float], ...],
    n_walkers: int = 1,
    start_fraction: float = 0.5,
    save: bool = True,
    verbose: bool = True,
    seed: int | None = None,
):
    nd = len(bounds)
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    Cm = 1.0 / (upper - lower) ** 2

    Ne = len(initial_ensemble)
    nr = n_resample // max(1, n_walkers)

    rng = np.random.default_rng(seed)
    start = np.argmax(log_ppd)

    accumulator = MCIntegrals(nd, save)

    for x in random_walk(
        rng, nr, start, initial_ensemble, log_ppd, lower, upper, Cm, verbose
    ):
        accumulator.accumulate(x)

    results = {
        "mean": accumulator.mean(),
        "sample_mean_error": accumulator.sample_mean_error(),
        "covariance": accumulator.covariance(),
        "sample_covariance_error": accumulator.sample_covariance_error(),
    }

    if save and accumulator.samples is not None:
        results["samples"] = np.stack(accumulator.samples)

    return results


def random_walk(
    rng: np.random.Generator,
    nr: int,
    start_k: int,
    initial_ensemble: NDArray,
    log_ppd: NDArray,
    lower: NDArray,
    upper: NDArray,
    Cm: NDArray,
    verbose: bool,
):
    nd = initial_ensemble.shape[1]
    xA = initial_ensemble[start_k].copy()

    for _ in tqdm(range(nr), desc="NAII - Random Walk", disable=not verbose):
        for axis in range(nd):
            intersections, cells = axis_intersections(
                axis, xA, initial_ensemble, Cm
            )
            xA[axis] = random_step(
                axis, intersections, cells, rng, lower, upper, log_ppd
            )
        yield xA.copy()

def axis_intersections(
    axis: int,
    xA: NDArray,
    initial_ensemble: NDArray,
    Cm: NDArray,
):
    d = (xA - initial_ensemble) ** 2
    d2 = np.sum(d * Cm, axis=1)
    k = np.argmin(d2)

    dk2 = np.sum(
        np.delete(d, axis, 1) * np.delete(Cm, axis), axis=1
    )

    down_i, down_c = get_axis_intersections(
        axis, k, dk2, initial_ensemble, Cm, down=True
    )
    down_i, down_c = down_i[::-1], down_c[::-1]

    up_i, up_c = get_axis_intersections(
        axis, k, dk2, initial_ensemble, Cm, up=True
    )

    intersections = np.array(down_i + up_i)
    cells = np.array(down_c + [k] + up_c)

    return intersections, cells


def get_axis_intersections(
    axis: int,
    k: int,
    di2: NDArray,
    initial_ensemble: NDArray,
    Cm: NDArray,
    down: bool = False,
    up: bool = False,
):
    intersections = []
    cells = []

    vk = initial_ensemble[k]
    vki = vk[axis]
    vji = initial_ensemble[:, axis]

    a = di2[k] - di2
    b = Cm[axis] * (vki - vji)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        xji = 0.5 * (vki + vji + a / b)

    if down:
        mask = (vki <= vji) | ~np.isfinite(xji)
        closest = np.argmax
    else:
        mask = (vki >= vji) | ~np.isfinite(xji)
        closest = np.argmin

    xji = np.ma.array(xji, mask=mask)

    if xji.count() > 0:
        k_new = closest(xji)
        intersections.append(xji[k_new])
        cells.append(k_new)

        ni, nc = get_axis_intersections(
            axis, k_new, di2, initial_ensemble, Cm, down, up
        )
        return intersections + ni, cells + nc

    return intersections, cells


def random_step(
    axis: int,
    intersections: NDArray,
    cells: NDArray,
    rng: np.random.Generator,
    lower: NDArray,
    upper: NDArray,
    log_ppd: NDArray,
):
    while True:
        xpi = rng.uniform(lower[axis], upper[axis])
        k = identify_cell(xpi, intersections, cells)

        r = rng.uniform()
        logPmax = np.max(log_ppd[cells])

        if np.log(r) < log_ppd[k] - logPmax:
            return xpi

def identify_cell(
    xp: float,
    intersections: NDArray,
    cells: NDArray,
) -> int:
    idx = np.argmin(np.abs(intersections - xp))
    cell_id = idx if xp < intersections[idx] else idx + 1
    return cells[cell_id]
