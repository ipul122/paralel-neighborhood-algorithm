from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from typing import Tuple
import warnings
from joblib import Parallel, delayed
from tqdm import tqdm
from os import cpu_count

from ._mcintegrals import MCIntegrals


class NAAppraiser:

    def __init__(
        self,
        n_resample: int,                                           #       
        n_walkers: int = 1,                                        # 
        initial_ensemble: NDArray | None = None,                   #
        log_ppd: NDArray | None = None,                            #
        bounds: Tuple[Tuple[float, float], ...] | None = None,     #
        verbose: bool = True,
        seed: int | None = None,
    ):

        self.initial_ensemble = initial_ensemble    #
        self.log_ppd = log_ppd                      #
        self.bounds = bounds                        #
        self.nd = len(bounds) 
        self.lower = np.array([b[0] for b in bounds])
        self.upper = np.array([b[1] for b in bounds])
        self.Cm = 1 / (self.upper - self.lower) ** 2
        self.verbose = verbose

        self.Ne = len(initial_ensemble)
        self.j = n_walkers if n_walkers >= 1 else 1
        self.nr = n_resample // n_walkers

        ss = np.random.SeedSequence(seed)
        self.rngs = [np.random.default_rng(s) for s in ss.spawn(self.j)]
        #rngs random seed for each walker 

    def run(self, save: bool = True, start_fraction: float = 0.5) -> None:
        
        if self.j == 1:
            accumulator = self._run_serial(save)
        else:
            if start_fraction < 0 or start_fraction > 1:
                raise ValueError("start_fraction must be between 0 and 1")
            accumulator = self._run_parallel(save, start_fraction)

        self.mean = accumulator.mean()
        self.sample_mean_error = accumulator.sample_mean_error()
        self.covariance = accumulator.covariance()
        self.sample_covariance_error = accumulator.sample_covariance_error()
        if save and accumulator.samples is not None:
            self.samples = np.stack(accumulator.samples)

    def _run_serial(self, save: bool = True) :#-> MCIntegrals:
        start = np.argmax(self.log_ppd)
        accumulator = MCIntegrals(self.nd, save)
        #print (accumulator)
        #exit()
        for x in self._random_walk_through_parameter_space(self.rngs[0], start):
            accumulator.accumulate(x)
        return accumulator
    def _run_parallel(
        self, save: bool = True, start_fraction: float = 0.5
    ) -> MCIntegrals:
        n_jobs = min(self.j, cpu_count())
        with Parallel(n_jobs=n_jobs) as parallel:
            # select start points for the random walks
            # these are taken from the best start_fraction*100% of cells to avoid walking
            # in low probability regions
            int_threshold = int(self.Ne * start_fraction)
            start_points = np.random.choice(
                np.argpartition(self.log_ppd, -int_threshold)[-int_threshold:],
                self.j,
                replace=False,
            )
            # ensure that at least one walker starts at the best cell
            start_points[0] = np.argmax(self.log_ppd)

            # create a MCIntegrals object for each walker
            accumulators = [MCIntegrals(self.nd, save) for _ in range(self.j)]

            # run the walkers in parallel
            accumulators = parallel(
                delayed(self._appraise)(acc, rng, start)
                for acc, rng, start in zip(accumulators, self.rngs, start_points)
            )

        # combine the results
        accumulator = MCIntegrals(self.nd, save)
        for acc in accumulators:
            accumulator.accumulate(acc)

        return accumulator

    def _appraise(
        self,
        accumulator: MCIntegrals,
        rng: np.random.Generator,
        start_k: int = 0,
    ):
        for x in self._random_walk_through_parameter_space(rng, start_k):
            accumulator.accumulate(x)
        return accumulator

    def _random_walk_through_parameter_space(
        self, rng: np.random.Generator, start_k: int = 0
    ):
        """
        Perform the random walk through parameter space.
        Yields a new sample at each iteration to be used for calculating summary statistics.
        """
        xA = self.initial_ensemble[start_k].copy()
        for _ in tqdm(
            range(self.nr), desc="NAII - Random Walk", disable=not self.verbose
        ):
            for i in range(self.nd):
                intersections, cells = self._axis_intersections(i, xA)
                xpi = self._random_step(i, intersections, cells, rng)
                xA[i] = xpi
            yield xA

    def _axis_intersections(self, axis: int, xA: NDArray) -> Tuple[NDArray, NDArray]:
        """
        Calculate the intersections of an axis passing through point vk in the kth cell
        with the boundaries of all cells

        Returns the intersection points and the cells the axis passes through
        """

        # The perpendicular distance to the axis from the current point in the walk
        # is constant as we traverse along the axis, so calculate before recursion
        d = (
            xA - self.initial_ensemble
        ) ** 2  # component-wise squared distance to all other cells
        d2 = np.sum(d * self.Cm, axis=1)  # total scaled distance to all other cells
        k = np.argmin(d2)  # index of the nearest cell
        dk2 = np.sum(
            np.delete(d, axis, 1) * np.delete(self.Cm, axis), axis=1
        )  # perpendicular distance to axis

        # Travel down the axis
        down_intersections, down_cells = self._get_axis_intersections(
            axis, k, dk2, down=True
        )
        # reverse the order of the down intersections and cells
        # so that the order of the intersections is from lowest to highest
        down_intersections = down_intersections[::-1]
        down_cells = down_cells[::-1]

        # Travel up the axis
        up_intersections, up_cells = self._get_axis_intersections(axis, k, dk2, up=True)

        return np.array(down_intersections + up_intersections), np.array(
            down_cells + [k] + up_cells
        )

    def _random_step(self, axis, intersections, cells, rng):
        """
        intersections are the points where the axis intersects the boundaries of the cells
        """
        while True:
            xpi = rng.uniform(self.lower[axis], self.upper[axis])  # proposed step
            k = self._identify_cell(xpi, intersections, cells)  # cell containing xpi

            r = rng.uniform(0, 1)
            logPxpi = self.log_ppd[k]
            logPmax = np.max(self.log_ppd[cells])
            if np.log(r) < logPxpi - logPmax:  # eqn (24) Sambridge 1999(II)
                return xpi

    def _get_axis_intersections(
        self, axis: int, k: int, di2: NDArray, down: bool = False, up: bool = False
    ):
        """
        axis: int - the axis to travel along
        k: int - the index of the current cell
        di2: NDArray - the perpendicular distance to the axis from current point in walk
        down: bool - whether to travel down the axis
        up: bool - whether to travel up the axis

        Returns:
            intersections: NDArray - the intersection points
            cells: NDArray - the cells the axis passes through
        """
        assert not (down and up)

        intersections = []
        cells = []

        # eqn (19) Sambridge 1999
        vk = self.initial_ensemble[k]
        vki = vk[axis]
        vji = self.initial_ensemble[:, axis]
        a = di2[k] - di2
        b = self.Cm[axis] * (vki - vji)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            xji = 0.5 * (vki + vji + a / b)

        if down:
            # isfinite check handles previous divide by zero
            mask = (vki <= vji) | ~np.isfinite(xji)
            closest = np.argmax
        else:
            mask = (vki >= vji) | ~np.isfinite(xji)
            closest = np.argmin

        xji = np.ma.array(xji, mask=mask)
        if xji.count() > 0:  # valid intersections found
            k_new = closest(xji)  # closest to vk
            intersections += [xji[k_new]]
            cells += [k_new]

            new_intersections, new_cells = self._get_axis_intersections(
                axis, k_new, di2, down, up
            )
            return intersections + new_intersections, cells + new_cells

        return intersections, cells

    def _identify_cell(self, xp: float, intersections: NDArray, cells: NDArray) -> int:
        """
        Given a set of intersections and the cells they pass through,
        identify the cell that contains the point xp.
        """
        closest_intersection = np.argmin(np.abs(intersections - xp))
        cell_id = (
            closest_intersection
            if xp < intersections[closest_intersection]
            else closest_intersection + 1
        )
        return cells[cell_id]
