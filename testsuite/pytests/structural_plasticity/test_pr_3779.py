# -*- coding: utf-8 -*-
#
# test_pr_3779.py
#
# This file is part of NEST.
#
# Copyright (C) 2004 The NEST Initiative
#
# NEST is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# NEST is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with NEST.  If not, see <http://www.gnu.org/licenses/>.

import math
import time
import unittest

import nest

# Check that NEST is installed with MPI support and mpi4py is available.
# If mpi4py is missing, we get an ImportError
# If mpi4py is installed but libmpi is missing, we get a RuntimeError.
# This only happens if we explicitly import MPI.
try:
    from mpi4py import MPI
    HAVE_MPI4PY = True
except (ImportError, RuntimeError):
    HAVE_MPI4PY = False

have_mpi = nest.build_info["have_mpi"]
test_with_mpi = have_mpi and HAVE_MPI4PY and nest.num_processes > 1
HAVE_GSL = nest.build_info["have_gsl"]

class TestGlobalShuffle(unittest.TestCase):
    def setUp(self):
        nest.ResetKernel()
        nest.verbosity = nest.VerbosityLevel.ERROR
        if test_with_mpi:
            self.comm = MPI.COMM_WORLD
            self.rank = self.comm.Get_rank()
            assert nest.Rank() == self.rank

    def _setup_structural_plasticity_network(self, n_pre, n_post, z_elements):
        pre_elements = {
            "Axon_ex": {"z": z_elements, "growth_rate": 0.0},
        }
        post_elements = {
            "Den_ex": {"z": z_elements, "growth_rate": 0.0},
        }

        pre = nest.Create("iaf_psc_alpha", n_pre, {"synaptic_elements": pre_elements})
        post = nest.Create("iaf_psc_alpha", n_post, {"synaptic_elements": post_elements})

        nest.CopyModel("static_synapse", "synapse_ex")

        nest.structural_plasticity_synapses = {
            "synapse_ex": {
                "synapse_model": "static_synapse",
                "pre_synaptic_element": "Axon_ex",
                "post_synaptic_element": "Den_ex",
            }
        }
        nest.structural_plasticity_update_interval = 1.0
        nest.EnableStructuralPlasticity()
        return pre, post

    def _measure_update_walltime(self, n_pre, n_post, z_elements, repetitions=3):
        """
        Measure one structural-plasticity update step.

        In MPI runs we use the maximum elapsed rank time as the effective
        wall time for a synchronized step and return the median over several
        repetitions to reduce timing noise.
        """
        measured = []

        for _ in range(repetitions):
            nest.ResetKernel()
            nest.local_num_threads = 1
            self._setup_structural_plasticity_network(n_pre, n_post, z_elements)

            # Warm-up to reduce one-time setup effects.
            nest.Simulate(1.0)

            if test_with_mpi:
                MPI.COMM_WORLD.Barrier()

            start = time.perf_counter()
            nest.Simulate(1.0)
            elapsed = time.perf_counter() - start

            if test_with_mpi:
                elapsed = MPI.COMM_WORLD.allreduce(elapsed, op=MPI.MAX)

            measured.append(elapsed)

        measured.sort()
        return measured[len(measured) // 2]

    @unittest.skipIf(not HAVE_GSL, "GSL is not available")
    def test_global_shuffle_scaling_is_not_quadratic(self):
        """
        Regression test for PR #3779.

        We cannot call SPManager.global_shuffle directly from Python, so this
        test exercises it via structural plasticity updates. It estimates a
        scaling exponent p in t ~ n^p for increasing network sizes and checks
        that p is clearly below quadratic behavior.

        """
        sizes = [300, 600, 1200]
        times = [self._measure_update_walltime(n, n, z_elements=5, repetitions=10) for n in sizes]

        ratio_t = times[-1] / times[0]
        ratio_n = sizes[-1] / sizes[0]
        exponent = math.log(ratio_t) / math.log(ratio_n)

        # O(n^2) would be near p=2; allow headroom for runtime noise.
        self.assertLess(
            exponent,
            1.6,
            msg=(
                f"Observed scaling exponent {exponent:.3f} suggests near-quadratic behavior: "
                f"times={times}, sizes={sizes}"
            ),
        )

def suite():
    test_suite = unittest.makeSuite(TestGlobalShuffle, "test")
    return test_suite


if __name__ == "__main__":
    unittest.main()