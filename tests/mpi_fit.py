#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Fit one seeded synthetic case under MPI and write the results, so that
``test_bmd_mpi.py`` can compare the output of different rank counts. Usage::

    mpirun -n 2 python tests/mpi_fit.py <savedir>
'''
import os
import sys

import numpy as np
from mpi4py import MPI

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from pybmd.bmd.standard import Standard      # noqa: E402
import pybmd.utils.parallel as utils_par    # noqa: E402


def main(savedir):
    comm = MPI.COMM_WORLD

    # allreduce must convert a non-native buffer, not reinterpret its bytes
    local = (np.arange(4.0) + comm.rank).astype('>f8')   # non-native
    total = utils_par.allreduce(local, comm)
    expected = comm.size * np.arange(4.0) + sum(range(comm.size))
    assert np.array_equal(total, expected), (total, expected)

    data = np.random.default_rng(0).standard_normal((200, 6, 5, 1))
    params = dict(n_dft=16, time_step=0.5, n_space_dims=2, n_variables=1,
                  n_overlap=8, regions=[1, 2], max_freq_idx=4,
                  store_modes=True, save_modes=True, savedir=savedir)
    bmd = Standard(params=params, comm=comm).fit(data)
    if comm.rank == 0:
        np.save(os.path.join(bmd.savedir_sim, 'modes_stored.npy'), bmd.modes)


if __name__ == '__main__':
    main(sys.argv[1])
