#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Fit two seeded synthetic cases (BMD and CBMD) under MPI and write the
results to ``<savedir>/standard`` and ``<savedir>/cross``, so that
``test_bmd_mpi.py`` can compare the output of different rank counts. Only rank
0 passes the data and the (non-uniform) weights; the others pass None, and
would fall back to uniform weights if they were not broadcast. Usage::

    mpirun -n 2 python tests/mpi_fit.py <savedir>
    python tests/mpi_fit.py <savedir> serial     # comm=None reference
'''
import os
import pickle
import sys

import numpy as np
from mpi4py import MPI

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from pybmd.bmd.cross import Cross            # noqa: E402
from pybmd.bmd.standard import Standard      # noqa: E402
import pybmd.utils.parallel as utils_par    # noqa: E402


def main(savedir, serial=False):
    if serial:
        _fit_all(savedir, None)
        return
    comm = MPI.COMM_WORLD

    # allreduce must convert a non-native buffer, not reinterpret its bytes
    local = (np.arange(4.0) + comm.rank).astype('>f8')   # non-native
    total = utils_par.allreduce(local, comm)
    expected = comm.size * np.arange(4.0) + sum(range(comm.size))
    assert np.array_equal(total, expected), (total, expected)

    _fit_all(savedir, comm)


def _fit_all(savedir, comm):
    root = comm is None or comm.rank == 0
    # nx = 30 points, which 4 ranks do not divide evenly
    rng = np.random.default_rng(0)
    data = rng.standard_normal((200, 6, 5, 3))
    w = rng.uniform(0.5, 1.5, (6, 5))
    common = dict(n_dft=16, time_step=0.5, n_space_dims=2, n_overlap=8,
                  regions=[1, 2], max_freq_idx=4, store_modes=True,
                  save_modes=True)
    cases = [
        (Standard, dict(common, n_variables=1,
                        savedir=os.path.join(savedir, 'standard')),
         data[..., :1], w[..., np.newaxis]),
        (Cross, dict(common, n_variables=3, state_idx=[0, 1],
                     qr_idx=[[1, 2], [0, 2]],
                     savedir=os.path.join(savedir, 'cross')),
         data, w),
    ]
    for cls, params, d, wt in cases:
        weights = dict(weights=wt, weights_name='random') if root else None
        bmd = cls(params=params, weights=weights,
                  comm=comm).fit(d if root else None)
        if root:
            # through a pickle round trip: the object holds MPI handles and
            # shared memory, which must be dropped or copied when pickled
            bmd = pickle.loads(pickle.dumps(bmd))
            np.save(os.path.join(bmd.savedir_sim, 'modes_stored.npy'),
                    bmd.modes)


if __name__ == '__main__':
    main(sys.argv[1], serial=len(sys.argv) > 2 and sys.argv[2] == 'serial')
