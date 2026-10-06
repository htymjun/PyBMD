#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Results must be bit-identical across MPI rank counts and to a serial run:
every triad is solved by exactly one rank from the same node-shared data, the
DFT of each spatial point does not depend on which rank transforms it, and the
reduction only adds exact zeros. Run with ``OMP_NUM_THREADS=1``; a threaded BLAS reorders
reductions and breaks bitwise equality.
'''
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

SCRIPT = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'mpi_fit.py')


def _mpirun():
    exe = shutil.which('mpirun') or shutil.which('mpiexec')
    if exe is None:
        pytest.skip('mpirun is not available')
    version = subprocess.run([exe, '--version'], capture_output=True,
                             text=True).stdout
    return exe, ('--oversubscribe',) if 'Open MPI' in version else ()


CASES = ('standard', 'cross')


def _run(exe, extra, n_ranks, savedir):
    '''Run the script on ``n_ranks`` ranks, or serially (``comm=None``)
    for None; return the results directory of each case.'''
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               MKL_NUM_THREADS='1')
    if n_ranks is None:
        cmd = [sys.executable, SCRIPT, str(savedir), 'serial']
    else:
        cmd = [exe, '-n', str(n_ranks), *extra, sys.executable, SCRIPT,
               str(savedir)]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True,
                          timeout=600)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    out = {}
    for case in CASES:
        # the single nfft*_novlp*_nblks* directory
        sim, = os.listdir(os.path.join(savedir, case))
        out[case] = os.path.join(savedir, case, sim)
    return out


def _assert_identical(serial, parallel):
    with np.load(os.path.join(serial, 'bispectrum.npz')) as a, \
            np.load(os.path.join(parallel, 'bispectrum.npz')) as b:
        for key in ('L', 'T', 'freq', 'f_idx'):
            np.testing.assert_array_equal(a[key], b[key])
    for name in ('coeffs.npy', 'modes_stored.npy'):
        np.testing.assert_array_equal(np.load(os.path.join(serial, name)),
                                      np.load(os.path.join(parallel, name)))

    files = sorted(os.listdir(os.path.join(serial, 'modes')))
    assert files == sorted(os.listdir(os.path.join(parallel, 'modes')))
    assert len(files) == np.load(os.path.join(serial, 'coeffs.npy')).shape[0]
    for f in files:
        np.testing.assert_array_equal(
            np.load(os.path.join(serial, 'modes', f)),
            np.load(os.path.join(parallel, 'modes', f)))


@pytest.mark.mpi
def test_results_are_bit_identical_across_rank_counts(tmp_path):
    pytest.importorskip('mpi4py')
    exe, extra = _mpirun()
    reference = _run(exe, extra, None, tmp_path / 'serial')
    # 4 ranks split the 30 spatial points of the DFT unevenly
    for n_ranks in (1, 2, 4):
        parallel = _run(exe, extra, n_ranks, tmp_path / f'ranks{n_ranks}')
        for case in CASES:
            _assert_identical(reference[case], parallel[case])
