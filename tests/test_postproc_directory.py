import os

import numpy as np
import yaml

from pybmd.bmd import utils as bmd_utils
from pybmd.bmd.postproc import (
    find_result_directories,
    load_results,
    resolve_results_path,
    top_triads,
)


def _write_result(path):
    path.mkdir(parents=True)
    triads = bmd_utils.triad_indices(
        n_dft=16, dt=0.5, regions=(1, 2), max_freq_idx=4)
    L = np.full((triads.n_freq, triads.n_freq), np.nan, dtype=complex)
    T = np.full((triads.n_freq, triads.n_freq), np.nan)
    values = np.arange(1, triads.n_triads + 1, dtype=float)
    L[triads.f1_idx, triads.f2_idx] = values
    T[triads.f1_idx, triads.f2_idx] = -values

    np.savez(path / 'bispectrum.npz',
             L=L, T=T, freq=triads.freq, f_idx=triads.f_idx)
    triads.to_npz(path / 'triads.npz')
    with open(path / 'params_modes.yaml', 'w') as f:
        yaml.dump({'n_dft': 16, 'time_step': 0.5}, f)
    return triads, values


def test_load_results_from_parent_directory(tmp_path):
    sim = tmp_path / 'results' / 'nfft16_novlp8_nblks3'
    triads, _ = _write_result(sim)

    assert resolve_results_path(tmp_path / 'results') == os.path.abspath(sim)
    results = load_results(tmp_path / 'results')

    assert results.path == os.path.abspath(sim)
    assert np.array_equal(results.f_idx, triads.f_idx)
    assert np.array_equal(results.triads.k, triads.k)
    assert results.params['n_dft'] == 16


def test_find_result_directories_returns_sorted_matches(tmp_path):
    first = tmp_path / 'a' / 'nfft16_novlp8_nblks3'
    second = tmp_path / 'b' / 'nfft16_novlp4_nblks4'
    _write_result(second)
    _write_result(first)

    assert find_result_directories(tmp_path) == [
        os.path.abspath(first),
        os.path.abspath(second),
    ]


def test_top_triads_reports_frequency_metadata(tmp_path):
    sim = tmp_path / 'nfft16_novlp8_nblks3'
    triads, values = _write_result(sim)
    nz = np.flatnonzero((triads.k != 0) & (triads.l != 0))
    expected = nz[np.argmax(values[nz])]

    top = top_triads(load_results(sim), n=3)

    assert top.shape == (3,)
    assert top[0]['triad_idx'] == expected
    assert top[0]['k'] == triads.k[expected]
    assert top[0]['l'] == triads.l[expected]
    assert top[0]['kl'] == triads.kl[expected]
    assert top[0]['region'] == triads.region[expected]
    assert top[0]['value'] == values[expected]
