# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

This repo's CLAUDE.md is split hierarchically: this file covers commands and the project
overview; detail specific to a subtree lives in that subtree's own CLAUDE.md and is only relevant
when working there — [`pybmd/bmd/CLAUDE.md`](pybmd/bmd/CLAUDE.md) for the BMD/CBMD implementation,
[`tests/CLAUDE.md`](tests/CLAUDE.md) for testing strategy.

## Commands

```bash
pip install -e '.[mpi,io,test]'      # editable install; extras: mpi, io (.mat/.nc), test, docs
git submodule update --init          # populate refs/bmd (the MATLAB reference), only needed
                                     # for tests/test_octave_reference.py -- see tests/CLAUDE.md

pytest                                # full suite, ~90 s, 169 tests: 13 `slow` (Octave
                                      # cross-validation, figure regeneration), 1 `mpi`
pytest -m "not slow and not mpi"      # fast subset, ~30 s
pytest tests/optimizers -q            # one directory (numerical-radius solver tests,
                                      # one test function per file)
pytest tests/test_bmd_serial.py::test_bispectrum_matches_closed_form -q   # one test
pytest -k "conjugate or closed_form"  # by name

python -m pyflakes pybmd/ tests/ examples/   # only linter used; ignore the
                                             # "f-string is missing placeholders"
                                             # hits, they match the PySPOD print idiom
```

`tests/conftest.py` puts this checkout first on `sys.path`, so the suite tests the source tree
even if a non-editable `pybmd` is installed in site-packages; still prefer the editable install
above so that scripts and the examples see the same code.

The MPI test (`tests/test_bmd_mpi.py`, marker `mpi`) launches `mpirun -n 1` and `-n 2` itself and
asserts **bit-identical** `L`, `T`, `coeffs` and modes between them; it self-skips without `mpirun`
or `mpi4py`. It sets `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1` for its subprocesses — threaded
BLAS reorders reductions — so set the same when comparing MPI runs by hand.

Examples run from any directory (the fixture path is resolved relative to `examples/data.py`):
`MPLBACKEND=Agg python examples/example1_cylinder.py`; they write `example*_out/` in the working
directory.

## Architecture

Bispectral mode decomposition, ported from O. T. Schmidt's MATLAB `bmd.m`/`cbmd.m`, structured
like [PySPOD](https://github.com/MathEXLab/PySPOD) for conventions. BMD finds triads `(k, l, k+l)`
whose components are quadratically phase coupled, and returns a mode bispectrum `L` plus two modes
per triad (optionally four, with `constituent_modes` adding the modes at `k` and `l` themselves).

The implementation (`pybmd/bmd/`) — the Base/Standard/Cross split, data flow, invariants that are
easy to break, the ported-solver deviations from the MATLAB reference, and data conventions — is
documented in [`pybmd/bmd/CLAUDE.md`](pybmd/bmd/CLAUDE.md).

## Testing strategy

See [`tests/CLAUDE.md`](tests/CLAUDE.md) for the regression strategy, including the Octave
cross-validation setup against the `refs/bmd` submodule.
