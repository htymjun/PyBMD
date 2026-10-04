# Complex-conjugation audit

## Finding

`normalize_data=True` is incorrect when PyBMD is given complex-valued time
domain data.  In `Base._compute_blocks`, the variance estimate is calculated
as

```python
np.sum((q_blk - np.mean(q_blk, axis=0))**2, axis=0) / den
```

([`pybmd/bmd/base.py:575`](../pybmd/bmd/base.py#L575)).  For a complex random
variable, variance must use the Hermitian square,

\[
  \operatorname{var}(q) = \frac{1}{N-1}\sum_t |q_t-\bar q|^2,
\]

not the algebraic square \((q_t-\bar q)^2\).  The latter is phase dependent
and can be complex or cancel to zero.  The subsequent threshold and square
root then standardize by the wrong quantity.

### Minimal reproducer

```python
z = np.array([1, 1j, -1, -1j], dtype=complex)
d = z - z.mean()

np.sum(d**2) / 3          # 0j: current PyBMD calculation
np.sum(np.abs(d)**2) / 3  # 4/3: correct variance
```

The current code replaces the zero result with one and leaves this signal with
variance `4/3`, rather than normalizing it to one.  Other complex signals can
produce a complex scale factor, arbitrarily rotating and rescaling the data
before the FFT.  This can change `B`, its numerical-radius optimizer result,
and the modes.  Normal real-valued input is unaffected because `x**2` and
`abs(x)**2` agree for real `x`.

## Comparison with the MATLAB reference

The BMD matrix construction itself is *not* the fault:

| Operation | PyBMD | `refs/bmd` | Result |
| --- | --- | --- | --- |
| Cross-spectral matrix | `q_sum.conj().T @ (...)` | `Q_hat_f3' * (...)` | Both use a conjugate transpose. |
| Mode norm | `np.vdot(psi, psi * w)` | `Psi' * (Psi .* weight)` | Both are Hermitian weighted inner products. |
| Transfer term | `np.vdot(psi_sum, psi_prod)` | `(...)' * (...)` | Both conjugate the sum-interaction factor. |

MATLAB's apostrophe is conjugate transpose for complex arrays (as opposed to
`.'`, its non-conjugating transpose).  Thus `refs/bmd/bmd.m:185,205,218` and
`refs/bmd/cbmd.m:161,181,196` use the right convention.  PyBMD mirrors it in
`Base._triad_loop` (`base.py:646,663`) and `normalize_mode`
(`pybmd/bmd/utils.py:374`).  Replacing any of these with a plain transpose
would be a separate, serious error, but that error is not present in the
audited BMD/CBMD matrix products.

The reference has no `normalize_data` option, so it has no direct equivalent
of this defect.  Consequently this is a PyBMD extension bug, not a PyBMD vs.
MATLAB porting mismatch.

## Recommended fix and regression coverage

Replace the variance line with a Hermitian magnitude square:

```python
centered = q_blk - np.mean(q_blk, axis=0)
q_var = np.sum(np.abs(centered)**2, axis=0) / den
```

Add a regression test with the four-point reproducer above and assert that,
after normalization, each non-degenerate column has
`sum(abs(q_blk)**2) / (n_dft - 1) == 1`.  A real-data regression should also
assert unchanged output, since the proposed expression is identical for real
inputs.

## Verification performed

- Inspected every conjugate-transpose/inner-product site in `pybmd/bmd` and
  `refs/bmd/bmd.m` / `refs/bmd/cbmd.m`.
- Executed the reproducer: current variance `0j`; Hermitian variance
  `1.3333333333333333`.
- Ran `pytest tests/optimizers -q`: **120 passed**.  Those tests exercise the
  complex numerical-radius matrices, but they do not cover complex input with
  `normalize_data=True`.

## Resolution

Complex input never reached this code path in practice: `get_data_array`
cast the data to the requested float type, which silently discarded the
imaginary part (a `ComplexWarning` only). Both ends are now closed:

- `pybmd/utils/io.py::get_data_array` raises `TypeError` on complex input,
  since BMD's two-sided spectrum and sum/difference regions rely on the
  conjugate symmetry of a real signal (`tests/test_io_rejects_complex.py`);
- the variance in `Base._compute_blocks` uses the Hermitian square
  `np.abs(centered)**2`, which is identical for real data.
