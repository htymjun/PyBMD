# Cylinder Wake: Mode Bispectrum and Bispectral Modes

`examples/example5_cylinder_paper.py` reproduces the cylinder-wake results of Schmidt (2020,
*Nonlinear Dynamics*) with PyBMD on the `refs/bmd/wake_Re500.mat` dataset (the `refs/bmd` git
submodule): the mode bispectrum of Fig. 7 (`refs/figures/cylinder_bispectrum_sumdiff.pdf`) and the
bispectral modes of Figs. 8 and 9 (see [below](#spatial-modes-figs-8-and-9)).

```bash
git submodule update --init
MPLBACKEND=Agg python examples/example5_cylinder_paper.py   # ~2-4.5 min, ~0.9 GB RAM
```

It writes `example5_out/cylinder_bispectrum_sumdiff.png` and `example5_out/modes_k{k}_l{l}.png` in
the working directory. Both use the same BMD settings with `q = [u, v]`, as in the paper.

## How the reference parameters were recovered

Nothing in the repo generates this figure -- there is no "sumdiff" script anywhere, and
`refs/bmd/example1.m` only draws an interactive, index-labelled version on the fly. The reference
PDF itself was rasterized at 400 dpi and measured directly (pcolor cell pitch, axis extents, marker
positions, colorbar ticks) to recover the parameters it was produced with:

- panel (a) spans `xlim=[0, f(end)]`, `ylim=[-f(end), f(end)/2]` with `f(end) = 8.316`, i.e.
  `df = 1/57.6 = 0.017361`;
- panel (b) zooms to `[0, 0.8] x [-0.8, 0.8]` and circles six triads at multiples of
  `12 df = 0.2083`: `(12,12)`, `(12,0)`, `(24,12)`, `(24,24)`, `(36,12)`, `(36,24)`.

Matching `df = 0.017361` requires `dt = 0.06`, `n_dft = 960` -- twice the time resolution of the
`wake_Re500.mat` shipped here (`dt = 0.12`, `nt = 1024`). The shipped file is that same dataset,
subsampled 2x in time over the same total duration (`nt * dt = 122.88` either way). Consequently:

- **`n_dft = 480` at `dt = 0.12` reproduces the reference's exact frequency grid.** Verified with
  `pybmd.bmd.utils.triad_indices`: all six labelled triads exist in `regions=[1, 2]` at exactly the
  physical frequencies the reference marks them at, e.g. `(12,12,24)` -> `{0.2083, 0.2083, 0.4167}`.
- The dataset's own vortex-shedding frequency, measured from a zero-padded spectrum of `v`, is
  `f0 = 0.2074` (harmonics at 0.4145, 0.6225, 0.8302) `= 11.94 df`, i.e. index **12** -- which is
  why the reference labels the fundamental triad `(12,12,24)`.

Two differences from the published panel are unavoidable given the shipped (subsampled) data, and
are not attempts to hide a bug:

1. **Panel (a)'s extent is halved** -- `[0, 4.15] x [-4.15, 2.07]` instead of `[0, 8.32] x
   [-8.32, 4.16]` -- because the Nyquist frequency halves when the sampling rate halves at fixed
   `nt * dt`. Same picture, half the plane.
2. **3 blocks** at the reference's 50% overlap (`n_overlap = 240`), against presumably more in the
   original full-rate dataset, so the non-resonant background sits higher and the lattice contrast
   is a little weaker.

The reference's colorbar (jet, with the dark-blue end faded to white) was reproduced by sampling
its pixels directly rather than guessed; see `CMAP` in `example5_cylinder_paper.py`.

### Spatial weight: match `bmd.m`'s own default, not a "better" one

`B = Q3^H (Q1*Q2*w)/n_blocks` is linear in the spatial weight `w`, so `log|lambda_1|` (and the
whole colour scale) shifts by a constant depending on which weighting convention is used --
independent of everything above. `refs/bmd/bmd.m:279-281` defaults to `weight = ones(nx,1)`
("uniform") when no weight is passed, and `refs/bmd/example1.m` calls `bmd(u)` with none, so the
published figure was made with a **uniform** weight, not a physically-motivated quadrature one.
Using `pybmd.utils.weights.trapz_2d` instead (a reasonable default for other PyBMD work) shifted
this figure's `[vmin, vmax]` to `[-29.99, -4.49]` against the reference's measured
`[-28.4, +0.37]`; switching to `pybmd.utils.weights.uniform((n1, n2), n_vars=1, dV=1.0)` (matching
`bmd.m`'s default) brings it to `[-25.9, -0.48]` for `u` alone.

The paper analyses `q = [u, v]`, so the script now uses both variables. That gives `[-25.55, -0.69]`,
which moves `vmax` slightly further from the reference rather than closing the gap. An earlier
version of this note guessed that the remaining difference of about 1 in `log|lambda_1|` came
from fitting `u` alone; this measurement rules that out. The remaining difference is unexplained.
The different number of blocks (item 2 above) is one candidate that has not been tested.

## What to check when re-running this

The script prints the top 15 triads overall (via `pybmd.bmd.postproc.top_triads`). The acceptance
criterion is physical, not pixel-exact: the labelled triads should sit among the strongest
non-trivial (`k != 0`, `l != 0`) entries, and the global maximum should fall on the shedding
lattice (`k`, `l` multiples of 12). A run with `q = [u, v]` gave:

```
top 15 triads (k != 0 and l != 0):
  ( 12,-12,  0)  |lambda_1| = 4.8380e-01
  ( 12, 12, 24)  |lambda_1| = 4.7334e-01
  ( 24,-12, 12)  |lambda_1| = 4.3104e-01
  ...
```

The paper places the global maximum at `(12,12,24)`. Here the difference self-interaction
`(12,-12,0)`, the `{f0, -f0, 0}` triad that drives the mean-flow deformation, is 2% above it. With
`u` alone the order was `(24,-12,12)` at 0.617 and then `(12,12,24)` at 0.611. In both cases the
leading triads lie on the shedding lattice and are within a few percent of each other.

## Spatial modes (Figs. 8 and 9)

The script does not redraw the paper's layout for these figures. It writes one `plot_triad_modes`
figure per triad circled in Fig. 7b, `example5_out/modes_k{k}_l{l}.png`, each showing the u and v
components of `phi_{k+l}`, `phi_{k o l}` and the interaction map `|phi_{k o l} phi_{k+l}|`. Fig. 8
is the `phi_{k+l}` u panel of the six figures; Fig. 9 is the whole of `modes_k12_l12.png`.

Holding the modes of all ~43k triads of the Fig. 7 fit would take ~15 GB. The modes therefore come
from a second fit with the same settings, restricted to `regions=[1]` and `max_freq_idx=36` (703
triads). Each triad is solved independently, so the restriction leaves the results of the remaining
triads unchanged. The script checks this: `|lambda_1|` of the six labelled triads agrees between
the two fits to a relative difference of 0.

The script prints the properties that Figs. 8 and 9 show. A run gave:

```
   (k, l)   n  |lambda_1|   sym(u)  sym(v)  lambda_x  n*lambda_x
  (12, 0)   1  5.0091e-01   0.001   1.000     3.998     3.998
  (12,12)   2  4.7334e-01   0.998   0.001     1.972     3.943
  (24,12)   3  9.3953e-02   0.001   0.999     1.371     4.112
  (24,24)   4  4.1311e-03   0.998   0.002     1.021     4.083
  (36,12)   4  2.0278e-02   0.996   0.005     1.023     4.091
  (36,24)   5  2.5207e-03   0.003   0.999     0.834     4.172
Fig. 9: interaction map psi_{12,12} = |phi_{12o12} phi_{12+12}|
  u: max 2.0877e-04 at (x, y) = (5.65, 0.65); 3.4% of the domain above half its max
  v: max 8.7707e-04 at (x, y) = (12.54, 0.39); 11.6% of the domain above half its max
```

- `n` is the harmonic of `f0` at which `phi_{k+l}` oscillates. `sym` is the share of the mode's
  energy that is even in `y`. For every triad, u is antisymmetric at odd `n` and symmetric at even
  `n`, and v has the opposite parity. This is the symmetry of the vortex-shedding harmonics that
  Fig. 8 shows.
- `lambda_x` is the peak streamwise wavelength of `phi_{k+l}`, measured on v. The product
  `n*lambda_x` stays within 4.0 +/- 0.2 along the cascade, so each interaction shortens the
  wavelength in proportion to the frequency. This matches the paper's statement that each
  interaction yields new streamwise wavenumber components.
- `(12,12,24)` is the strongest triad with `k, l != 0` here too.
- For Fig. 9, the peak of the v interaction map is 4.2 times that of the u map, and the v map
  covers 3.4 times as much of the domain above half its maximum. The paper reports both: "the
  transverse component furthermore attains a larger maximum value than the streamwise component
  and is less spatially confined".

One claim of Fig. 9 cannot be checked on this dataset. The paper finds the interaction strongest
"in the wake region just downstream of the cylinder", but `wake_Re500.mat` covers only
`x = 2.7`-`14.9`, so the region near the cylinder is not in the data. The figures therefore show no
cylinder, and the location of the maximum above is the maximum within that window only.
