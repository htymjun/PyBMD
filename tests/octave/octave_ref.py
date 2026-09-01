'''
Cross-validation harness against the original MATLAB ``bmd.m``/``cbmd.m``,
run unmodified (Tier B) or lightly instrumented (Tier A) under Octave.

See ``docs/octave_cross_validation.md`` for the fairness rules this module
exists to enforce (weight flatten order, single- vs double-precision input,
window/overlap parity, the CBMD variable-axis position, ...), and
``tests/test_octave_reference.py`` for the tests that use it.

Everything here is read-only with respect to ``refs/bmd/``: instrumentation
writes a *copy* of ``bmd.m``/``cbmd.m`` into a scratch directory, the
originals are never touched.
'''
import hashlib
import os
import re
import shutil
import subprocess
import tempfile

import numpy as np
import pytest
import scipy.io

__all__ = ['require_octave', 'require_refs', 'require_full_dataset', 'run',
           'brute_force_radius', 'check_solver_reachable', 'to_matlab_weight',
           'octave_hamming_window']

_HERE = os.path.dirname(os.path.realpath(__file__))
_REPO_ROOT = os.path.realpath(os.path.join(_HERE, '../../'))
_REFS_BMD = os.path.join(_REPO_ROOT, 'refs', 'bmd')

_CACHE = {}          # in-process cache: same parameters -> reused result
_SCRATCH = None       # lazily created, shared for the life of the process


def _require_or_skip(message):
    '''
    Skip optional local cross-validation prerequisites by default, but fail
    hard when the dedicated CI job says those prerequisites must be present.
    '''
    flag = os.environ.get('PYBMD_REQUIRE_OCTAVE_REF', '').lower()
    if flag in ('1', 'true', 'yes', 'on'):
        pytest.fail(message)
    pytest.skip(message)


def require_octave():
    '''Return the ``octave-cli`` executable, or skip the test.'''
    exe = shutil.which('octave-cli') or shutil.which('octave')
    if exe is None:
        _require_or_skip(
            'octave-cli not found on PATH; install Octave to run the '
            'reference cross-validation.')
    return exe


def require_refs():
    '''Return the path to ``refs/bmd/``, or skip the test.'''
    if not os.path.exists(os.path.join(_REFS_BMD, 'bmd.m')):
        _require_or_skip(
            f'{_REFS_BMD}/bmd.m not found; refs/bmd is a git submodule '
            f'(research/non-commercial license, so it is a pointer rather '
            f'than vendored code) -- run `git submodule update --init` to '
            f'populate it.')
    return _REFS_BMD


def require_full_dataset():
    '''Return the path to ``refs/bmd/wake_Re500.mat``, or skip the test.'''
    refs_dir = require_refs()
    path = os.path.join(refs_dir, 'wake_Re500.mat')
    if not os.path.exists(path):
        _require_or_skip(f'{path} not found.')
    return path


def _scratch_dir():
    '''A scratch directory, created once per process and reused thereafter.'''
    global _SCRATCH
    if _SCRATCH is None:
        _SCRATCH = tempfile.mkdtemp(prefix='pybmd_octave_')
    return _SCRATCH


# ---------------------------------------------------------------------------
# instrumentation: a scratch copy of bmd.m/cbmd.m that also returns Q_hat and
# every per-triad B, so the DFT/blocking/weighting stage can be checked
# independently of the numerical-radius solver (Tier A).
# ---------------------------------------------------------------------------

_PATCHES = {
    'bmd': dict(
        src='bmd.m', out_name='bmd_instr.m', func_name='bmd_instr',
        signature=("function [L,P,f,idx,T] = bmd(X,varargin)",
                   "function [L,P,f,idx,T,Q_hat,B_all] = bmd_instr(X,varargin)"),
        declare=("P       = zeros(2,nTriads,nx);",
                "P       = zeros(2,nTriads,nx);\nB_all   = zeros(nBlks,nBlks,nTriads);"),
        capture=("    B                   = Q_hat_f3'*bsxfun(@times,Q_hat_f1.*Q_hat_f2,weight)/nBlks;",
                "    B                   = Q_hat_f3'*bsxfun(@times,Q_hat_f1.*Q_hat_f2,weight)/nBlks;\n"
                "    B_all(:,:,i)        = B;"),
    ),
    'cbmd': dict(
        src='cbmd.m', out_name='cbmd_instr.m', func_name='cbmd_instr',
        signature=("function [L,P,f,idx,T] = cbmd(X,varargin)",
                   "function [L,P,f,idx,T,Q_hat,B_all] = cbmd_instr(X,varargin)"),
        declare=("P           = zeros(2,nTriads,nx*nState);",
                "P           = zeros(2,nTriads,nx*nState);\nB_all       = zeros(nBlks,nBlks,nTriads);"),
        capture=("    B                  = Q_hat_s'*bsxfun(@times,Q_hat_qr,weights)/nBlks;    ",
                "    B                  = Q_hat_s'*bsxfun(@times,Q_hat_qr,weights)/nBlks;\n"
                "    B_all(:,:,i)       = B;"),
    ),
}


def _apply_patch(text, old, new, label):
    n = text.count(old)
    if n != 1:
        raise RuntimeError(
            f'expected exactly one occurrence of the {label!r} anchor in '
            f'the reference source, found {n}; refs/bmd/*.m may have '
            f'changed and the instrumentation needs updating.')
    return text.replace(old, new, 1)


def instrument(kind, refs_dir):
    '''
    Write an instrumented copy of ``bmd.m``/``cbmd.m`` to the scratch
    directory and return ``(scratch_dir, function_name)``. ``refs/bmd/*.m``
    itself is never modified. Idempotent: safe to call every time.
    '''
    spec = _PATCHES[kind]
    src_path = os.path.join(refs_dir, spec['src'])
    with open(src_path) as f:
        text = f.read()
    text = _apply_patch(text, *spec['signature'], 'signature')
    text = _apply_patch(text, *spec['declare'], 'B_all declaration')
    text = _apply_patch(text, *spec['capture'], 'B capture')

    scratch = _scratch_dir()
    out_path = os.path.join(scratch, spec['out_name'])
    with open(out_path, 'w') as f:
        f.write(text)
    return scratch, spec['func_name']


# ---------------------------------------------------------------------------
# running the reference
# ---------------------------------------------------------------------------

def to_matlab_weight(w):
    '''
    Flatten a PyBMD weight array into the plain vector ``bmd.m``/``cbmd.m``
    expect, in the *same* order :func:`_flatten_for_matlab` uses for the data
    -- see its docstring for why this must be a C-order (PyBMD-native)
    flatten, not MATLAB's own column-major one.
    '''
    return np.ascontiguousarray(w, dtype=np.float64).ravel()


def _flatten_for_matlab(data, kind):
    '''
    Collapse every axis but time (and, for CBMD, the variable axis) into one,
    in PyBMD's own C order, *before* the array is handed to Octave.

    MATLAB flattens trailing dimensions in column-major (Fortran) order --
    ``X(idx,:)`` and ``weight(:)`` both do this internally -- while PyBMD
    flattens in row-major (C) order. Sending a genuinely multi-dimensional
    spatial array across and letting MATLAB's own colon-flatten act on it
    would silently disagree with PyBMD's flatten for any non-square spatial
    shape (verified: element 0 of the two flattenings always matches, element
    1 already does not). Flattening in Python first, with the same
    :func:`numpy.ravel` convention used for the weight vector, sidesteps the
    cross-language ambiguity entirely: MATLAB then only ever flattens an
    already-1-D axis, where order is moot. This is a *different* hazard from
    the intra-PyBMD one ``CLAUDE.md`` documents (flat weight vectors are
    rejected there because *nothing* records the order they were built in);
    here both sides' order is pinned explicitly, by construction.

    ``L``, ``T`` and the mode *directions* are unaffected by which spatial
    layout is used, since B is a sum over the spatial/variable index weighted
    consistently with the data -- so this does not weaken the comparison, it
    only changes the shape ``P`` comes back in (flat, rather than reshaped
    to ``xshape``).

    :param numpy.ndarray data: PyBMD-layout data, ``(nt, *xshape, nv)``.
    :param str kind: ``'bmd'`` or ``'cbmd'``.

    :return: ``(nt, nxv)`` for ``'bmd'``; ``(nt, nv, nx)`` for ``'cbmd'``
        (variable axis kept separate, matching ``cbmd.m``'s own layout).
    :rtype: numpy.ndarray
    '''
    nt = data.shape[0]
    if kind == 'bmd':
        return data.reshape(nt, -1)
    data = np.moveaxis(data, -1, 1)          # (nt, nv, *xshape)
    return data.reshape(nt, data.shape[1], -1)


def _hash_array(a):
    a = np.ascontiguousarray(a)
    return f'{a.shape}{a.dtype}{hashlib.sha1(a.tobytes()).hexdigest()[:16]}'


def _cache_key(**kwargs):
    parts = []
    for k in sorted(kwargs):
        v = kwargs[k]
        if isinstance(v, np.ndarray):
            parts.append(f'{k}={_hash_array(v)}')
        else:
            parts.append(f'{k}={v!r}')
    return hashlib.sha1('|'.join(parts).encode()).hexdigest()[:20]


def _run_octave(driver, env, timeout):
    exe = require_octave()
    proc = subprocess.run(
        [exe, '--no-gui', '--quiet', driver], env=env,
        capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(
            f'octave-cli failed (exit {proc.returncode}) running {driver}\n'
            f'--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}')


def run(kind, data, *, window, weight, n_overlap, dt, regions, tol=1e-6,
        n_it_max=500, solver='MengiOverton', max_freq_idx=None,
        s_idx=None, qr_idx=None, instrumented=False, timeout=600):
    '''
    Run ``bmd.m`` or ``cbmd.m`` under Octave and return its outputs.

    :param str kind: ``'bmd'`` or ``'cbmd'``.
    :param numpy.ndarray data: the data, laid out as PyBMD expects it
        (``(nt, *xshape, nv)``, variables last); flattened internally with
        :func:`_flatten_for_matlab` (see its docstring for why).
    :param numpy.ndarray window: the temporal window, length ``n_dft``.
    :param numpy.ndarray weight: the *PyBMD-shaped* weight array (full
        spatial shape, plus a trailing variable axis for ``'bmd'``); flattened
        internally with :func:`to_matlab_weight`.
    :param int n_overlap: overlap, in snapshots.
    :param float dt: time-step.
    :param regions: sequence of region numbers, 1..8.
    :param float tol: solver tolerance.
    :param int n_it_max: solver iteration cap.
    :param str solver: ``'MengiOverton'`` or ``'HeWatson'``.
    :param int max_freq_idx: bound on ``|k|``, ``|l|``. None means no bound.
    :param s_idx: ``cbmd.m``'s 1-based ``opts.s_idx`` (only for ``'cbmd'``).
    :param qr_idx: ``cbmd.m``'s 1-based ``opts.qr_idx`` (only for ``'cbmd'``).
    :param bool instrumented: if True, run the patched copy that also
        returns ``Q_hat`` and every per-triad ``B`` (Tier A).
    :param float timeout: subprocess timeout, in seconds.

    :return: dict with keys ``L``, ``P``, ``f``, ``idx``, ``T``, and, if
        ``instrumented``, also ``Q_hat`` and ``B_all``.
    :rtype: dict
    '''
    assert kind in ('bmd', 'cbmd')
    refs_dir = require_refs()
    require_octave()

    data = np.asarray(data, dtype=np.float64)
    window = np.asarray(window, dtype=np.float64).ravel()
    weight_mat = to_matlab_weight(weight)
    regions = np.asarray(sorted(regions), dtype=np.float64)
    nfreq = np.array([] if max_freq_idx is None else [float(max_freq_idx)])
    data_mat = _flatten_for_matlab(data, kind)

    if kind == 'cbmd':
        s_idx_mat = np.asarray(s_idx, dtype=np.float64).reshape(-1, 1)
        qr_idx_mat = np.asarray(qr_idx, dtype=np.float64)
    else:
        s_idx_mat = np.array([])
        qr_idx_mat = np.array([])

    key = _cache_key(
        kind=kind, data=data_mat, window=window, weight=weight_mat,
        n_overlap=n_overlap, dt=dt, regions=regions, tol=tol,
        n_it_max=n_it_max, solver=solver, nfreq=nfreq, s_idx=s_idx_mat,
        qr_idx=qr_idx_mat, instrumented=instrumented)
    if key in _CACHE:
        return _CACHE[key]

    scratch = _scratch_dir()
    if instrumented:
        addpath_dir, func_name = instrument(kind, refs_dir)
    else:
        addpath_dir, func_name = refs_dir, kind

    in_path = os.path.join(scratch, f'in_{key}.mat')
    out_path = os.path.join(scratch, f'out_{key}.mat')
    mdict = dict(X=data_mat, window=window, weight=weight_mat,
                nOvlp=np.array([[float(n_overlap)]]), dt=np.array([[float(dt)]]),
                regions=regions, nfreq=nfreq, tol=np.array([[float(tol)]]),
                nitmax=np.array([[float(n_it_max)]]), solver=solver)
    if kind == 'cbmd':
        mdict['s_idx'] = s_idx_mat
        mdict['qr_idx'] = qr_idx_mat
    scipy.io.savemat(in_path, mdict)

    driver = os.path.join(_HERE, f'run_{kind}.m')
    env = dict(os.environ)
    env.update(PYBMD_OCT_ADDPATH=addpath_dir, PYBMD_OCT_FUNC=func_name,
              PYBMD_OCT_IN=in_path, PYBMD_OCT_OUT=out_path)
    _run_octave(driver, env, timeout)

    with open(out_path, 'rb') as f:
        out = dict(scipy.io.loadmat(f))
    out = {k: v for k, v in out.items() if not k.startswith('__')}
    _CACHE[key] = out
    return out


# ---------------------------------------------------------------------------
# solver reachability (independent of the driver above: no data needed)
# ---------------------------------------------------------------------------

def check_solver_reachable(kind, solver_name, n=3):
    '''
    Try ``opts.solver = solver_name`` on a tiny random matrix and report
    whether the reference accepts and runs it.

    :return: ``(ok, message)`` -- ``ok`` is False both when the option
        validator rejects the name and when it passes validation but the
        inner ``switch`` has no matching case (``'Unknown solver.'`` either
        way, from two different call sites).
    :rtype: tuple(bool, str)
    '''
    refs_dir = require_refs()
    exe = require_octave()
    if kind == 'cbmd':
        script = f'''
addpath('{refs_dir}');
rand('state', 1);
X = rand(64, 3, 2, 2);
w = ones(4, 1);
opts.regions = [1 2];
opts.nfreq = 2;
opts.solver = '{solver_name}';
try
  evalc('[L,P,f,idx] = cbmd(X, 16, w, 8, 1/16, opts);');
  disp('__OK__');
catch err
  printf('__ERR__%s\\n', err.message);
end
'''
    else:
        script = f'''
addpath('{refs_dir}');
rand('state', 1);
X = rand(64, 4, 3);
w = ones(12, 1);
opts.regions = [1 2];
opts.nfreq = 2;
opts.solver = '{solver_name}';
try
  evalc('[L,P,f,idx] = bmd(X, 16, w, 8, 1/16, opts);');
  disp('__OK__');
catch err
  printf('__ERR__%s\\n', err.message);
end
'''
    proc = subprocess.run([exe, '--no-gui', '--quiet', '--eval', script],
                          capture_output=True, text=True, timeout=120)
    out = proc.stdout
    if '__OK__' in out:
        return True, 'ran'
    m = re.search(r'__ERR__(.*)', out)
    return False, (m.group(1).strip() if m else (out + proc.stderr).strip())


# ---------------------------------------------------------------------------
# brute-force numerical radius, for the solver-vs-solver-vs-truth comparison
# ---------------------------------------------------------------------------

def brute_force_radius(A, n_theta=40001, refine=True):
    '''
    Numerical radius by dense search over the rotation angle, optionally
    refined in the winning grid cell.
    '''
    from pybmd.bmd.optimizers import max_fov
    theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    vals = max_fov(A, theta)
    i = int(np.argmax(vals))
    best = float(vals[i])
    if not refine:
        return best

    import scipy.optimize as opt
    step = 2 * np.pi / n_theta
    center = float(theta[i])

    def objective(th):
        return -float(max_fov(A, np.mod(th, 2 * np.pi))[0])

    res = opt.minimize_scalar(
        objective, bounds=(center - step, center + step), method='bounded',
        options={'xatol': 1e-13})
    if res.success:
        best = max(best, -float(res.fun))
    return best


def octave_hamming_window(n):
    '''
    ``bmd.m``'s own internal ``hammwin(N)`` (its default window, used when no
    window is passed), evaluated live under Octave -- not just read off the
    source -- so it can be compared against
    :func:`pybmd.bmd.utils.hamming_window`.
    '''
    exe = require_octave()
    scratch = _scratch_dir()
    out_path = os.path.join(scratch, f'hammwin_{n}.mat')
    script = (f"window = 0.54-0.46*cos(2*pi*(0:{int(n)}-1)/({int(n)}-1))';"
             f"save('-v7','{out_path}','window');")
    proc = subprocess.run([exe, '--no-gui', '--quiet', '--eval', script],
                          capture_output=True, text=True, timeout=60)
    if proc.returncode != 0:
        raise RuntimeError(proc.stdout + proc.stderr)
    with open(out_path, 'rb') as f:
        return scipy.io.loadmat(f)['window'].ravel()
