% Thin driver for cbmd.m / cbmd_instr.m, used by octave_ref.py.
%
% Parameterized entirely via environment variables, so this file is static
% and never templated:
%   PYBMD_OCT_ADDPATH  directory to addpath -- refs/bmd, or the scratch
%                       directory holding the instrumented cbmd_instr.m
%   PYBMD_OCT_FUNC      'cbmd' (unmodified reference) or 'cbmd_instr'
%                       (also returns Q_hat and every per-triad B)
%   PYBMD_OCT_IN        input .mat: X (nt, nVar, *xshape), window, weight
%                       (purely spatial, no variable axis), nOvlp, dt,
%                       regions, nfreq (empty means unset), tol, nitmax,
%                       solver, s_idx, qr_idx (both 1-based, MATLAB layout)
%   PYBMD_OCT_OUT       output .mat path

addpath(getenv('PYBMD_OCT_ADDPATH'));
d = load(getenv('PYBMD_OCT_IN'));

opts.regions = double(d.regions(:).');
if isfield(d, 'nfreq') && numel(d.nfreq) > 0
    opts.nfreq = double(d.nfreq(1));
end
opts.tol     = double(d.tol(1));
opts.nitmax  = double(d.nitmax(1));
opts.solver  = strtrim(char(d.solver));
opts.s_idx   = double(d.s_idx);
opts.qr_idx  = double(d.qr_idx);

func_name = getenv('PYBMD_OCT_FUNC');
X = double(d.X);
window = double(d.window(:));
weight = double(d.weight(:));
nOvlp = double(d.nOvlp(1));
dt = double(d.dt(1));

if strcmp(func_name, 'cbmd')
    [L, P, f, idx, T] = cbmd(X, window, weight, nOvlp, dt, opts);
    save('-v7', getenv('PYBMD_OCT_OUT'), 'L', 'P', 'f', 'idx', 'T');
elseif strcmp(func_name, 'cbmd_instr')
    [L, P, f, idx, T, Q_hat, B_all] = cbmd_instr(X, window, weight, nOvlp, dt, opts);
    save('-v7', getenv('PYBMD_OCT_OUT'), 'L', 'P', 'f', 'idx', 'T', 'Q_hat', 'B_all');
else
    error('run_cbmd:unknown_func', 'unknown PYBMD_OCT_FUNC %s', func_name);
end
