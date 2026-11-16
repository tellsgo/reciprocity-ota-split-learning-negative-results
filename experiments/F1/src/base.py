"""Frozen G10 experiment. All training imports and writes are local to this run."""
import os
for name in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[name] = '1'
import sys, pathlib, json, time, hashlib, traceback, resource, signal, platform
sys.dont_write_bytecode = True
import numpy as np
from scipy.special import j0, jn_zeros
import learning_core as K
R = pathlib.Path(__file__).resolve().parents[1]
C = K.C
N = C['nr']
SEEDS = C['seeds']
METHODS = C['methods']
resource.setrlimit(resource.RLIMIT_CPU, (900, 900))
signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('wall budget 1800s')))
signal.alarm(1800)

def mm(a, b):
    return np.einsum('ij,jk->ik', a, b, optimize=False)

def dump(p, x):
    p.write_text(json.dumps(x, indent=2, allow_nan=False))

def pack(z):
    return {'real': np.asarray(z).real.tolist(), 'imag': np.asarray(z).imag.tolist()}

def unpack(z):
    return np.asarray(z['real']) + 1j * np.asarray(z['imag'])

def stats(v):
    v = np.asarray(v, float)
    assert v.shape == (8,) and np.isfinite(v).all()
    ix = np.random.default_rng(C['bootstrap_seed']).integers(0, 8, (C['bootstrap_replicates'], 8))
    return {'mean': float(v.mean()), 'ci95': np.quantile(v[ix].mean(1), [0.025, 0.975]).tolist(), 'min': float(v.min()), 'max': float(v.max()), 'per_seed': v.tolist()}

def shift(method):
    if method == 'S0':
        return np.eye(N, dtype=complex)
    m = np.zeros((N, N), complex)
    (dx, dy) = (1, 0) if method == 'S10' else (0, 1)
    for x in range(3):
        for y in range(3):
            if x + dx < 3 and y + dy < 3:
                m[3 * (x + dx) + y + dy, 3 * x + y] = 1
    return m

def geo():
    pos = np.array([(x * 0.5, y * 0.5) for x in range(3) for y in range(3)])
    theta = np.arange(128) * 2 * np.pi / 128
    uv = np.array([np.cos(theta), np.sin(theta)])
    s0 = np.exp(mm(-2j * np.pi * pos, uv)) / np.sqrt(128)
    st = s0 * np.exp(2j * np.pi * 0.383 * np.sin(theta))
    r = mm(s0, s0.conj().T)
    t = mm(st, s0.conj().T)
    a = np.linalg.solve(r.T, t.T).T
    q = r - mm(mm(a, r), a.conj().T)
    q = (q + q.conj().T) / 2

    def root(m):
        (eig, v) = np.linalg.eigh(m)
        assert eig.min() > -1e-09
        return mm(v * np.sqrt(np.maximum(eig, 0)), v.conj().T)
    rt = root(r)
    qr = root(q)
    jr = np.block([[rt, np.zeros((N, N))], [mm(a, rt), qr]])
    return dict(pos=pos, s0=s0, st=st, r0=r, rt=t, a=a, q=q, root=rt, qr=qr, jr=jr)

class World(K.World):

    def __init__(self, seed, g):
        rng = np.random.default_rng(C['train_seed_offset'] + seed)
        n = C['rf_observation_quota']
        self.inputs = rng.choice([-1.0, 1.0], (n, 3))
        self.labels = (self.inputs[:, 0] * self.inputs[:, 1] > 0).astype(float)
        both = np.einsum('ij,bjn->bin', g['jr'], K.cn(rng, (n, 2 * N, 2)))
        self._f = both[:, :N]
        self._b = both[:, N:]
        self._nf = C['forward_noise_std'] * K.cn(rng, (n, N))
        self._nb = C['reverse_noise_std'] * K.cn(rng, (n, 2))
        self._np = C['pilot_noise_std'] * K.cn(rng, (n, N, 2))
        crng = np.random.default_rng(C['calibration_seed_offset'] + seed)
        k = 64
        pair = np.einsum('ij,bjn->bin', g['jr'], K.cn(crng, (k, 2 * N, 2)))
        self.cal_fhat = pair[:, :N] + C['pilot_noise_std'] * K.cn(crng, (k, N, 2))
        self.cal_bhat = pair[:, N:] + C['pilot_noise_std'] * K.cn(crng, (k, N, 2))
        trng = np.random.default_rng(C['test_seed_offset'] + seed)
        n = C['test_samples']
        self.test_inputs = trng.choice([-1.0, 1.0], (n, 3))
        self.test_labels = (self.test_inputs[:, 0] * self.test_inputs[:, 1] > 0).astype(float)
        self._test_f = np.einsum('ij,bjn->bin', g['root'], K.cn(trng, (n, N, 2)))
        self._test_n = C['forward_noise_std'] * K.cn(trng, (n, N))
        self.test_epsilon = K.cn(trng, (n, 2))

def qcomplex(x, bound):
    half = 2 ** (C['control_bits_per_real'] - 1) - 1

    def q(v):
        return np.rint(np.clip(v, -bound, bound) / bound * half) / half * bound
    clips = int(np.count_nonzero(abs(x.real) > bound) + np.count_nonzero(abs(x.imag) > bound))
    return (q(x.real) + 1j * q(x.imag), clips)

def estimate(w):
    (f, clips) = qcomplex(w.cal_fhat, 4)
    cross = np.einsum('bmn,bln->ml', w.cal_bhat, f.conj()) / 128
    r = np.einsum('bmn,bln->ml', f, f.conj()) / 128 - 0.01 * np.eye(N)
    r = (r + r.conj().T) / 2 + 0.02 * np.eye(N)
    (eig, v) = np.linalg.eigh(r)
    c = mm(v * np.maximum(eig, 0.02), v.conj().T)
    a = np.linalg.solve(c.T, cross.T).T
    (u, _, vh) = np.linalg.svd(a.conj().T)
    exact = mm(vh.conj().T, u.conj().T)
    (mq, mc) = qcomplex(exact, 32)
    (u, _, vh) = np.linalg.svd(mq)
    m = mm(u, vh)
    return (a, m, 1.0, {'A': pack(a), 'M': pack(m), 'Creg': pack(c), 'C10hat': pack(cross), 'channel_clips': clips, 'matrix_clips': mc, 'covariance_floor_count': int(np.count_nonzero(eig < 0.02)), 'M_unitarity_error': float(np.linalg.norm(mm(m.conj().T, m) - np.eye(N))), 'quantization_error_fro': float(np.linalg.norm(m - exact))})
