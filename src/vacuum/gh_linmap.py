#!/usr/bin/env python3
"""Leading eigenvalues of the linearized evolution map over a short time,
from a stored bisection checkpoint (GPU block Arnoldi): which perturbations grow, how fast and where they live.  This is the
test any change of gauge or formulation has to pass before a periodic-orbit
Newton / Floquet computation can work (the zero-shift maximal-lapse gauge
fails it on the L = 12 box: a continuum of gauge modes grows on the axis;
on the production L = 24 box three blocks do not converge, the leading
vectors being P_TT outside the similarity horizon).

    python src/vacuum/gh_linmap.py out/gh/bis_gpu/t24d_pair.npz \
        out/gh/bis_gpu/t24d_round0.json --T 11 --tau 0.5 --n 256 \
        --p 16 --blocks 3 [--frozen] --out out/gh/linmap/t24d_T11

J x = (Phi(v + e sc x) - Phi(v)) / (e |sc x|) in variables scaled per field
(the RMS deviation from flat; W and P_W near the axis, elsewhere as the g_YY
they represent), with Phi the map over tau at the checkpoint's zoom rate:
fixed RK4 steps (0.8 of the CFL limit at v), maximal target re-solved every
mrefresh with the LU frozen at v.  The base and the block of p perturbed
states are evolved as one batch.  --frozen: every perturbed state uses the
background's gauge source (delta H = 0; gh_cuda.CudaGH.evolve_frozen), so
pure-gauge perturbations obey the wave equation instead of the
state-dependent drivers, and the lapse target lA is left out (it is then
passive).  --n: the state interpolated to n X nodes (nz scaled)."""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/

NAMES = ['gTT', 'gTX', 'gTZ', 'gXX', 'gXZ', 'gZZ', 'W', 'PTT', 'PTX', 'PTZ', 'PXX', 'PXZ', 'PZZ', 'PW', 'lA']


def field_scales(G, v, flat):
    """Per-point weights of the active variables: the RMS of each field's
    deviation from flat (floored at 1e-2 of the median), except for W and
    P_W, weighted near the axis by their own RMS and elsewhere as the g_YY
    (P_YY) they represent, min(rms_W, rms_XX / X^2)."""
    from vacuum import gh_sim as g
    dev = (v - flat).reshape(g.NA, G.N)
    rms = np.sqrt(np.mean(dev ** 2, axis=1))
    rms = np.maximum(rms, 1e-2 * np.median(rms))
    sc = np.repeat(rms[:, None], G.N, axis=1)
    X2 = (G.X ** 2).ravel()
    for w, ref in ((6, 3), (13, 10)):              # (W, g_XX), (P_W, P_XX)
        sc[w] = np.minimum(rms[w], rms[ref] / X2)
    return sc.ravel()


def describe(G, y):
    """Dominant fields, peak and 90% extent of an active vector (scaled)."""
    e2f = (y.reshape(len(NAMES), -1) ** 2)
    fe = e2f.sum(axis=1) / e2f.sum()
    e2 = e2f.sum(axis=0)
    pk = int(np.argmax(e2))
    ex = np.cumsum(e2.reshape(G.shape).sum(axis=1)) / e2.sum()
    ez = np.cumsum(e2.reshape(G.shape).sum(axis=0)) / e2.sum()
    x90 = G.ax.x[np.searchsorted(ex, 0.9)]
    z05, z95 = G.az.x[np.searchsorted(ez, 0.05)], G.az.x[min(np.searchsorted(ez, 0.95), G.shape[1] - 1)]
    top = np.argsort(-fe)[:2]
    return (' '.join(f'{NAMES[f]}:{fe[f]:.2f}' for f in top)
            + f' | peak X={G.X.ravel()[pk]:.3f} Z={G.Z.ravel()[pk]:.3f}, 90% in X<{x90:.3f}, Z {z05:.3f}-{z95:.3f}')


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('pair', help='<out>_pair.npz of a bisection')
    p.add_argument('round', help='a round json of the same run (its configuration)')
    p.add_argument('--mem', default='lo', choices=('lo', 'hi'))
    p.add_argument('--T', type=float, required=True, help='checkpoint time')
    p.add_argument('--tau', type=float, default=0.5)
    p.add_argument('--n', type=int, default=0, help='X nodes (default: the checkpoint grid)')
    p.add_argument('--p', type=int, default=16, help='block size')
    p.add_argument('--blocks', type=int, default=3)
    p.add_argument('--eps', type=float, default=1e-6)
    p.add_argument('--nc', type=int, default=10, help='coarse grid of the smooth random start block')
    p.add_argument('--frozen', action='store_true')
    p.add_argument('--out', required=True)
    a = p.parse_args()
    import warnings
    warnings.simplefilter('ignore')
    from vacuum import gh_sim as g
    from vacuum.gh_analysis import Checkpoints
    from vacuum.gh_cuda import CudaGH
    t0 = time.time()
    G, gh, u, Ti, L, tc = Checkpoints(a.pair, a.round).state(a.mem, a.T)
    if a.n and a.n != G.shape[0]:
        cfg = dict(n=G.shape[0], nz=G.shape[1], L=G.L, ax=G.ax.a, az=G.az.a, b=G.ax.b, zsym=int(G.zsym),
                   gamma0=gh.gamma0, g0p=gh.g0p, muL=gh.muL, muS=gh.muS, pL=gh.pL, qL=gh.qL,
                   maximal=gh.maximal[0], cK=gh.maximal[1], mgrid=gh.mgrid, mfixed=gh.mfixed, afloor=gh.afloor)
        G2, gh = g.make(cfg, a.n)
        u, G = g.transfer(G, u, G2), G2
    ell = float(u[-1])
    nspt = int(np.ceil(1.0 / (0.8 * g.cfl_dt(G, u, 1.2))))
    nstep, refresh = max(1, int(round(nspt * a.tau))), max(1, int(round(0.1 * nspt)))
    v0 = g.active(G, u)
    sc = field_scales(G, v0, g.active(G, g.flat_state(G, ell)))
    E = CudaGH(G, gh)
    E.set_ell(ell)
    E.freeze_lu(E.from_active(v0[None]))
    ev = E.evolve_frozen if a.frozen else E.evolve
    keep = np.ones((g.NA, G.N), bool)
    if a.frozen:
        keep[g.NA - 1] = False          # lA is passive under a frozen gauge source
    keep = keep.ravel()
    print(f'checkpoint T = {Ti:.2f} ({a.mem}), tc = {tc:.4f}, grid {G.shape}, ell = {ell:.4f}; tau = {a.tau} in '
          f'{nstep} steps, refresh every {refresh}; {"frozen" if a.frozen else "full"} gauge source', flush=True)

    def apply(X):
        """J on the columns of X (scaled active vectors)."""
        P = X.shape[1]
        es = a.eps * np.linalg.norm(v0 / sc) / np.maximum(np.linalg.norm(X, axis=0), 1e-300)
        batch = np.vstack([v0[None], (v0[:, None] + X * es[None] * sc[:, None]).T])
        out = E.to_active(ev(E.from_active(batch), a.tau, nstep, refresh))
        return ((out[1:] - out[0]) / sc / es[:, None]).T * keep[:, None]

    rng = np.random.default_rng(1)
    Gc = g.Grid(a.nc, a.nc, G.L, G.ax.a, G.az.a, zsym=G.zsym, b=G.ax.b)
    X0 = np.zeros((g.NA * G.N, a.p))
    for j in range(a.p):
        F = np.array([Gc.ax.interp(G.ax.x, g.FPAR[i][0]) @ rng.standard_normal(Gc.shape)
                      @ Gc.az.interp(G.az.x, g.FPAR[i][1]).T for i in range(g.NA)])
        X0[:, j] = F.ravel() * keep
    Q = [np.linalg.qr(X0)[0]]
    P, K = a.p, a.blocks
    H = np.zeros(((K + 1) * P, K * P))
    for k in range(K):
        t1 = time.time()
        W = apply(Q[k])
        for _ in range(2):
            for j in range(k + 1):
                h = Q[j].T @ W
                H[j * P:(j + 1) * P, k * P:(k + 1) * P] += h
                W -= Q[j] @ h
        Qn, R = np.linalg.qr(W)
        H[(k + 1) * P:(k + 2) * P, k * P:(k + 1) * P] = R
        Q.append(Qn)
        n_ = (k + 1) * P
        vals, Y = np.linalg.eig(H[:n_, :n_])
        res = np.linalg.norm(R @ Y[-P:, :], axis=0)
        o = np.argsort(-np.abs(vals))
        Qm = np.concatenate(Q[:k + 1], axis=1)
        print(f'block {k + 1}/{K} [{time.time() - t1:.0f}s, total {time.time() - t0:.0f}s]: leading Ritz values',
              flush=True)
        for i in o[:min(n_, P)]:
            mu = vals[i]
            print(f'  {mu.real:+9.4f} {mu.imag:+9.4f}i  |mu| {abs(mu):8.4f}  rate ln|mu|/tau {np.log(abs(mu)) / a.tau:+7.3f}  '
                  f'res {res[i]:.1e} | ' + describe(G, np.real(Qm @ Y[:, i])), flush=True)
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    np.savez(a.out + '.npz', vals=vals[o], vecs=(Qm @ Y[:, o[:P]]) * sc[:, None], res=res[o], tau=a.tau, T=Ti,
             shape=np.array(G.shape))


if __name__ == '__main__':
    main()
