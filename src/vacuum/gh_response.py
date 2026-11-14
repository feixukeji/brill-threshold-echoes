#!/usr/bin/env python3
"""Linear response of a threshold trajectory to a set of perturbation
directions, all evolved as one GPU batch (Sec. IV of the paper): how
many independent directions grow along the trajectory, read off from
gauge-invariant, trivial-mode-free observables of the central Weyl scalar
(gh_analysis.py growth).

    python src/vacuum/gh_response.py out/gh/bis_gpu/t24d_pair.npz \
        out/gh/bis_gpu/t24d_round1.json --T0 8 --track out/gh/bis_gpu/t24d_round0.json \
        --dir out/gh/bis_gpu/e105_pair.npz --dir out/gh/bis_gpu/e095_pair.npz \
        --dir out/gh/bis_gpu/h085_pair.npz --nrand 3 --eps 1e-8 --Tend 50 --out out/gh/resp/t24d

The base is the midpoint of the flipping pair of the given round, rebuilt at
T0 on the line between the pair checkpoints (the round's segment; at T0 < the
round's restart time the line is the same family to second order in the
pair separation).  T0 must lie on the LU refreeze schedule of the run the
pair comes from, which is also replayed (frame and substeps), so the base
follows the bisected trajectory.  Members: the base; the bracket direction
u_hi - u_lo (the tuned mode) at +eps, -eps and +4 eps (linearity); every
other direction at +eps and -eps.  A direction is either the midpoint of
another bisection's pair at T0 minus the base (another family: for T0 < Ts
every run has the same frame), or a smooth random field (coarse random
nodes, every active variable with its parity, times the variable's scale)
windowed by exp(-(R/rw)^4), or (T0 = 0) a random shape deformation of the
Brill data (gh_data.q_shape: the difference of two exact solutions of the
constraints, so a physical direction).  Every direction is scaled to an RMS of eps in
the variables divided by their scales (gh_linmap.field_scales: the RMS
deviation of the base from flat), over R < rw: the plain norm is 96% P_W
near the axis.  The random fields violate the constraints at O(eps), and
that is not damped away: it grows with the perturbation (at the amplitude
of gh_codim.py's knobs it ends the run).  The Brill shape directions do not
have this problem.  --onesided: the directions other than A at +eps only
(half the members, but the response then keeps the second-order part, which
in the last two echoes is c^2 times that of A for a direction with tuned
content c: as large as the untuned responses at eps = 1e-8).  --qcoef: the
tuned-free combinations of the shape deformations (gh_analysis.py qfree),
which can run at a much larger eps than A (--epsA), so that the untuned
responses stand far above the noise of e_c.

Writes <out>_round0.json (gh_bisect's format; members [(label, outcome,
T_end)], the directions in 'dirs') and <out>_states.npz (every member's
active vector at the multiples of --every; with --axis the axis profiles
eax (refresh, member, node) at the Z nodes zax)."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/


def pair_state(path, T0):
    """u_lo, u_hi of a <out>_pair.npz at T0, with the frame and detector
    state there."""
    from vacuum.gh_analysis import npz_mmap
    z = np.load(path)
    T = z['T']
    i = int(np.argmin(np.abs(T - T0)))
    assert abs(T[i] - T0) < 1e-6, f'{path}: no checkpoint at T = {T0}'
    lo, hi = np.array(npz_mmap(path, 'u_lo')[i]), np.array(npz_mmap(path, 'u_hi')[i])
    fr = tuple(float(z['fr_' + k][i]) for k in ('ell', 'lnL', 'tc', 'c'))
    return lo, hi, fr, float(z['peak'][i]), float(z['Tpk'][i])


def random_field(G, rng, nc, rw):
    """A smooth random active vector (unscaled): every field on nc x nc
    coarse nodes, interpolated with its parity, times exp(-(R/rw)^4)."""
    from vacuum import gh_sim as g
    Gc = g.Grid(nc, nc, G.L, G.ax.a, G.az.a, zsym=G.zsym, b=G.ax.b)
    win = np.exp(-(G.R / rw) ** 4)
    F = [Gc.ax.interp(G.ax.x, g.FPAR[i][0]) @ rng.standard_normal(Gc.shape) @ Gc.az.interp(G.az.x, g.FPAR[i][1]).T
         for i in range(g.NA)]
    return (np.array(F) * win[None]).ravel()


def shape_dirs(G, gh, cfg, A, T0, nq, deg, h, seed, coef=None):
    """The Brill shape directions (name, du) at T0 = 0: (data(q + h dq_k) -
    data(q)) / h for nq deformations gh_data.q_shape of the t24d-type q_holz(A),
    drawn from default_rng((seed, 1)), or (coef: q_poly coefficient matrices,
    gh_analysis.py qfree) dq_k = q_poly(coef[k]), named qf<k>."""
    from vacuum import gh_data as gdat
    from vacuum import gh_sim as g
    assert T0 == 0.0, 'Brill shape directions are initial data: T0 = 0'
    bd = lambda extra=(): g.active(G, gdat.brill_data(G, A, cfg['sigma'], ell=cfg['ell0'], gh=gh, nr=cfg['nr'],
                                                     nmu=cfg['nmu'], extra=extra)[0])
    u0 = bd()
    if coef is not None:
        return [(f'qf{k}', (bd(((h, gdat.q_poly(c, A, cfg['sigma'])),)) - u0) / h) for k, c in enumerate(coef)]
    rng = np.random.default_rng((seed, 1))
    return [(f'bq{k}', (bd(((h, gdat.q_shape(rng, A, cfg['sigma'], 1.0, deg)),)) - u0) / h) for k in range(nq)]


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('pair', help='<out>_pair.npz of the base bisection')
    p.add_argument('round', help='the round whose flipping pair defines the base (its members lie on the pair line)')
    p.add_argument('--T0', type=float, required=True, help='start time (a checkpoint on the LU refreeze schedule)')
    p.add_argument('--track', default='', help='earlier round json(s) of the same run, comma separated (frame and '
                   'substeps before the given round)')
    p.add_argument('--dir', action='append', default=[], help='another bisection\'s <out>_pair.npz: its midpoint '
                   'at T0 minus the base')
    p.add_argument('--nrand', type=int, default=3, help='number of smooth random directions')
    p.add_argument('--nc', type=int, default=10, help='coarse nodes of the random fields')
    p.add_argument('--rw', type=float, default=3.0, help='window radius of the random fields and of the normalization (frame units)')
    p.add_argument('--nq', type=int, default=0, help='number of Brill shape directions (T0 = 0 only)')
    p.add_argument('--qdeg', type=int, default=3, help='degree of the shape polynomial (gh_data.q_shape)')
    p.add_argument('--qh', type=float, default=1e-6, help='finite-difference step of the shape directions')
    p.add_argument('--qsel', default='', metavar='K0:K1', help='only the shape directions K0 <= k < K1 of the '
                   '--nq draws (to split one seed over several batches)')
    p.add_argument('--qcoef', default='', help='gh_analysis.py qfree output: its tuned-free Brill shape '
                   'directions (T0 = 0 only); with --qsel a subset')
    p.add_argument('--onesided', action='store_true', help='directions other than A at +eps only')
    p.add_argument('--epsA', type=float, default=None, help='eps of the tuned direction A (default --eps)')
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--eps', type=float, default=1e-8)
    p.add_argument('--Tend', type=float, default=50.0)
    p.add_argument('--every', type=float, default=10.0, help='state output interval')
    p.add_argument('--axis', default='', metavar='ZMAX,STEP', help='also record E^Z_Z on the axis at every STEP-th '
                   'Z node below ZMAX (frame units) at every refresh, to <out>_states.npz (eax, zax)')
    p.add_argument('--dT', action='store_true', help='add the direction d_T u of the base at T0: a time translation '
                   'with the zoom of the frame, i.e. only the two exact symmetries, so whatever of its response '
                   'survives the projections is numerical (a control)')
    p.add_argument('--set', action='append', default=[], metavar='KEY=VALUE', help='override a CONFIG entry (JSON '
                   'value) for the evolution, e.g. a coarser grid: --set n=320 --set nz=768 --set "regrid=[20, 320, '
                   '8.0]"; the pair states are interpolated to it')
    p.add_argument('--out', required=True)
    a = p.parse_args()
    import warnings
    warnings.simplefilter('ignore')
    from vacuum import gh_sim as g
    from vacuum.gh_analysis import load_round
    from vacuum.gh_bisect import CONFIG, Track, evolve_round, regridder
    from vacuum.gh_cuda import CudaGH
    cfg = dict(CONFIG, ckpt=a.every)
    for kv in a.set:
        k, v = kv.split('=', 1)
        assert k in cfg, f'unknown CONFIG key {k}'
        cfg[k] = json.loads(v)
    G0 = g.grid_from_config(CONFIG)                 # the grid of the pair checkpoints
    G, gh = g.make(cfg)
    same = G.shape == G0.shape and np.array_equal(G.X, G0.X) and np.array_equal(G.Z, G0.Z)
    E = CudaGH(G, gh)
    Tg, n2, ax2 = cfg['regrid']
    G2, gh2 = g.make(dict(cfg, n=n2, nz=G.shape[1], ax=ax2))
    E2 = CudaGH(G2, gh2)
    Ts = cfg['tclock'][0]
    assert a.T0 < Tg, 'T0 after the regrid is not implemented'

    d = load_round(a.round)
    st = [m[1] for m in d['members']]
    i = [k for k in range(len(st) - 1) if st[k] == 'collapse' and st[k + 1] == 'disperse']
    assert len(i) == 1, f'{a.round}: not a single flip'
    s_mid = 0.5 * (d['members'][i[0]][0] + d['members'][i[0] + 1][0])
    s_mid = (s_mid - d['lo']) / (d['hi'] - d['lo'])
    lo, hi, fr0, pk0, tp0 = pair_state(a.pair, a.T0)
    to = (lambda v: v) if same else (lambda v: g.transfer_fields(G0, v.reshape(g.NA, *G0.shape), G, fr0[0]).ravel())
    lo, hi = to(lo), to(hi)
    ub, dA = lo + s_mid * (hi - lo), hi - lo
    z = np.load(a.pair)
    assert abs(float(z['lo']) - d['lo']) < 1e-12 and abs(float(z['hi']) - d['hi']) < 1e-12, 'round not on this pair'
    from vacuum.gh_linmap import field_scales
    sc = field_scales(G, ub, g.active(G, g.flat_state(G, fr0[0])))
    win = np.tile((G.R < a.rw).ravel(), g.NA)
    snorm = lambda du: np.sqrt(np.mean((du / sc)[win] ** 2))
    print(f'base: {a.round} pair midpoint s = {s_mid:.6f} of [{d["lo"]:.15f}, {d["hi"]:.15f}] at T0 = {a.T0}; '
          f'tuned direction u_hi - u_lo: scaled RMS {snorm(dA):.2e}', flush=True)

    dirs = [('A', dA)]
    for f in a.dir:
        l2, h2 = pair_state(f, a.T0)[:2]
        l2, h2 = to(l2), to(h2)
        um = l2 + 0.5 * (h2 - l2)
        dirs.append((os.path.basename(f).replace('_pair.npz', ''), um - ub))
    if a.dT:
        E.set_ell(fr0[0])
        S1 = E.from_active(ub[None])
        E.freeze_lu(S1)
        dirs.append(('dT', E.to_active(E.rhs(S1, E.maximal(S1)))[0]))
    rng = np.random.default_rng(a.seed)
    for k in range(a.nrand):
        dirs.append((f'rand{k}', sc * random_field(G, rng, a.nc, a.rw)))
    if a.nq:
        Amid = d['lo'] + s_mid * (d['hi'] - d['lo'])
        qs = slice(*map(int, a.qsel.split(':'))) if a.qsel else slice(None)
        for name, du in shape_dirs(G, gh, cfg, Amid, a.T0, a.nq, a.qdeg, a.qh, a.seed)[qs]:
            dirs.append((name, du))
    if a.qcoef:
        Amid = d['lo'] + s_mid * (d['hi'] - d['lo'])
        qs = slice(*map(int, a.qsel.split(':'))) if a.qsel else slice(None)
        for name, du in shape_dirs(G, gh, cfg, Amid, a.T0, 0, 0, a.qh, 0, np.load(a.qcoef)['coef'])[qs]:
            dirs.append((name, du))
    labels, U, meta = ['base'], [ub], []
    for name, du in dirs:
        e = a.epsA if name == 'A' and a.epsA else a.eps
        f0 = e / snorm(du)
        meta.append(dict(name=name, norm=float(snorm(du)), scale=float(f0), eps=e))
        for f in ((1.0, -1.0, 4.0) if name == 'A' else (1.0,) if a.onesided else (1.0, -1.0)):
            labels.append(f'{name}{f:+g}')
            U.append(ub + f * f0 * du)
        print(f'  direction {name}: scaled RMS {meta[-1]["norm"]:.3e}', flush=True)
    U = np.array(U)
    print(f'{len(U)} members: ' + ' '.join(labels), flush=True)

    track = Track()
    for fn in [x for x in a.track.split(',') if x] + [a.round]:
        with open(fn) as f:
            track.add(json.load(f)['hist'], Ts)

    def sched(T):                       # the base run's schedule (its round 0 started at T = 0)
        q = (T - (Tg if T >= Tg - 1e-9 else 0.0)) / (cfg['relu'] * cfg['mrefresh'])
        return abs(q - round(q)) < 1e-6
    assert sched(a.T0), f'T0 = {a.T0} is not on the LU refreeze schedule'

    t0 = time.time()
    E.set_ell(fr0[0])
    S = E.from_active(U)
    iz, axis = None, None
    if a.axis:
        zmax, step = a.axis.split(',')
        axis = lambda Gk, zmax=float(zmax), step=int(step): np.flatnonzero(np.abs(Gk.az.x) < zmax)[::step]
        iz = axis(G)
    status, tend, hist, ck, frames = evolve_round(E, S, a.T0, fr0, pk0, tp0, cfg, a.Tend, track, sched,
                                                  [(Tg, E2, regridder(E, E2, cfg))], axis=axis)
    eax = np.array(hist.pop('eax')) if iz is not None else np.zeros(0)
    hist.pop('eax_T', None)
    print(f'done [{time.time() - t0:.0f}s]: ' + ' '.join(f'{l}:{s}' for l, s in zip(labels, status)), flush=True)
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    rec = dict(args=vars(a), template=cfg, lo=d['lo'], hi=d['hi'], T0=a.T0, s_base=s_mid, dirs=meta, eps=a.eps,
               members=[(l, s, float(t) if np.isfinite(t) else a.Tend) for l, s, t in zip(labels, status, tend)],
               hist=hist)
    with open(f'{a.out}_round0.json', 'w') as f:
        json.dump(rec, f)
    np.savez(f'{a.out}_states.npz', labels=np.array(labels), eax=eax, zax=G.az.x[iz] if iz is not None else eax,
             **{f'T{b}': np.array([c[0] for c in ck[b]]) for b in range(len(U))},
             **{f'u{b}': np.array([c[1] for c in ck[b]]) for b in range(len(U))})


if __name__ == '__main__':
    main()
