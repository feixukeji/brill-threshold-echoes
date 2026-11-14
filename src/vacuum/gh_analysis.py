#!/usr/bin/env python3
"""Post-processing of the threshold bisections and of the responses along
a threshold trajectory (gh_bisect.py, gh_response.py outputs).

    python src/vacuum/gh_analysis.py brackets out/gh/bis_gpu/h085
    python src/vacuum/gh_analysis.py central  out/gh/bis_gpu/h085_round3.json:7,8
    python src/vacuum/gh_analysis.py gamma    out/gh/bis_gpu/e105 out/gh/bis_gpu/e095
    python src/vacuum/gh_analysis.py response out/gh/bis_gpu/e105r_round0.json:8 \
        out/gh/bis_gpu/pert_p2_round1.json:pair out/gh/bis_gpu/pert_m2_round1.json:pair 0.1
    python src/vacuum/gh_analysis.py lobes    <round.json>:m ...
    python src/vacuum/gh_analysis.py growth   out/gh/resp/t24d_T8f_round0.json out/gh/resp/t24d_T8rf_round0.json [--lobes t0,t1,...]
    python src/vacuum/gh_analysis.py qfree    out/gh/resp/t24d_T0q{1a,1b,2a,2b}_round0.json out/gh/resp/qfree.npz
    python src/vacuum/gh_analysis.py gram     out/gh/resp/t24d_T0qf{a,b}f_round0.json out/gh/resp/t24d_T0qf_gram.npz
    python src/vacuum/gh_analysis.py growth   out/gh/resp/t24d_T0qf{a,b}4f_round0.json --gram out/gh/resp/t24d_T0qf_gram.npz \
        [--cmp out/gh/resp/t24d_T0qfaf_round0.json,out/gh/resp/t24d_T0qfbf_round0.json]
    python src/vacuum/gh_analysis.py axis     out/gh/resp/t24d_T8{axf,dTf}_round0.json   (batches written with --axis)
    python src/vacuum/gh_analysis.py departure out/gh/bis_gpu/t24d_round1.json out/gh/bis_gpu/t24d_round0.json
    python src/vacuum/gh_analysis.py curvature <pair.npz> <round.json> lo|hi T1,T2,...
    python src/vacuum/gh_analysis.py horizons  <pair.npz> <round.json> lo|hi T1,T2,...
    python src/vacuum/gh_analysis.py weyl      <pair.npz> <round.json> T1,T2,...
    python src/vacuum/gh_analysis.py combos    out/gh/resp/t24d_T0qf{a,b}4f_round0.json --gram out/gh/resp/t24d_T0qf_gram.npz 2 <out.npz>
    python src/vacuum/gh_analysis.py weylresp  <out>_states.npz <out>_round0.json
    python src/vacuum/gh_analysis.py imax      out/gh/thr/imaxf_round0.json out/gh/thr/imaxn_round0.json
    python src/vacuum/gh_analysis.py lobe4     out/gh/bis_gpu/t24d_round{0,1}.json out/gh/thr/scan{c,d}_round0.json
    python src/vacuum/gh_analysis.py decide    <A*> out/gh/bis_res/{nb600,noc600,dg600,dh600}_round0.json
    python src/vacuum/gh_analysis.py ricci     <save-T npz | members npz | pair npz> <round.json> <members> [T,...]
    python src/vacuum/gh_analysis.py axisah    <members npz | save-T npz> <round.json> <members> T1,T2,...
    python src/vacuum/gh_analysis.py constraint <save-T npz> <round.json> <members>
        (curvature, horizons: also <out>_members.npz of a --members run, member by index)

The central Weyl scalar.  The centre is a fixed point of both symmetries,
so there E = diag(-e/2, -e/2, e) and B = 0 (I = 12 e^2), and e_c = E^Z_Z
along the central worldline is a completely gauge-invariant function of the
member's own central proper time tc = int L alpha_c dT.  Scaled form:
y = e_c rem^2 against x = -ln rem, rem = tc* - tc (tc* the focus of the
tclock frame).  A member is given as <round.json>:m (m the member index).

Trivial modes of a response along the threshold: a time translation of the
solution changes y by d g, g = -(y0_x + 2 y0) e^x (kappa = 1), a shift of the
phase / scale by p y0_x (kappa = 0); both are exact symmetries, so p and d
are constants.  Lobe invariants: with t_n, e_n the extrema of e_c(tc) and
tau_n = t_{n+1} - t_n, E_n = e_n tau_n tau_{n-1} is invariant under time
translation and scaling."""
from __future__ import annotations

import glob
import json
import os
import re
import sys
import time
import zipfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/


# ----------------------------------------------------------------------------
# reading
# ----------------------------------------------------------------------------
def load_round(path):
    """A round json with d['cfg'] the configuration, regrid and tclock
    included (rounds written before gh_bisect.CONFIG carried them have them
    as strings in args)."""
    with open(path) as f:
        d = json.load(f)
    cfg = dict(d['template'])
    for k in ('regrid', 'tclock'):
        if k not in cfg:
            v = d['args'].get(k) or ''
            cfg[k] = [float(x) for x in v.split(',')] if v else None
    cfg.setdefault('mfixed', 40)        # (always 40; not in the configuration before)
    d['cfg'] = cfg
    return d


def rounds(prefix):
    """The round jsons <prefix>_round<k>.json in order."""
    fs = glob.glob(f'{prefix}_round*.json')
    return [load_round(f) for f in sorted(fs, key=lambda f: int(re.search(r'_round(\d+)\.json$', f).group(1)))]


def accepted(d):
    """Whether the round's flip was accepted as the new bracket (its width
    then shrank by B + 1; a round stopped by cbad or by several flips keeps
    the old width)."""
    w, B = d.get('w'), len(d['members'])
    return w is not None and abs(w * (B + 1) / (d['hi'] - d['lo']) - 1) < 1e-6


def final_bracket(ds):
    """(A*, width) of the last accepted round (midpoint of the flipping
    pair), or (None, None)."""
    for d in ds[::-1]:
        if accepted(d):
            M = d['members']
            A = [d['lo']] + [m[0] for m in M] + [d['hi']]
            st = ['collapse'] + [m[1] for m in M] + ['disperse']
            i = [i for i in range(len(st) - 1) if st[i] == 'collapse' and st[i + 1] == 'disperse'][0]
            return 0.5 * (A[i] + A[i + 1]), abs(A[i + 1] - A[i])
    return None, None


def central(d, m):
    """(tc, e_c, T) of member m of round d: its own central proper time
    (trapezoid of L alpha_c from the round's common tc at T0) and the
    physical central E^Z_Z, up to its decision; from the substep samples
    when the round has them (efine).  The first sample (the restart state,
    before the first step) is dropped."""
    h = d['hist']
    if 'fT' in h:
        T = np.r_[h['T'][0], h['fT']]
        L = np.exp(np.r_[h['lnL'][0], h['flnL']])
        a0 = np.r_[h['a0'][0][m], np.array(h['fa0'], dtype=float)[:, m]]
        ez = np.r_[h['ez'][0][m], np.array(h['fez'], dtype=float)[:, m]]
    else:
        T = np.array(h['T'])
        L = np.exp(np.array(h['lnL']))
        a0 = np.array(h['a0'])[:, m]
        ez = np.array(h['ez'], dtype=float)[:, m]
    tc = h['tc'][0] + np.concatenate([[0.0], np.cumsum(0.5 * (L[1:] * a0[1:] + L[:-1] * a0[:-1]) * np.diff(T))])
    e = ez / L ** 2
    k = (T <= d['members'][m][2] + 1e-9) & np.isfinite(e)
    k[0] = False
    return tc[k], e[k], T[k]


def member(spec):
    """'<round.json>:m' -> (round dict, m); '<round.json>:a,b' or
    '<round.json>:pair' (the members on either side of the flip) ->
    (round dict, [a, b])."""
    f, ms = spec.rsplit(':', 1)
    d = load_round(f)
    if ms == 'pair':
        st = [m[1] for m in d['members']]
        i = [i for i in range(len(st) - 1) if st[i] == 'collapse' and st[i + 1] == 'disperse'][0]
        return d, [i, i + 1]
    ms = [int(x) for x in ms.split(',')]
    return d, (ms if len(ms) > 1 else ms[0])


def tcstar(d):
    return d['cfg']['tclock'][1]


def npz_mmap(path, key):
    """Memory map of one array of an uncompressed npz (np.savez): the pair
    files are several GB, and np.load decompresses a whole array per access."""
    zf = zipfile.ZipFile(path)
    info = zf.getinfo(key + '.npy')
    assert info.compress_type == 0, 'compressed member'
    with open(path, 'rb') as f:
        f.seek(info.header_offset)
        h = f.read(30)
        n, m = int.from_bytes(h[26:28], 'little'), int.from_bytes(h[28:30], 'little')
        f.seek(info.header_offset + 30 + n + m)
        ver = np.lib.format.read_magic(f)
        shape, fortran, dtype = (np.lib.format.read_array_header_1_0(f) if ver == (1, 0)
                                 else np.lib.format.read_array_header_2_0(f))
        off = f.tell()
    return np.memmap(path, dtype=dtype, mode='r', offset=off, shape=shape, order='F' if fortran else 'C')


class Checkpoints:
    """The checkpoints of a <out>_pair.npz (members 'lo', 'hi') or of a
    --members <out>_members.npz (members by index) as gh_sim states, on the
    grid of their time (the regrid of the round's configuration)."""

    def __init__(self, pair, round_json):
        from vacuum import gh_sim as g
        self.g = g
        self.path = pair
        cfg = load_round(round_json)['cfg']
        self.grids = [g.make(cfg)]
        self.Tg = np.inf
        if cfg.get('regrid'):
            self.Tg, n2, ax2 = cfg['regrid'][:3]
            G1 = self.grids[0][0]
            self.grids.append(g.make(dict(cfg, n=int(n2), nz=G1.shape[1], ax=ax2)))
            if cfg.get('regrid2'):
                T2, n3, nz3 = cfg['regrid2']
                self.grids.append(g.make(dict(cfg, n=int(n3), nz=int(nz3), ax=ax2, **(cfg.get('gauge2') or {}))))
        names = zipfile.ZipFile(pair).namelist()
        self.single = 'shape.npy' in names          # a --save-T file: every member at one time
        self.members = 'amps.npy' in names and not self.single
        if self.single:
            z = np.load(pair)
            self.T = np.array([float(z['T'])])
            self.fr = {k: np.array([float(z['fr'][j])]) for j, k in enumerate(('ell', 'lnL', 'tc'))}
            return
        tkey = 'frT' if self.members else 'T'
        self.T = np.array(npz_mmap(pair, tkey))
        self.fr = {k: np.array(npz_mmap(pair, 'fr_' + k)) for k in ('ell', 'lnL', 'tc')}

    def state(self, mem, T):
        """(G, gh, u, T_i, L, tc) of member 'lo' or 'hi' (pair file) or of
        member index mem (members file) at its checkpoint nearest T."""
        if self.single:
            v = np.asarray(npz_mmap(self.path, 'u')[int(mem)])
            G, gh = next(p for p in self.grids if self.g.NA * p[0].N == v.size)
            Ti, j = float(self.T[0]), 0
        elif self.members:
            Tm = np.array(npz_mmap(self.path, f'T{int(mem)}'))
            i = int(np.argmin(np.abs(Tm - T)))
            Ti = float(Tm[i])
            v = np.asarray(npz_mmap(self.path, f'u{int(mem)}')[i])
            j = int(np.argmin(np.abs(self.T - Ti)))
            G, gh = next(p for p in self.grids if self.g.NA * p[0].N == v.size)
        else:
            i = j = int(np.argmin(np.abs(self.T - T)))
            Ti = float(self.T[i])
            G, gh = self.grids[1] if Ti >= self.Tg - 1e-9 else self.grids[0]
            v = np.asarray(npz_mmap(self.path, 'u_' + mem)[i])
        u = self.g.full_state(G, v, float(self.fr['ell'][j]))
        return G, gh, u, Ti, float(np.exp(self.fr['lnL'][j])), float(self.fr['tc'][j])


# ----------------------------------------------------------------------------
# commands
# ----------------------------------------------------------------------------
def cmd_brackets(prefixes):
    """Per round: start time, bracket, width, outcomes; the final A*."""
    for pre in prefixes:
        ds = rounds(pre)
        print(f'{pre} (eta {ds[0]["args"].get("eta", 1.0)}):')
        for k, d in enumerate(ds):
            st = ''.join(m[1][0] for m in d['members'])
            dec = [m[2] for m in d['members'] if m[1] != 'undecided']
            w = d.get('w', np.nan)
            print(f'   round {k}: T0 {d.get("T0", 0.0):5.1f}  [{d["lo"]:.12f}, {d["hi"]:.12f}]  {st}  '
                  f'T_end {min(dec, default=np.nan):.1f}-{max(dec, default=np.nan):.1f}  '
                  + (f'-> ln w {np.log(w):.2f}' if accepted(d) else 'not accepted'))
        As, w = final_bracket(ds)
        print(f'   A* = {As:.10f} +- {w / 2:.1e}' if As is not None else '   no single-flip round')


def crossings(tc, e):
    i = np.where(e[:-1] * e[1:] < 0)[0]
    return tc[i] - e[i] * (tc[i + 1] - tc[i]) / (e[i + 1] - e[i])


def extrema(tc, y, ymin=0.0):
    """Local extrema (t, value) of y(tc), parabolic refinement, |value| > ymin."""
    out = []
    for i in range(1, len(y) - 1):
        if (y[i] - y[i - 1]) * (y[i + 1] - y[i]) < 0:
            a, b, c = y[i - 1], y[i], y[i + 1]
            den = a - 2 * b + c
            p = 0.5 * (a - c) / den if den != 0 else 0.0
            v = b - 0.25 * (a - c) * p
            if abs(v) > ymin:
                out.append((tc[i] + p * 0.5 * (tc[i + 1] - tc[i - 1]), v))
    return np.array(out).reshape(-1, 2)


def cmd_central(specs):
    """Zero crossings of e_c(tc) and the extrema of y = e_c rem^2."""
    for spec in specs:
        d, ms = member(spec)
        for m in np.atleast_1d(ms):
            tc, e, T = central(d, m)
            rem = tcstar(d) - tc
            ok = rem > 0
            A, st, te = d['members'][m]
            print(f'{spec.rsplit(":", 1)[0]}:{m}  A {A:.15f} {st} at T {te:.1f}, tc {tc[-1]:.4f}')
            print('   zero crossings tc: ' + ' '.join(f'{t:.4f}' for t in crossings(tc, e)))
            ex = extrema(tc[ok], (e * rem ** 2)[ok], 5.0)
            print('   extrema of y (tc:x:y): ' + ' '.join(f'{t:.4f}:{-np.log(tcstar(d) - t):.2f}:{v:+.0f}' for t, v in ex))


def cmd_gamma(prefixes, ranges=((-25.0, -5.0), (-25.0, -9.0), (-25.0, -11.0))):
    """The tuned mode from the curvature maxima of the dispersing members:
    ln max(K_ij K^ij / L^2) = c - 2 gamma ln|A - A*| (A* the final bracket;
    members closer than twice its width left out), per family and range of
    ln|A - A*|."""
    for pre in prefixes:
        ds = rounds(pre)
        As, w = final_bracket(ds)
        pts = []
        for d in ds:
            h = d['hist']
            T, L, KK = np.array(h['T']), np.exp(np.array(h['lnL'])), np.array(h['KK'])
            for m, (A, st, te) in enumerate(d['members']):
                if st == 'disperse' and abs(A - As) > 2 * w:
                    k = T <= te + 1e-9
                    pts.append((np.log(abs(A - As)), np.log(np.max(KK[k, m] / L[k] ** 2))))
        pts = np.array(sorted(pts))
        row = []
        for lo, hi in ranges:
            s = (pts[:, 0] > lo) & (pts[:, 0] < hi)
            if s.sum() > 3:
                c = np.polyfit(pts[s, 0], pts[s, 1], 1)
                row.append(f'({lo:.0f}, {hi:.0f}): {s.sum():2d} pts, gamma {-c[0] / 2:.3f}')
        print(f'{pre}: A* {As:.12f}, ln|A - A*| in [{pts[0, 0]:.1f}, {pts[-1, 0]:.1f}]; ' + '; '.join(row))
        print('   ' + ' '.join(f'{a:.1f}:{b:.2f}' for a, b in pts[::max(1, len(pts) // 24)]))


def response(bspec, pspec, mspec, eps, dx=0.002, x0=0.5):
    """x, the base y0 and the antisymmetric response R = (y_+ - y_-) / (2 eps)
    of the +-eps tuned pairs (mean over the four combinations), its half
    spread and the symmetric part ((y_+ + y_-) / 2 - y0) / |eps|, on a
    common x grid up to the shortest member."""
    db, mb = member(bspec)
    (dp, mp), (dm, mm) = member(pspec), member(mspec)
    tcs = tcstar(db)
    base = central(db, mb)[:2]
    P = [central(dp, m)[:2] for m in mp]
    M = [central(dm, m)[:2] for m in mm]
    t1 = min([tcs - 0.01, base[0][-1]] + [r[0][-1] for r in P + M])
    x = np.arange(x0, -np.log(tcs - t1), dx)
    tg, rem = tcs - np.exp(-x), np.exp(-x)
    Y = lambda r: np.interp(tg, *r) * rem ** 2
    y0 = Y(base)
    Rs = [(Y(a) - Y(b)) / (2 * eps) for a in P for b in M]
    Ss = [((Y(a) + Y(b)) / 2 - y0) / abs(eps) for a in P for b in M]
    return x, y0, np.mean(Rs, 0), np.ptp(Rs, 0) / 2, np.mean(Ss, 0)


def _support(A, nang=720):
    """Support function h(theta) of the polygon {c : |A c|_inf <= 1} (2-D)."""
    from scipy.optimize import linprog
    th = np.linspace(0, np.pi, nang, endpoint=False)
    hs = []
    for t in th:
        r = linprog(-np.array([np.cos(t), np.sin(t)]), A_ub=np.vstack([A, -A]), b_ub=np.ones(2 * len(A)),
                    bounds=[(None, None)] * 2, method='highs')
        hs.append(-r.fun)
    return th, np.array(hs)


def cmd_response(bspec, pspec, mspec, eps, win=(0.56, 0.68), period=None):
    """The linear response along the threshold and its trivial part.

    The early window, where the perturbation has not reached the centre,
    bounds the trivial coefficients: with |R| <= r on it and the physical
    part not exactly cancelling, (p, d) lies in {|p y0_x + d g| <= r}, a
    polygon; its support gives |p|, |d| and, at every later x, the largest
    trivial part |p y0_x + d g| (columns 'triv')."""
    x, y0, R, band, S = response(bspec, pspec, mspec, eps)
    yx = np.gradient(y0, x)
    g = -(yx + 2 * y0) * np.exp(x)
    kw = (x >= win[0]) & (x <= win[1])
    r = np.abs(R[kw]).max()
    A = np.stack([yx[kw], g[kw]], 1) / r
    th, h = _support(A)
    pmax, dmax = h[0], h[len(th) // 2]
    ang = np.arctan2(g, yx) % np.pi
    triv = np.hypot(yx, g) * np.interp(ang, th, h, period=np.pi)
    print(f'early window x in {win}: max|R| = {r:.2f}  ->  |p| <= {pmax:.1e}, |d| <= {dmax:.1e}')
    print('   x     |R|   band  |sym|  triv  |y0|')
    for xa in np.arange(np.floor(x[0] * 20) / 20, x[-1], 0.05):
        k = (x >= xa) & (x < xa + 0.05)
        if k.sum() > 2:
            mx = lambda f: np.abs(f[k]).max()
            print(f'  {xa:4.2f} {mx(R):6.1f} {mx(band):5.1f} {mx(S):6.1f} {mx(triv):5.1f} {mx(y0):6.1f}')
    if period:
        # growth per period: the ratio of |R| maxima (and of |y0|) one period apart
        print(f'growth over one period Delta = {period}: x, |R|(x + Delta) / |R|(x), |y0|(x + Delta) / |y0|(x)')
        env = lambda f, xa: np.abs(f[(x >= xa - 0.05) & (x < xa + 0.05)]).max()
        for xa in np.arange(0.75, x[-1] - period - 0.05, 0.05):
            print(f'  {xa:4.2f}  {env(R, xa + period) / env(R, xa):5.2f}  {env(y0, xa + period) / env(y0, xa):5.2f}')


def invariants(tc, e, tcs, ymin=1.0):
    """Extrema t_n, e_n of e_c(tc) (|e rem^2| > ymin), tau_n, the lobe
    invariants E_n = e_n tau_n tau_{n-1} and the ratios tau_{n+1} / tau_n."""
    ex = extrema(tc, e)
    ex = ex[np.abs(ex[:, 1]) * (tcs - ex[:, 0]) ** 2 > ymin]
    t, v = ex[:, 0], ex[:, 1]
    tau = np.diff(t)
    return t, v, v[1:-1] * tau[1:] * tau[:-1], tau[1:] / tau[:-1]


def cmd_lobes(specs):
    for spec in specs:
        d, m = member(spec)
        tcs = tcstar(d)
        t, v, E, rr = invariants(*central(d, m)[:2], tcs)
        print(f'{spec}:')
        print('    n    t_n     x_n        e_n          E_n    tau ratio')
        for k in range(len(E)):
            print(f'  {k:3d} {t[k + 1]:7.4f} {-np.log(tcs - t[k + 1]):6.3f} {v[k + 1]:11.1f} {E[k]:12.3f}  {rr[k]:7.3f}')


def response_members(rjs):
    """gh_response.py batches with one base (equal up to roundoff, which
    depends on the batch size; directions repeated across batches are
    taken once): the first batch's base tc samples (tc < tc*, the tclock
    focus, only to stop at the frame's end) on the span common to all
    batches, its e_c there, and per direction the antisymmetric and
    symmetric responses (e_+ - e_-) / (2 eps), (e_+ + e_- - 2 e_0) / (2 eps)
    in e units, e_0 the base of the direction's own batch ((e_+ - e_0) / eps
    and zero for --onesided batches; eps per direction (--epsA); every
    member's e_c a cubic spline of its own samples, read on a uniform grid in
    tc, so that sums over it are integrals d tc), A+4 against A+1
    (linearity of the tuned probe) and the batch json of each direction."""
    from scipy.interpolate import CubicSpline
    names, R, S, src, lin = [], [], [], [], None
    ds = [load_round(rj) for rj in rjs]
    bases = []
    for d in ds:
        t, e = central(d, [m[0] for m in d['members']].index('base'))[:2]
        bases.append((t[t < tcstar(d)], e[t < tcstar(d)]))
    tb = bases[0][0][bases[0][0] <= min(t[-1] for t, _ in bases)]
    # a uniform grid in central proper time: equal weights are then the d tau_c integral (the samples, every
    # 16 substeps, crowd where the central lapse is small)
    tb = np.linspace(tb[0], tb[-1], len(tb))
    e0 = eb0 = CubicSpline(*bases[0])(tb)
    for rj, (t, e) in zip(rjs, bases):
        de = np.abs(CubicSpline(t, e)(tb) - e0).max() / np.abs(e0).max()
        assert de < 1e-4, f'{rj}: another base ({de:.1e})'
    for rj, d in zip(rjs, ds):
        labs = [m[0] for m in d['members']]
        ec = {lab: CubicSpline(*central(d, m)[:2])(tb) for m, lab in enumerate(labs)}
        e0 = ec['base']
        for m in d['dirs']:
            n, eps = m['name'], m.get('eps', d['eps'])
            key = f'{n}:{d["args"]["seed"]}' if n.startswith(('rand', 'bq')) else n
            if key in names:
                continue
            two = n + '-1' in ec                # --onesided batches: +eps only
            r = (ec[n + '+1'] - ec[n + '-1']) / (2 * eps) if two else (ec[n + '+1'] - e0) / eps
            names.append(key)
            R.append(r)
            S.append((ec[n + '+1'] + ec[n + '-1'] - 2 * e0) / (2 * eps) if two else np.zeros_like(e0))
            src.append(rj)
            if n == 'A':
                lin = ((ec['A+4'] - e0) / 4 - (ec['A+1'] - e0)) / eps
    return names, tb, eb0, np.array(R).T, np.array(S).T, lin


def scaling_coef(rjs, names):
    """The exact change of scale of the Brill data, q(x) -> q(x / lambda), as
    coefficients over names (directions of unit RMS): dq = A x 2(1 - x - y)
    e^(-x-y) lies in the cubic shape space of --qcoef batches (d lambda = -1
    per unit coefficient), so its response must be the scaling mode alone.
    None unless the batches hold A and every qf direction."""
    ds = [load_round(rj) for rj in rjs]
    cf = {d['args'].get('qcoef') for d in ds} - {None, ''}
    if len(cf) != 1:
        return None
    qf = np.load(cf.pop())
    if "mono" not in qf:                     # combos files: no shape space
        return None
    mono = [tuple(m) for m in qf['mono']]
    keys = ['A'] + [f'qf{k}' for k in range(len(qf['coef']))]
    if not all(k in names for k in keys) or (0, 1) not in mono or (1, 0) not in mono:
        return None
    d0 = ds[0]
    Amid = d0['lo'] + d0['s_base'] * (d0['hi'] - d0['lo'])
    nrm = {m['name']: m['norm'] for d in ds for m in d['dirs']}
    Wd = np.zeros((len(keys), len(mono)))
    Wd[0, mono.index((0, 0))] = (d0['hi'] - d0['lo']) / nrm['A']
    for n, c in enumerate(qf['coef']):
        Wd[n + 1] = Amid / nrm[f'qf{n}'] * np.array([c[i + 1, j] for i, j in mono])
    v = np.zeros(len(mono))
    v[mono.index((0, 0))], v[mono.index((1, 0))], v[mono.index((0, 1))] = 2 * Amid, -2 * Amid, -2 * Amid
    x = np.zeros(len(names))
    x[[names.index(k) for k in keys]] = np.linalg.solve(Wd.T, v)
    return x


def trivial_basis(tb, e0):
    """The responses of e_c(tc) to the two exact symmetries, e units: a time
    translation, e0', and a scaling about any point, 2 e0 + tc e0' (the span
    does not depend on the point)."""
    from scipy.interpolate import CubicSpline
    de = CubicSpline(tb, e0)(tb, 1)
    return np.stack([de, 2 * e0 + (tb - tb.mean()) * de], 1)


def main_lobes(tb, e0, sign=-1):
    """Echo boundaries: the extrema of e_c of the given sign that are records
    (|e| above every earlier one of that sign), and the next one after the
    last record."""
    ex = extrema(tb, e0)
    ex = ex[np.sign(ex[:, 1]) == sign]
    out, best = [], 0.0
    for t, v in ex:
        if abs(v) > best:
            out.append(t)
            best = abs(v)
        elif out and t > out[-1] and len(out) and out[-1] == ex[np.argmax(np.abs(ex[:, 1])), 0]:
            out.append(t)
            break
    return np.array(out)


def lobe_invariants(tb, e0, R, t0, t1):
    """Responses of the lobe invariants between t0 and t1: for consecutive
    extrema t_n (|e| > 1e-3 max) of e0, delta ln|e_{n+1} / e_n| and
    delta ln(tau_n / tau_{n-1}) (tau_n = t_{n+1} - t_n), both blind to the
    two symmetries; at an extremum delta e = R and delta t = -R' / e0''."""
    from scipy.interpolate import CubicSpline
    s0 = CubicSpline(tb, e0)
    w = (tb > t0) & (tb < t1)
    ex = extrema(tb[w], e0[w], 1e-3 * np.abs(e0[w]).max())
    tn = np.array([min(s0.derivative().roots(), key=lambda r: abs(r - t)) for t in ex[:, 0]])
    en, e2 = s0(tn), s0(tn, 2)
    a = np.empty((len(tn), R.shape[1]))
    dt = np.empty_like(a)
    for i in range(R.shape[1]):
        sR = CubicSpline(tb, R[:, i])
        a[:, i], dt[:, i] = sR(tn) / en, -sR(tn, 1) / e2
    tau = np.diff(tn)
    rows = [(tn[n + 1], 'lnE', a[n + 1] - a[n]) for n in range(len(tn) - 1)]
    rows += [(tn[n + 1], 'lnTau', (dt[n + 2] - dt[n + 1]) / tau[n + 1] - (dt[n + 1] - dt[n]) / tau[n])
             for n in range(len(tn) - 2)]
    return sorted(rows, key=lambda r: (r[0], r[1]))


def joint_components(tb, e0, R, ech):
    """Template-free components over several echoes: the SVD of the
    responses R (columns: directions) of all echoes stacked, every echo
    with the two symmetries projected out and divided by the base's norm
    there.  A component is one input combination (a right singular vector,
    the same in every echo); amp[j, n] is the norm of its output in echo n."""
    F = trivial_basis(tb, e0)
    blocks = []
    for a, b in ech:
        k = (tb >= a) & (tb < b)
        Q = np.linalg.qr(F[k])[0]
        blocks.append((R[k] - Q @ (Q.T @ R[k])) / np.linalg.norm(e0[k]))
    u, s, vt = np.linalg.svd(np.vstack(blocks), full_matrices=False)
    idx = np.cumsum([0] + [len(b) for b in blocks])
    amp = np.array([[np.linalg.norm(u[idx[n]:idx[n + 1], j]) * s[j] for n in range(len(ech))] for j in range(len(s))])
    return vt, amp


def cmd_growth(rjs, lobes=None, gram=None, cmp=None, floor=None, perwin=False):
    """How many independent directions grow along a threshold trajectory
    (gh_response.py batches with one base), per echo and relative to the
    base, so that nothing depends on the focus tc*:
      - echoes: between consecutive main negative lobes of the base's e_c
        (main_lobes, or the given boundaries); Delta from the ratio of
        consecutive lobe spacings, the base's growth per echo;
      - the tuned mode: the coefficients c of R_A in every direction's
        response over the last two echoes, and over each echo alone;
        ||R_A|| / ||e_0|| per echo (the two symmetries projected out of
        every echo window) and the growth per echo mu_1, gamma = Delta /
        ln mu_1;
      - what grows besides it: singular values of the responses of the
        combinations without tuned content (orthogonal to c; dT left out),
        relative to ||e_0||, per echo, and the overlap of their leading right singular
        vectors between echoes (a mode keeps its combination);
      - the lobe invariants (exactly blind to both symmetries): tuned part
        and the rest;
      - the same without the tuned mode singled out: joint_components over
        the last three echoes, the growth of every component per echo;
      - checks: A+4 against A+1, the symmetric (second-order) part, and the
        direction dT (time translation with the frame's zoom: everything of
        it that survives the projections is numerical).
    gram (cmd_gram output): input combinations are measured by their RMS,
    not by the Euclidean norm of their coefficients (the directions are each
    of unit RMS but not orthogonal), so the singular values are per unit
    input RMS; and for fixed input combinations in the leading m-dimensional
    joint subspace, the smallest and largest growth from one echo to the next
    (generalized singular values), and the smallest with the noise floor
    counted against the growth and for it (below one: a resolved decaying
    combination); against the growth also with the scaling control's residual
    as a floor for every combination of unit input RMS, and with cmp (batches
    of the same directions at another amplitude) the full difference of the
    two responses (each with its own tuned content removed) as well.
    floor: for batches with one base and no qf shape space, the noise floor
    per unit coefficient and the scaling residual of other batches on the
    same trajectory, log10 over the last three echoes ('n1,n2,n3:s1,s2,s3',
    from their growth output).
    perwin: in the m table, R_A removed from every window with that window's
    own least-squares coefficient instead of the one of the last two."""
    names, tb, e0, R, S, lin = response_members(rjs)
    F = trivial_basis(tb, e0)

    def proj(k, M):
        Q = np.linalg.qr(F[k])[0]
        return M[k] - Q @ (Q.T @ M[k])
    L = main_lobes(tb, e0) if lobes is None else np.array(lobes)
    ech = list(zip(L[:-1], L[1:]))
    print(f'{len(names)} directions: {" ".join(names)}')
    print('main negative lobes tc: ' + ' '.join(f'{t:.4f}' for t in L))
    tau = np.diff(L)
    for n in range(1, len(tau)):
        print(f'  spacing ratio {tau[n] / tau[n - 1]:.3f} -> Delta {-np.log(tau[n] / tau[n - 1]):.3f}')
    iA = names.index('A')
    k = (tb >= ech[-2][0]) & (tb < ech[-1][1])
    Rk = proj(k, R)
    c = Rk[:, iA] @ Rk / (Rk[:, iA] @ Rk[:, iA])
    ch = c / np.linalg.norm(c)
    print('tuned-mode coefficients (last two echoes): ' + ' '.join(f'{n}:{v:+.3g}' for n, v in zip(names, c)))
    phys = [i for i, n in enumerate(names) if n != 'dT']
    W = np.zeros((len(names), len(phys) - 1))
    W[phys] = np.linalg.svd(ch[None, phys])[2][1:].T
    if gram:
        z = np.load(gram)
        ix = [list(z['names']).index(names[i]) for i in phys]
        Gp = z['gram'][np.ix_(ix, ix)]
        w, U = np.linalg.eigh(W[phys].T @ Gp @ W[phys])
        W = W @ (U / np.sqrt(w)) @ U.T          # orthonormal in the input RMS
        print(f'input metric {gram}: RMS of unit coefficient vectors {np.sqrt(w.min()):.2f} ... {np.sqrt(w.max()):.2f}')
    rnd = [i for i, n in enumerate(names) if n.startswith('rand')]
    print('per echo (log10, relative to ||e_0||): base ||e_0|| ratio | ||R_A|| | rest: sigma_1..5 | cos(c_echo, c) '
          '| max sym/R (random median) | A lin | dT')
    # noise floor: the bases of batches of different sizes agree only up to roundoff; their difference over
    # 2 eps is the noise of one direction's response (per unit coefficient), and of a combination with
    # coefficient vector w it is |w| times that
    from scipy.interpolate import CubicSpline
    dsb = [load_round(rj) for rj in rjs]
    eb = [CubicSpline(*central(d_, [m[0] for m in d_['members']].index('base'))[:2])(tb) for d_ in dsb]
    dif = max((eb[i] - eb[0] for i in range(1, len(eb))), key=lambda v: np.abs(v).max(), default=None)
    epsd = max(d_['eps'] for d_ in dsb)
    print('per echo, noise floor (log10, per unit coefficient; then for sigma_1..5): '
          + ('one base, no estimate' if dif is None else f'from base differences of up to {np.abs(dif).max() / np.abs(e0).max():.1e}'))
    prev, out, sv = None, [], []
    for a, b in ech:
        k = (tb >= a) & (tb < b)
        nb = np.linalg.norm(e0[k]) / np.sqrt(k.sum())
        Rk = proj(k, R)
        ra = np.linalg.norm(Rk[:, iA]) / np.sqrt(k.sum()) / nb
        ce = Rk[:, iA] @ Rk / (Rk[:, iA] @ Rk[:, iA])
        u, s, vt = np.linalg.svd(Rk @ W, full_matrices=False)
        s = s / np.sqrt(k.sum()) / nb
        if dif is not None:
            nu = np.linalg.norm(proj(k, dif[:, None])) / (2 * epsd) / np.sqrt(k.sum()) / nb
            wk = np.linalg.norm(W @ vt.T, axis=0)
            print(f'    noise [{a:.4f},{b:.4f}) {np.log10(nu):6.2f} | ' + ' '.join(f'{np.log10(nu * v):5.2f}' for v in wk[:5]))
        sr = np.linalg.norm(proj(k, S), axis=0) / np.linalg.norm(Rk, axis=0)
        ov = '' if prev is None else f' ov {abs(prev @ vt[0]):.2f}'
        prev = vt[0]
        dT = f'{np.log10(np.linalg.norm(Rk[:, names.index("dT")]) / np.sqrt(k.sum()) / nb):6.2f}' if 'dT' in names else '  -'
        out.append((ra, s[0]))
        sv.append((s[:5], vt[:4]))
        print(f'  [{a:.4f},{b:.4f}) {nb:9.3g} | {np.log10(ra):6.2f} | ' + ' '.join(f'{np.log10(v):5.2f}' for v in s[:5])
              + f'{ov} | {abs(ce @ c) / np.linalg.norm(ce) / np.linalg.norm(c):.3f} | {sr.max():.1e} '
              f'({np.median(sr[rnd]) if rnd else np.nan:.1e}) | {np.linalg.norm(lin[k]) / np.linalg.norm(R[k, iA]):.1e} | {dT}')
    xs = scaling_coef(rjs, names) if gram else None
    scl = {}
    if xs is not None:
        xs = xs / np.sqrt(xs[phys] @ Gp @ xs[phys])
        print('scaling control (exact change of scale of the data, unit RMS; log10): per echo what survives the '
              'projections | its noise floor | that over its trivial part')
        for a, b in ech:
            k = (tb >= a) & (tb < b)
            nb = np.linalg.norm(e0[k]) / np.sqrt(k.sum())
            r = (R @ xs)[:, None]
            pr = np.linalg.norm(proj(k, r)) / np.sqrt(k.sum()) / nb
            tr = np.linalg.norm(r[k] - proj(k, r)) / np.sqrt(k.sum()) / nb
            nz = (np.linalg.norm(proj(k, dif[:, None])) / (2 * epsd) / np.sqrt(k.sum()) / nb * np.linalg.norm(xs)
                  if dif is not None else np.nan)
            scl[a] = pr
            print(f'  [{a:.4f},{b:.4f}) {np.log10(pr):6.2f} | {np.log10(nz):6.2f} | {np.log10(pr / tr):6.2f}')
    for n in range(1, len(out)):
        mu, mu2 = out[n][0] / out[n - 1][0], out[n][1] / out[n - 1][1]
        D = -np.log(tau[n] / tau[n - 1]) if n < len(tau) else np.nan
        print(f'  echo {n}: tuned growth mu_1 {mu:6.2f}, gamma = Delta / ln mu_1 {D / np.log(mu):.3f}; '
              f'leading rest {mu2:5.2f}; sigma_1..5 ' + ' '.join(f'{x:.2f}' for x in sv[n][0] / sv[n - 1][0])
              + '; leading 4-dim input subspaces apart by '
              + ' '.join(f'{np.degrees(np.arccos(min(1.0, x))):.0f}' for x in np.linalg.svd(sv[n][1] @ sv[n - 1][1].T, compute_uv=False)) + ' deg')
    if gram:
        from scipy.optimize import minimize
        Rc = None
        if cmp:
            nc, tbc, e0c, Rc = response_members(cmp)[:4]
            assert nc == names, 'cmp: the same directions'
            Rc = CubicSpline(tbc, Rc)(tb)
            kk = (tb >= ech[-2][0]) & (tb < ech[-1][1])
            Pc = proj(kk, Rc)
            cc = Pc[:, iA] @ Pc / (Pc[:, iA] @ Pc[:, iA])
        B, nu, sc, Dc = [], [], [], []
        fl = [[10 ** float(v) for v in f.split(',')] for f in floor.split(':')] if floor else None
        for j, (a, b) in enumerate(ech[-3:]):
            k = (tb >= a) & (tb < b)
            nb = np.linalg.norm(e0[k]) / np.sqrt(k.sum())
            Pk = proj(k, R)
            if perwin:
                Pk = Pk - np.outer(Pk[:, iA], Pk[:, iA] @ Pk / (Pk[:, iA] @ Pk[:, iA]))
            B.append(Pk @ W / np.sqrt(k.sum()) / nb)
            if dif is not None:
                nu.append(np.linalg.norm(proj(k, dif[:, None])) / (2 * epsd) / np.sqrt(k.sum()) / nb)
            elif fl:
                nu.append(fl[0][j])
            sc.append(scl.get(a, fl[1][j] if fl else 0.0))
            if Rc is not None:
                P0, P1 = proj(k, R), proj(k, Rc)
                Dc.append(((P1 - np.outer(P1[:, iA], cc)) - (P0 - np.outer(P0[:, iA], c))) @ W / np.sqrt(k.sum()) / nb)
        V = np.linalg.svd(np.vstack(B), full_matrices=False)[2].T
        print('fixed input combinations in the leading m joint components: min / max growth between echoes'
              + ('; (min with the noise floor against the growth: the later output minus its floor, the earlier plus;'
                 ' then min with it for the growth' + ('; min against the growth with noise and scaling residual'
                                                     if scl or fl else '') + ('; and with the amplitude difference' if Dc else '')
                 + ')' if nu else ''))
        rng = np.random.default_rng(0)
        for m in range(2, V.shape[1] + 1):
            out = []
            P = V[:, :m]
            for n0, n1 in ((0, 1), (1, 2), (0, 2)):
                Rn = np.linalg.qr(B[n0] @ P)[1]
                _, g, vt_ = np.linalg.svd(B[n1] @ P @ np.linalg.inv(Rn))
                out.append(f'{g.min():6.2f} {g.max():6.2f}')
                if nu:
                    def worst(x):
                        cn = np.linalg.norm(W @ P @ x)
                        return ((np.linalg.norm(B[n1] @ P @ x) - nu[n1] * cn)
                                / (np.linalg.norm(B[n0] @ P @ x) + nu[n0] * cn))
                    def best(x):
                        cn = np.linalg.norm(W @ P @ x)
                        den = np.linalg.norm(B[n0] @ P @ x) - nu[n0] * cn
                        return (np.linalg.norm(B[n1] @ P @ x) + nu[n1] * cn) / den if den > 0 else np.inf
                    def budget(x, amp):
                        y = P @ x
                        cn = np.linalg.norm(W @ y)
                        e = [nu[n] * cn + sc[n] * np.linalg.norm(y) + (np.linalg.norm(Dc[n] @ y) if amp else 0.0)
                             for n in (n0, n1)]
                        return (np.linalg.norm(B[n1] @ y) - e[1]) / (np.linalg.norm(B[n0] @ y) + e[0])
                    x0 = [np.linalg.solve(Rn, vt_[-1])] + [rng.standard_normal(m) for _ in range(20)]
                    opt = dict(method='Nelder-Mead', options=dict(xatol=1e-6, fatol=1e-8, maxiter=4000))
                    rob = min(minimize(worst, x, **opt).fun for x in x0)
                    dec = min(minimize(best, x, **opt).fun for x in x0)
                    out[-1] += f' ({rob:.2f} {dec:.2f}'
                    if scl or fl:
                        out[-1] += f'; {min(minimize(budget, x, args=(False,), **opt).fun for x in x0):.2f}'
                    if Dc:
                        out[-1] += f' {min(minimize(budget, x, args=(True,), **opt).fun for x in x0):.2f}'
                    out[-1] += ')'
            print(f'  m = {m}: ' + ' | '.join(out) + '   (echo 1 -> 2 | 2 -> 3 | 1 -> 3)')
    Rj = R[:, phys]
    if gram:
        w_, U_ = np.linalg.eigh(Gp)
        Rj = Rj @ (U_ / np.sqrt(w_)) @ U_.T
    vt, amp = joint_components(tb, e0, Rj, ech[-3:])
    print('joint over the last three echoes (one input combination per component, dT left out): growth per echo')
    for j in range(min(6, len(phys))):
        print(f'  comp {j}: ' + ' '.join(f'{amp[j, n + 1] / amp[j, n]:6.2f}' for n in range(amp.shape[1] - 1)))
    print('lobe invariants: t_n, kind, log10 |tuned part|, log10 |rest|' + (', log10 |dT|' if 'dT' in names else ''))
    for t, kind, v in lobe_invariants(tb, e0, R, ech[0][0] - 1e-9, ech[-1][1] + 1e-9):
        x = f'  {t:.4f} {kind:5s} {np.log10(abs(v @ ch)):6.2f} {np.log10(np.linalg.norm(v @ W)):6.2f}'
        print(x + (f' {np.log10(abs(v[names.index("dT")])):6.2f}' if 'dT' in names else ''))


def cmd_qfree(rjs, out):
    """The tuned content of the Brill shape directions as one linear function
    on the coefficient space of the deformations, and its kernel.  Batches
    with --nq (T0 = 0): direction k is the response to dq_k = A x P_k e^(-x-y)
    (gh_data.q_shape) scaled to unit RMS, A to (A_hi - A_lo) x e^(-x-y), so its
    coefficient vector w is known; the tuned coefficients c (last two echoes,
    as in growth) are fitted by c = w . gamma, and the residual measures how
    linear the responses are in the coefficients.  Writes the kernel of gamma
    (orthonormal in the coordinates p (i+j)!, in which the random draws are
    isotropic) as q_poly coefficient matrices: tuned-free directions for
    gh_response.py --qcoef."""
    from math import factorial
    from vacuum import gh_data as gdat
    names, tb, e0, R = response_members(rjs)[:4]
    F = trivial_basis(tb, e0)
    L = main_lobes(tb, e0)
    k = (tb >= L[-3]) & (tb < L[-1])
    Q = np.linalg.qr(F[k])[0]
    Rk = R[k] - Q @ (Q.T @ R[k])
    iA = names.index('A')
    c = Rk[:, iA] @ Rk / (Rk[:, iA] @ Rk[:, iA])
    deg = {load_round(rj)['args']['qdeg'] for rj in rjs}
    assert len(deg) == 1, 'one degree'
    deg = deg.pop()
    mono = [(i, j) for i in range(deg + 1) for j in range(deg + 1 - i)]
    fac = np.array([factorial(i + j) for i, j in mono], float)
    W = np.full((len(names), len(mono)), np.nan)
    for rj in rjs:
        d = load_round(rj)
        Amid = d['lo'] + d['s_base'] * (d['hi'] - d['lo'])
        nrm = {m['name']: m['norm'] for m in d['dirs']}
        W[iA] = 0.0
        W[iA, mono.index((0, 0))] = (d['hi'] - d['lo']) / nrm['A']
        rng = np.random.default_rng((d['args']['seed'], 1))
        for kq in range(d['args']['nq']):
            cq = gdat.shape_coef(rng, deg)
            key = f'bq{kq}:{d["args"]["seed"]}'
            if key in names and f'bq{kq}' in nrm:
                W[names.index(key)] = Amid / nrm[f'bq{kq}'] * np.array([cq[i + 1, j] for i, j in mono])
    assert np.isfinite(W).all(), 'a direction without coefficients'
    gam, _, rk, sv = np.linalg.lstsq(W, c, rcond=None)
    res = c - W @ gam
    print(f'{len(names)} directions, {len(mono)} coefficients (rank {rk}, condition {sv[0] / sv[-1]:.1e}): '
          f'|c - w.gamma| / |c| = {np.linalg.norm(res) / np.linalg.norm(c):.1e}, max {np.abs(res).max():.1e}')
    print('gamma (x 1/(i+j)!): ' + ' '.join(f'x^{i + 1}y^{j}:{g / f:+.3g}' for (i, j), g, f in zip(mono, gam, fac)))
    B = np.linalg.svd((gam / fac)[None])[2][1:] / fac        # kernel, orthonormal in p (i+j)!
    coef = np.zeros((len(B), deg + 2, deg + 1))
    for n, b in enumerate(B):
        for (i, j), v in zip(mono, b):
            coef[n, i + 1, j] = v
    print(f'{len(B)} tuned-free directions: |gamma . p| / (|gamma| |p|) <= {np.abs(B @ gam).max() / np.linalg.norm(gam) / np.linalg.norm(B, axis=1).min():.1e}')
    np.savez(out, coef=coef, gamma=gam, mono=np.array(mono), resid=res, names=np.array(names), rjs=np.array(rjs))
    print(f'wrote {out}')


def cmd_gram(rjs, out):
    """The input metric of gh_response.py batches with one base: every
    direction is rebuilt as in gh_response.py (gh_codim.rebuild: same base,
    field scales and window), each of unit RMS, and the Gram matrix of the
    directions in that RMS inner product is written with their names (the
    keys of response_members).  growth --gram uses it to measure input
    combinations by their RMS instead of the Euclidean norm of their
    coefficients."""
    from vacuum import gh_sim as g
    from vacuum.gh_codim import rebuild
    from vacuum.gh_linmap import field_scales
    b = rebuild(rjs)
    G, a0 = b['G'], load_round(rjs[0])['args']
    sc = field_scales(G, b['ub'], g.active(G, g.flat_state(G, b['fr'][0])))
    win = np.tile((G.R < a0['rw']).ravel(), g.NA)
    Y = np.array([(d / sc)[win] for d in b['D']])
    gram = Y @ Y.T / Y.shape[1]
    w = np.linalg.eigvalsh(gram)
    print(f'{len(b["keys"])} directions; diagonal {np.diag(gram).min():.3f} ... {np.diag(gram).max():.3f}; '
          f'eigenvalues {w.min():.3f} ... {w.max():.3f}')
    np.savez(out, names=np.array(b['keys']), gram=gram, rjs=np.array(rjs))
    print(f'wrote {out}')


def axis_members(rjs):
    """The batches written with --axis: tc(T) of the base, the axis nodes,
    the base's physical E^Z_Z(T, Z) on the axis and, per direction (keys as
    in response_members, dT left out), its antisymmetric response there.
    The base is bitwise the same in every batch on the common refreshes."""
    T0, Z, E0, tc, R = None, None, None, None, {}
    for rj in rjs:
        z = np.load(rj.replace('_round0.json', '_states.npz'))
        if 'eax' not in z.files or not z['eax'].size:
            continue
        d = load_round(rj)
        labs = list(z['labels'])
        h = d['hist']
        T, L = np.array(h['T']), np.exp(np.array(h['lnL']))
        a0 = np.array(h['a0'])[:, labs.index('base')]
        t = h['tc'][0] + np.concatenate([[0.0], np.cumsum(0.5 * (L[1:] * a0[1:] + L[:-1] * a0[:-1]) * np.diff(T))])
        E = np.array(z['eax']) / L[:, None, None] ** 2
        if T0 is None:
            T0, Z, E0, tc = T, np.array(z['zax']), E[:, labs.index('base')], t
        n = min(len(T), len(T0))
        assert np.array_equal(T[:n], T0[:n]) and np.array_equal(E[:n, labs.index('base')], E0[:n]), f'{rj}: another base'
        T0, E0, tc = T0[:n], E0[:n], tc[:n]
        for m in d['dirs']:
            key = f'{m["name"]}:{d["args"]["seed"]}' if m['name'].startswith('rand') else m['name']
            if key not in R and key != 'dT':
                R[key] = (E[:, labs.index(m['name'] + '+1')] - E[:, labs.index(m['name'] + '-1')]) / (2 * d['eps'])
    n = len(T0)
    return T0, tc, Z, E0, {k: v[:n] for k, v in R.items()}


def axis_critical_points(T, Z, E, tmask):
    """Critical points of E(T, Z) on the axis surface (both gradient
    components change sign in a 3 x 3 neighbourhood of a node, then Newton on
    the gradient of a bicubic spline): (T, Z, E, kind)."""
    from scipy.interpolate import RectBivariateSpline
    from scipy.optimize import root
    sp = RectBivariateSpline(T, Z, E, kx=3, ky=3)
    dT, dZ = T[1] - T[0], Z[1] - Z[0]
    out = []
    for i in np.flatnonzero(tmask)[1:-1]:
        for j in range(1, len(Z) - 1):
            w = E[i - 1:i + 2, j - 1:j + 2]
            if (w[2, 1] - w[1, 1]) * (w[1, 1] - w[0, 1]) > 0 or (w[1, 2] - w[1, 1]) * (w[1, 1] - w[1, 0]) > 0:
                continue
            r = root(lambda p: [sp(*p, dx=1)[0, 0] * dT, sp(*p, dy=1)[0, 0] * dZ], [T[i], Z[j]], method='hybr')
            t, zz = r.x
            if not r.success or not (T[i - 1] <= t <= T[i + 1] and Z[j - 1] <= zz <= Z[j + 1]):
                continue
            H = np.array([[sp(t, zz, dx=2)[0, 0], sp(t, zz, dx=1, dy=1)[0, 0]],
                          [sp(t, zz, dx=1, dy=1)[0, 0], sp(t, zz, dy=2)[0, 0]]])
            ev = np.linalg.eigvalsh(H)
            kind = 'max' if ev.max() < 0 else 'min' if ev.min() > 0 else 'saddle'
            if all(abs(t - o[0]) > 2 * dT or abs(zz - o[1]) > 3 * dZ for o in out):
                out.append((t, zz, sp(t, zz)[0, 0], kind))
    return sorted(out)


def cmd_axis(rjs, lobes=None):
    """Whether the growing components found at the centre (growth) are all
    there is on the axis.  At a critical point of E(T, Z) on the axis surface
    the first-order response is gauge-free; the difference of delta ln E
    there and at the nearest central extremum is free of both symmetries
    (time translation does not change a critical value, scaling changes
    every ln E by the same constant).  Per off-centre critical point
    (|E| > 0.1 of the base's central RMS in that echo) and per pair of
    consecutive central extrema, this vector over the directions with axis
    data, each direction weighted by the inverse of its central response:
      - the residual outside the span of the first m joint components of
        the centre (all directions, last three echoes), m = 1..5, against
        the mean for random vectors;
      - per echo, the singular values of the row-normalized vectors (how
        many independent combinations the axis sees), all and with the
        tuned component (joint component 0) removed."""
    from scipy.interpolate import CubicSpline, RectBivariateSpline
    names, tb, e0, R, S, lin = response_members(rjs)
    phys = [i for i, n in enumerate(names) if n != 'dT']
    L = main_lobes(tb, e0) if lobes is None else np.array(lobes)
    ech = list(zip(L[:-1], L[1:]))[-3:]
    vt = joint_components(tb, e0, R[:, phys], ech)[0]
    T, tc, Z, E0, Rax = axis_members(rjs)
    keys = [k for k in Rax if k in names]
    sub = [names.index(k) for k in keys]
    V = vt[:5][:, [phys.index(i) for i in sub]].T
    k = (tb >= ech[0][0]) & (tb < ech[-1][1])
    F = np.linalg.qr(trivial_basis(tb, e0)[k])[0]
    w = 1.0 / np.linalg.norm(R[k][:, sub] - F @ (F.T @ R[k][:, sub]), axis=0)
    Vw = V * w[:, None]
    cen = [(t, np.array([CubicSpline(tb, R[:, i])(t) for i in sub]) / v)
           for t, v in extrema(tb, e0, 1e-3 * np.abs(e0[k]).max()) if ech[0][0] - 0.05 < t < ech[-1][1] + 0.01]
    rms = lambda t: next((np.sqrt(np.mean(e0[(tb >= a) & (tb < b)] ** 2)) for a, b in ech if a <= t < b), np.inf)
    tcf = lambda t: float(np.interp(t, T, tc))
    sp = [RectBivariateSpline(T, Z, Rax[key], kx=3, ky=3) for key in keys]
    rows = []
    for t, zz, v, kind in axis_critical_points(T, Z, E0, (tc > ech[0][0] - 0.02) & (tc < ech[-1][1] + 0.01)):
        if zz < 2 * (Z[1] - Z[0]) or abs(v) < 0.1 * rms(tcf(t)):
            continue
        r = np.array([f(t, zz)[0, 0] for f in sp]) / v
        j = int(np.argmin([abs(c[0] - tcf(t)) for c in cen]))
        rows.append((tcf(t), zz, kind, (r - cen[j][1]) * w))
    rows += [(t2, 0.0, 'centre', (r2 - r1) * w) for (t1, r1), (t2, r2) in zip(cen[:-1], cen[1:])]
    rows.sort(key=lambda r: r[0])

    def resid(x, m):
        a = np.linalg.lstsq(Vw[:, :m], x, rcond=None)[0]
        return np.linalg.norm(x - Vw[:, :m] @ a) / np.linalg.norm(x)
    rng = np.random.default_rng(0)
    base = [np.mean([resid(rng.standard_normal(len(sub)), m) for _ in range(500)]) for m in range(1, 6)]
    print(f'{len(keys)} directions with axis data: {" ".join(keys)}')
    print('echoes ' + ' '.join(f'[{a:.4f},{b:.4f})' for a, b in ech))
    print('   tc      Z (frame)  kind   | residual outside joint components 0..m-1, m = 1..5')
    for t, zz, kind, x in rows:
        print(f'  {t:.4f} {zz:8.4f}  {kind:6s} | ' + ' '.join(f'{resid(x, m):5.2f}' for m in range(1, 6)))
    print('  random vectors           | ' + ' '.join(f'{v:5.2f}' for v in base))
    qt = Vw[:, 0] / np.linalg.norm(Vw[:, 0])
    X12 = Vw[:, 1:3] - np.outer(qt, qt @ Vw[:, 1:3])
    print('per echo: median residual (m = 1..5) | singular values of the row-normalized vectors | the same, tuned removed '
          '| tuned removed: median residual outside component 1, outside 1 and 2')
    for a, b in ech:
        X = np.array([x for t, zz, kind, x in rows if a <= t < b])
        if len(X) < 2:
            continue
        med = [np.median([resid(x, m) for x in X]) for m in range(1, 6)]
        s1 = np.linalg.svd(X / np.linalg.norm(X, axis=1, keepdims=True), compute_uv=False)
        Y = X - np.outer(X @ qt, qt)
        s2 = np.linalg.svd(Y / np.linalg.norm(Y, axis=1, keepdims=True), compute_uv=False)
        Yn = Y / np.linalg.norm(Y, axis=1, keepdims=True)
        out = [np.median(np.linalg.norm(Yn - Yn @ Q @ Q.T, axis=1)) for Q in (np.linalg.qr(X12[:, :1])[0], np.linalg.qr(X12)[0])]
        print(f'  [{a:.4f},{b:.4f}) {len(X):3d} rows: ' + ' '.join(f'{v:.2f}' for v in med) + ' | '
              + ' '.join(f'{v / s1[0]:.3f}' for v in s1[:5]) + ' | ' + ' '.join(f'{v / s2[0]:.3f}' for v in s2[:5])
              + ' | ' + ' '.join(f'{v:.2f}' for v in out))


def departures(spec, others, thr=0.3):
    """When members leave the near-critical trajectory (cmd_departure):
    A*, the flip pair index, the tc where the pair itself differs by thr,
    rows (|A - A*|, side, outcome, T_end, departure tc, round json) sorted,
    and the reference (tc grid, e_ref)."""
    from scipy.interpolate import CubicSpline
    dr = load_round(spec)
    st = [m[1] for m in dr['members']]
    i = [k for k in range(len(st) - 1) if st[k] == 'collapse' and st[k + 1] == 'disperse'][0]
    Astar = 0.5 * (dr['members'][i][0] + dr['members'][i + 1][0])

    def spl(d, m):
        tc, e = central(d, m)[:2]
        return CubicSpline(tc, e), tc[0], tc[-1]
    a, b = spl(dr, i), spl(dr, i + 1)
    tg = np.linspace(max(a[1], b[1]), min(a[2], b[2]), 8000)
    eref = 0.5 * (a[0](tg) + b[0](tg))
    env = np.array([np.sqrt(np.mean(eref[(tg > t - 0.004) & (tg < t + 0.004)] ** 2)) for t in tg])
    dep = lambda e: tg[np.argmax(np.abs(e - eref) / env > thr)] if (np.abs(e - eref) / env > thr).any() else np.nan
    tpair = tg[np.argmax(np.abs(a[0](tg) - b[0](tg)) / env > thr)]
    rows = []
    for f in [spec] + list(others):
        d = dr if f == spec else load_round(f)
        for m, (A, o, te) in enumerate(d['members']):
            if f == spec and m in (i, i + 1):
                continue
            sm, t0, t1 = spl(d, m)
            k = (tg >= t0) & (tg <= t1)
            e = np.full(tg.size, np.nan)
            e[k] = sm(tg[k])
            rows.append((abs(A - Astar), np.sign(A - Astar), o, te, dep(np.where(k, e, eref)), f))
    rows.sort()
    return Astar, i, tpair, rows, tg, eref


def cmd_departure(spec, others, thr=0.3):
    """When members leave the near-critical trajectory, against their distance
    to the threshold.  Reference: the flipping pair of the given round
    (<round.json>, A* their midpoint), e_ref the mean of their e_c(tc).  A
    member departs at the first tc where |e_c - e_ref| exceeds thr times the
    RMS of e_ref over tc +- 0.004; members of the other round jsons (same
    family) are included.  A self-similar cascade makes the departures
    accumulate (tc steps shrinking by e^-Delta per echo); a phase at a fixed
    scale makes them equally spaced in tc.  Also the RMS of e_ref along tc."""
    Astar, i, tpair, rows, tg, eref = departures(spec, others, thr)
    dr = load_round(spec)
    st = [m[1] for m in dr['members']]
    print(f'A* = {Astar:.13f} (flip pair {i}, {i + 1}: {st[i]} at T = {dr["members"][i][2]:.1f}, '
          f'{st[i + 1]} at T = {dr["members"][i + 1][2]:.1f}); the pair itself differs by {thr} at tc = {tpair:.4f}')
    print('  |A - A*|  side outcome    T_end  departure tc')
    for r in rows:
        print(f'  {r[0]:.2e}  {r[1]:+.0f}  {r[2]:9s} {r[3]:6.1f}  {r[4]:.4f}')
    print('  RMS of e_ref over tc +- 0.01: ' + ' '.join(f'{t:.2f}:{np.sqrt(np.mean(eref[np.abs(tg - t) < 0.01] ** 2)):.2g}'
                                                    for t in np.arange(1.90, tg[-1], 0.02)))


def cmd_curvature(pair, rj, mem, Ts):
    """Kretschmann extremum on pair checkpoints: max |I| and its physical
    position (L X, L Z), in units of rem; the central value; the largest
    values on the axis and on the equator; and for the axis maximum its
    physical coordinate distance L Z and its proper distance from the centre
    along the axis in the slice, L int sqrt(g_ZZ) dZ on the first X node."""
    from vacuum import gh_sim as g
    C = Checkpoints(pair, rj)
    tcs = load_round(rj)['cfg']['tclock'][1]
    for T in Ts:
        G, gh, u, Ti, L, tc = C.state(mem, T)
        t0 = time.time()
        I = g.riemann_invariant(G, gh, u) / L ** 4
        rem = tcs - tc
        k = np.unravel_index(np.argmax(np.abs(I)), I.shape)
        ka, ke = int(np.argmax(np.abs(I[0]))), int(np.argmax(np.abs(I[:, 0])))
        Z = np.concatenate([[0.0], G.Z[0]])
        sq = np.sqrt(g.unpack(G, u)[0][5][0])
        sq = np.concatenate([[sq[0]], sq])
        s = np.concatenate([[0.0], np.cumsum(0.5 * (sq[1:] + sq[:-1]) * np.diff(Z))])
        print(f'T {Ti:5.1f} tc {tc:.4f}: max|I| {I[k]:+.3e} at (X, Z) L/rem = ({L * G.X[k] / rem:.3f}, '
              f'{L * G.Z[k] / rem:.3f}) | centre {I[0, 0]:+.3e} | axis {I[0, ka]:+.3e} at Z L/rem '
              f'{L * G.Z[0, ka] / rem:.3f}, z {L * G.Z[0, ka]:.4f}, proper {L * s[ka + 1]:.4f} | equator '
              f'{I[ke, 0]:+.3e} at X L/rem {L * G.X[ke, 0] / rem:.3f} [{time.time() - t0:.0f}s]', flush=True)


def centre_E(gh, u, L):
    """E^Z_Z at the centre node from the Riemann tensor (what the runs record)
    and from the Weyl tensor, E^W = E^R + Ric_perp/2 - h R_nn/2 - h R/6,
    physical values (/ L^2).  The Ricci terms vanish for an exact vacuum
    solution: they measure the constraint violation in e_c."""
    from vacuum import gh_sim as g
    T_, Z_ = g.T_, g.Z_
    aux = gh.rhs(u, full=True)[1]
    gg, gi, dg, ddg = (np.asarray(aux[k])[..., 0] for k in ('g', 'gi', 'dg', 'ddg'))
    ddg = ddg.copy()
    ddg[T_, T_] = np.asarray(aux['A'])[..., 0]
    Gl = 0.5 * (np.einsum('acb->cab', dg) + np.einsum('bca->cab', dg) - dg)
    Gu = np.einsum('cd,dab->cab', gi, Gl)
    R = 0.5 * (np.einsum('bcad->abcd', ddg) + np.einsum('adbc->abcd', ddg)
               - np.einsum('acbd->abcd', ddg) - np.einsum('bdac->abcd', ddg))
    R += np.einsum('ebc,fad,ef->abcd', Gu, Gu, gg) - np.einsum('ebd,fac,ef->abcd', Gu, Gu, gg)
    Ric = np.einsum('ac,abcd->bd', gi, R)
    Rs = np.einsum('bd,bd->', gi, Ric)
    nl = np.zeros(4)
    nl[T_] = -1.0 / np.sqrt(-gi[T_, T_])
    nu = gi @ nl
    h = gg + np.outer(nl, nl)
    P = np.eye(4) + np.outer(nu, nl)            # P[c, a] = delta^c_a + n^c n_a
    Er = np.einsum('acbd,c,d->ab', R, nu, nu)
    Ew = Er + 0.5 * P.T @ Ric @ P - 0.5 * h * (nu @ Ric @ nu) - Rs / 6.0 * h
    return (gi @ Er)[Z_, Z_] / L ** 2, (gi @ Ew)[Z_, Z_] / L ** 2   # E_ab n^b = 0: g^{Zc} E_cZ = E^Z_Z


def axis_weyl(G, gh, u, L):
    """On the axis nodes (X node 0): physical z, E^Z_Z from the Riemann and
    from the Weyl tensor (E^W = E^R + Ric_perp/2 - h R_nn/2 - h R/6), the
    Kretschmann scalar and Ric_ab Ric^ab, physical values (/ L^2, / L^4).
    The Ricci terms vanish for an exact vacuum solution."""
    from vacuum import gh_sim as g
    T_, Z_ = g.T_, g.Z_
    aux = gh.rhs(u, full=True)[1]
    k = np.arange(G.shape[1])                      # X node 0: flat indices 0 .. nz - 1
    gg, gi, dg = (np.asarray(aux[q])[..., k] for q in ('g', 'gi', 'dg'))
    ddg = np.asarray(aux['ddg'])[..., k].copy()
    ddg[T_, T_] = np.asarray(aux['A'])[..., k]
    Gl = 0.5 * (np.einsum('acbN->cabN', dg) + np.einsum('bcaN->cabN', dg) - dg)
    Gu = np.einsum('cdN,dabN->cabN', gi, Gl)
    R = 0.5 * (np.einsum('bcadN->abcdN', ddg) + np.einsum('adbcN->abcdN', ddg)
               - np.einsum('acbdN->abcdN', ddg) - np.einsum('bdacN->abcdN', ddg))
    R += np.einsum('ebcN,fadN,efN->abcdN', Gu, Gu, gg) - np.einsum('ebdN,facN,efN->abcdN', Gu, Gu, gg)
    Ric = np.einsum('acN,abcdN->bdN', gi, R)
    Rs = np.einsum('bdN,bdN->N', gi, Ric)
    nl = np.zeros((4, k.size))
    nl[T_] = -1.0 / np.sqrt(-gi[T_, T_])
    nu = np.einsum('abN,bN->aN', gi, nl)
    h = gg + np.einsum('aN,bN->abN', nl, nl)
    P = np.eye(4)[:, :, None] + np.einsum('cN,aN->caN', nu, nl)
    Er = np.einsum('acbdN,cN,dN->abN', R, nu, nu)
    Rnn = np.einsum('abN,aN,bN->N', Ric, nu, nu)
    Ew = (Er + 0.5 * np.einsum('caN,cdN,dbN->abN', P, Ric, P) - 0.5 * h * Rnn - h * Rs / 6.0)
    eR = np.einsum('cN,cN->N', gi[Z_], Er[:, Z_])
    eW = np.einsum('cN,cN->N', gi[Z_], Ew[:, Z_])
    Ru = np.einsum('abcdN,aeN,bfN,cgN,dhN->efghN', R, gi, gi, gi, gi)
    I = np.einsum('abcdN,abcdN->N', R, Ru)
    RR = np.einsum('abN,acN,bdN,cdN->N', Ric, gi, gi, Ric)
    return G.az.x * L, eR / L ** 2, eW / L ** 2, I / L ** 4, RR / L ** 4


def cmd_ricci(states, rj, mems, Ts=(None,)):
    """The vacuum residual of the axis curvature for members of a --save-T
    file or the checkpoints of a --members file: at the largest |E^Z_Z| on the
    axis, E^Z_Z from the Riemann tensor, the change from the Weyl tensor
    relative to it, and sqrt(Ric_ab Ric^ab / I); the largest of these two
    ratios over the axis where |E| > 0.1 of its peak."""
    C = Checkpoints(states, rj)
    for mem in mems:
        for T in Ts:
            G, gh, u, Ti, L, tc = C.state(mem, C.T[0] if T is None else T)
            z, eR, eW, I, RR = axis_weyl(G, gh, u, L)
            i = int(np.nanargmax(np.abs(eR)))
            k = np.abs(eR) > 0.1 * abs(eR[i])
            rel = np.abs(eW - eR) / abs(eR[i])
            ric = np.sqrt(np.abs(RR) / np.abs(I))
            print(f'member {mem} T {Ti:.2f}: peak E^Z_Z {eR[i]:+.4e} at z {z[i]:.5f}; Weyl - Riemann '
                  f'{(eW[i] - eR[i]) / eR[i]:+.2e}; |Ric|/|Riem| {ric[i]:.2e}; over |E| > 0.1 peak '
                  f'(z {z[k].min():.4f}-{z[k].max():.4f}): {rel[k].max():.2e}, {ric[k].max():.2e}', flush=True)


def cmd_weyl(pair, rj, Ts):
    """centre_E for both members of a pair file at its checkpoints: e_c from
    the Riemann and from the Weyl tensor, and the same for the difference
    hi - lo."""
    C = Checkpoints(pair, rj)
    for T in Ts:
        row = {}
        for mem in ('lo', 'hi'):
            G, gh, u, Ti, L, tc = C.state(mem, T)
            row[mem] = centre_E(gh, u, L) + (tc,)
        (er, ew, tc), (er2, ew2, _) = row['lo'], row['hi']
        print(f'T {Ti:5.1f} tc {tc:.4f}: e_c (Riemann) {er:+.5e}, Weyl - Riemann {ew - er:+.2e} '
              f'(rel {(ew - er) / er:+.1e}); hi - lo: Riemann {er2 - er:+.4e}, Weyl {ew2 - ew:+.4e} '
              f'(rel {((ew2 - ew) - (er2 - er)) / (er2 - er):+.1e})', flush=True)


def cmd_weylresp(states, rj):
    """centre_E on the saved states of a gh_response.py batch (run with a
    small --every): per direction and saved time, the response (e_+ - e_-)
    from the Riemann tensor, and how much the Weyl tensor changes it, relative
    to the response itself and to the RMS of the direction's response (with
    the two symmetries projected out) over the echo window containing that
    time."""
    from vacuum import gh_sim as g
    d = load_round(rj)
    cfg = d['cfg']
    grids = [g.make(cfg)]
    Tg, n2, ax2 = cfg['regrid'][:3]
    grids.append(g.make(dict(cfg, n=int(n2), nz=grids[0][0].shape[1], ax=ax2)))
    h = d['hist']
    fr = {round(T, 6): (e, np.exp(l)) for T, e, l in zip(h['T'], h['ell'], h['lnL'])}
    z = np.load(states)
    labels = list(z['labels'])
    names, tb, e0, Rr = response_members([rj])[:4]
    F = trivial_basis(tb, e0)
    L_ = main_lobes(tb, e0)
    ech = list(zip(L_[:-1], L_[1:]))
    tcs = {round(T, 6): t for T, t in zip(h['T'], h['tc'])}
    print('direction  T   tc  | response (Riemann) | Weyl - Riemann: relative to it, to the window RMS of the projected response')
    for T in z['T0']:
        Tk = round(float(T), 6)
        if Tk not in fr:
            continue
        ell, L = fr[Tk]
        G, gh = grids[1] if T >= Tg - 1e-9 else grids[0]
        e = {}
        for b, lab in enumerate(labels):
            i = int(np.argmin(np.abs(z[f'T{b}'] - T)))
            e[lab] = centre_E(gh, g.full_state(G, z[f'u{b}'][i], ell), L)
        tc = tcs.get(Tk, np.nan)
        win = [(a, b_) for a, b_ in ech if a <= tc < b_]
        for m in d['dirs']:
            n = m['name']
            if n + '-1' not in e:
                continue
            dr = e[n + '+1'][0] - e[n + '-1'][0]
            dw = e[n + '+1'][1] - e[n + '-1'][1]
            rms = np.nan
            if win:
                k = (tb >= win[0][0]) & (tb < win[0][1])
                Q = np.linalg.qr(F[k])[0]
                r = Rr[k, names.index(n)]
                rms = np.sqrt(np.mean((r - Q @ (Q.T @ r)) ** 2)) * 2 * m.get('eps', d['eps'])
            print(f'  {n:6s} {T:5.1f} {tc:.4f} | {dr:+.4e} | {(dw - dr) / dr:+.1e} {(dw - dr) / rms:+.1e}', flush=True)


def cmd_combos(rjs, gram, m, out):
    """The input combinations spanning the leading m joint components of the
    untuned responses over the last three echoes (the subspace of growth's
    'fixed input combinations' table), as q_poly coefficient matrices for
    gh_response.py --qcoef (each a combination of the amplitude direction and
    the qf directions of --qcoef batches with --gram), with their growth
    bounds as growth computes them."""
    names, tb, e0, R, S, lin = response_members(rjs)
    F = trivial_basis(tb, e0)

    def proj(k, M):
        Q = np.linalg.qr(F[k])[0]
        return M[k] - Q @ (Q.T @ M[k])
    L = main_lobes(tb, e0)
    ech = list(zip(L[:-1], L[1:]))[-3:]
    iA = names.index('A')
    k = (tb >= ech[-2][0]) & (tb < ech[-1][1])
    Rk = proj(k, R)
    c = Rk[:, iA] @ Rk / (Rk[:, iA] @ Rk[:, iA])
    ch = c / np.linalg.norm(c)
    phys = [i for i, n in enumerate(names) if n != 'dT']
    W = np.zeros((len(names), len(phys) - 1))
    W[phys] = np.linalg.svd(ch[None, phys])[2][1:].T
    zg = np.load(gram)
    ix = [list(zg['names']).index(names[i]) for i in phys]
    Gp = zg['gram'][np.ix_(ix, ix)]
    w, U = np.linalg.eigh(W[phys].T @ Gp @ W[phys])
    W = W @ (U / np.sqrt(w)) @ U.T
    B = []
    for a, b in ech:
        kk = (tb >= a) & (tb < b)
        B.append(proj(kk, R) @ W / np.linalg.norm(e0[kk]))
    V = np.linalg.svd(np.vstack(B), full_matrices=False)[2].T
    Y = (W @ V[:, :m]).T                                 # (m, directions), unit input RMS
    for n0, n1 in ((0, 1), (1, 2)):
        Rn = np.linalg.qr(B[n0] @ V[:, :m])[1]
        g_ = np.linalg.svd(B[n1] @ V[:, :m] @ np.linalg.inv(Rn), compute_uv=False)
        print(f'echo {n0 + 1} -> {n1 + 1}: growth in the plane {g_.min():.2f} ... {g_.max():.2f}')
    nrm, qc, d0 = {}, None, load_round(rjs[0])
    for rj in rjs:
        d = load_round(rj)
        nrm.update({mm['name']: mm['norm'] for mm in d['dirs']})
        qc = np.load(d['args']['qcoef'])['coef']
    Amid = d0['lo'] + d0['s_base'] * (d0['hi'] - d0['lo'])
    coef = np.zeros((m,) + qc.shape[1:])
    for j in range(m):
        coef[j, 1, 0] = Y[j, iA] * (d0['hi'] - d0['lo']) / (nrm['A'] * Amid)
        for i, n in enumerate(names):
            if n.startswith('qf'):
                coef[j] += Y[j, i] * qc[int(n[2:])] / nrm[n]
        print(f'combination {j}: ' + ' '.join(f'{n}:{v:+.3f}' for n, v in zip(names, Y[j]) if n != 'dT'))
    np.savez(out, coef=coef, Y=Y, names=np.array(names), rjs=np.array(rjs))
    print(f'wrote {out}')


def cmd_horizons(pair, rj, mem, Ts):
    """gh_ah.find_any_ah on pair checkpoints (centred, then about axis points)."""
    from vacuum import gh_ah
    C = Checkpoints(pair, rj)
    for T in Ts:
        G, gh, u, Ti, L, tc = C.state(mem, T)
        t0 = time.time()
        res = gh_ah.find_any_ah(G, gh, u)
        if res is None:
            print(f'T {Ti:5.1f} ({mem}): no apparent horizon [{time.time() - t0:.0f}s]', flush=True)
        else:
            print(f'T {Ti:5.1f} ({mem}): horizon r0 {res["a"][0]:.4f} (physical {L * res["a"][0]:.5f}), '
                  f'centre zc {res["zc"]}, mass {L * res["mass"]:.5f}, residual {res["resid"]:.1e} '
                  f'[{time.time() - t0:.0f}s]', flush=True)


def cmd_axisah(states, rj, mems, Ts):
    """gh_ah.axis_search on checkpoints: apparent horizons and trapped
    spheres about the center and about the axis minimum of the lapse, in the
    frame's X and in X stretched to the local g_XX / g_ZZ."""
    from vacuum import gh_ah
    C = Checkpoints(states, rj)
    for mem in mems:
        for T in Ts:
            G, gh, u, Ti, L, tc = C.state(mem, T)
            zc, s, rows = gh_ah.axis_search(G, gh, u)
            print(f'member {mem} T {Ti:.1f}: axis minimum of the lapse at Z {zc:.3f} (z {zc * L:.5f}), '
                  f'sqrt(g_XX / g_ZZ) {s:.2f}', flush=True)
            for r in rows:
                m = r['mots']
                print(f'  s {r["s"]:5.2f} {"centred" if r["zc"] is None else "about zc"}: min over spheres of '
                      f'max(Theta) r {r["best"][0]:+.2f} (r {r["best"][1]:.3f}), of min(Theta) r {r["low"][0]:+.2f} '
                      f'(r {r["low"][1]:.3f}); ' + ('no MOTS' if m is None else
                      f'MOTS a0 {m["a"][0]:.4f}, mass {L * m["mass"]:.3e}, residual {m["resid"]:.1e}'), flush=True)


def cmd_constraint(states, rj, mems, zr=(0.01, 0.02)):
    """|C_a| (Euclidean over a) of members of a --save-T file: its maximum
    and where, and its maximum on the z range zr (physical) of the deciding
    event next to the axis (x / box < 1e-3), in 2e-3 < x / box < 0.1, and
    beyond 0.1."""
    C = Checkpoints(states, rj)
    for mem in mems:
        G, gh, u, Ti, L, tc = C.state(mem, None)
        c = np.sqrt(np.sum(gh.constraint(u) ** 2, axis=0)).reshape(G.shape)
        x, z = G.ax.x / G.L, G.az.x * L
        i, j = np.unravel_index(np.argmax(c), c.shape)
        ev = (z > zr[0]) & (z < zr[1])
        part = [c[np.ix_(k, ev)].max() for k in (x < 1e-3, (x > 2e-3) & (x < 0.1), x > 0.1)]
        print(f'member {mem} T {Ti:.1f}: max|C| {c.max():.2e} at x/box {x[i]:.3f}, z {z[j]:.4f}; on z '
              f'{zr[0]}-{zr[1]}: x/box < 1e-3 {part[0]:.2e}, 2e-3-0.1 {part[1]:.2e}, > 0.1 {part[2]:.2e}', flush=True)


def cmd_decide(rjs, astar, T0=49.0, T1=58.0):
    """The deciding event on the axis: for every dispersing member of rounds
    written with --axis, the largest physical |E^Z_Z| on the axis in
    T0 <= T <= T1 (records every 0.1), with its T, central proper time,
    physical z and the coordinate FWHM in z of the profile there; members by
    their distance from astar (each grid's own threshold).  Also the slope of
    ln|E| against ln(A - astar)."""
    from scipy.signal import find_peaks
    rows = []
    for rj in rjs:
        d = load_round(rj)
        z = np.load(rj.replace('.json', '_axis.npz'))
        h = d['hist']
        T, L = np.array(h['T']), np.exp(np.array(h['lnL']))
        Ta = np.array(z['T'])
        La = L[np.searchsorted(T, Ta - 1e-9)]
        zp = np.array(z['zax'])[None, :] * La[:, None]
        for m, (A, st, Td) in enumerate(d['members']):
            if st != 'disperse':
                continue
            E = np.abs(np.array(z['eax'])[:, m]) / La[:, None] ** 2
            tc, _, Tm = central(d, m)
            k = np.flatnonzero((Ta >= T0) & (Ta <= min(T1, Td)))
            mx = np.nanmax(E[k], axis=1)
            j = k[int(np.argmax(mx))]
            i = int(np.nanargmax(E[j]))
            row, half = E[j], E[j, i] / 2
            lo, hi = i, i
            while lo > 0 and row[lo] > half:
                lo -= 1
            while hi < len(row) - 1 and row[hi] > half:
                hi += 1
            npk = len(find_peaks(mx, prominence=0.05 * mx.max())[0])
            rows.append((A - astar, Ta[j], float(np.interp(Ta[j], Tm, tc)), zp[j, i], E[j, i], zp[j, hi] - zp[j, lo],
                         npk, f'{os.path.basename(rj)}:{m}'))
    rows.sort()
    for dA, Tj, tcj, zj, Ej, w, npk, src in rows:
        print(f'{dA:+.3e}  T {Tj:5.1f}  tc {tcj:.4f}  z {zj:.4f}  |E| {Ej:.3e}  FWHM {w:.1e}  peaks {npk}  {src}')
    x, y = np.array([r[0] for r in rows]), np.array([r[4] for r in rows])
    k = x > 0
    if k.sum() > 2:
        print(f'd ln|E| / d ln(A - A*) = {np.polyfit(np.log(x[k]), np.log(y[k]), 1)[0]:.2f} over '
              f'[{x[k].min():.1e}, {x[k].max():.1e}]')


def cmd_lobe4(rjs, astar=-3.5091138668, win3=(1.97, 1.99), win4=(1.99, 2.045)):
    """Test for a fourth growing central echo without a collapse criterion:
    for every member of the rounds, the deepest central negative lobe of e_c in
    win4 (where a member within ~1e-7 of a threshold with one unstable mode
    would show its fourth echo) relative to the third lobe (win3), and the
    central proper time at which its record ends; the largest gap between
    consecutive members."""
    rows = []
    for rj in rjs:
        d = load_round(rj)
        for m, (A, st, Td) in enumerate(d['members']):
            tc, e, _ = central(d, m)
            k3 = (tc >= win3[0]) & (tc < win3[1])
            k4 = (tc >= win4[0]) & (tc < win4[1])
            e3 = float(np.min(e[k3])) if k3.any() else np.nan
            e4 = float(np.min(e[k4])) if k4.any() else np.nan
            rows.append((A - astar, st, e3, e4, float(tc[-1]), os.path.basename(rj)))
    rows.sort(key=lambda r: r[0])
    for dA, st, e3, e4, tend, src in rows:
        print(f'{dA:+.3e} {st:9s} e3 {e3:+.3e}  e4 {e4:+.3e}  e4/e3 {e4 / e3:5.2f}  end tc {tend:.4f}  {src}')
    dA = np.array([r[0] for r in rows])
    r = np.array([r[3] / r[2] for r in rows])
    print(f'{len(rows)} members in [{dA[0]:+.2e}, {dA[-1]:+.2e}], largest gap {np.max(np.diff(dA)):.2e}; '
          f'largest e4/e3 {np.nanmax(r):.2f}')


def imax_rows(rjs, astar=-3.5091138668, sigma=0.4):
    """Largest Kretschmann scalar of each member up to its decision, from the
    grid-wide record of a --set kret run (hist Imax, physical values), in
    units of the Brill width sigma (I sigma^4): rows (A - A*, outcome, I_max,
    T, the member's central proper time, physical (rho, z) of the maximum),
    sorted by |A - A*|."""
    rows = []
    for rj in rjs:
        d = load_round(rj)
        h = d['hist']
        IT, Imax, loc = np.array(h['IT']), np.array(h['Imax'], dtype=float), np.array(h['Iloc'], dtype=float)
        T, L = np.array(h['T']), np.exp(np.array(h['lnL']))
        for m, (A, st, Td) in enumerate(d['members']):
            a0 = np.array(h['a0'], dtype=float)[:, m]
            tc = h['tc'][0] + np.r_[0.0, np.cumsum(0.5 * (L[1:] * a0[1:] + L[:-1] * a0[:-1]) * np.diff(T))]
            k = IT <= Td + 1e-9
            j = int(np.nanargmax(Imax[k, m]))
            rows.append((A - astar, st, Imax[j, m] * sigma ** 4, IT[j], float(np.interp(IT[j], T, tc)), *loc[j, m]))
    return sorted(rows, key=lambda r: abs(r[0]))


def cmd_imax(rjs):
    for dA, st, I, Tj, tcj, rho, z in imax_rows(rjs):
        print(f'{dA:+.3e} {st:9s} I_max {I:.4e}  T {Tj:5.1f}  tc {tcj:.4f}  (rho, z) ({rho:.2e}, {z:.2e})')


def main():
    import warnings
    warnings.simplefilter('ignore')
    if len(sys.argv) < 2 or sys.argv[1] in ('-h', '--help'):
        print(__doc__)
        return
    cmd, args = sys.argv[1], sys.argv[2:]
    opt = {'perwin': '--perwin' in args}
    args = [a for a in args if a != '--perwin']
    for k in ('--win', '--period', '--lobes', '--gram', '--cmp', '--floor'):
        if k in args:
            i = args.index(k)
            opt[k[2:]] = args[i + 1]
            del args[i:i + 2]
    if cmd == 'brackets':
        cmd_brackets(args)
    elif cmd == 'central':
        cmd_central(args)
    elif cmd == 'gamma':
        cmd_gamma(args)
    elif cmd == 'response':
        win = tuple(float(v) for v in opt['win'].split(',')) if 'win' in opt else (0.56, 0.68)
        cmd_response(args[0], args[1], args[2], float(args[3]), win,
                     float(opt['period']) if 'period' in opt else None)
    elif cmd == 'lobes':
        cmd_lobes(args)
    elif cmd == 'growth':
        cmd_growth(args, [float(v) for v in opt['lobes'].split(',')] if 'lobes' in opt else None, opt.get('gram'),
                   opt['cmp'].split(',') if 'cmp' in opt else None, opt.get('floor'), opt['perwin'])
    elif cmd == 'axis':
        cmd_axis(args, [float(v) for v in opt['lobes'].split(',')] if 'lobes' in opt else None)
    elif cmd == 'qfree':
        cmd_qfree(args[:-1], args[-1])
    elif cmd == 'gram':
        cmd_gram(args[:-1], args[-1])
    elif cmd == 'departure':
        cmd_departure(args[0], args[1:])
    elif cmd == 'weyl':
        cmd_weyl(args[0], args[1], [float(t) for t in args[2].split(',')])
    elif cmd == 'weylresp':
        cmd_weylresp(args[0], args[1])
    elif cmd == 'combos':
        cmd_combos(args[:-2], opt['gram'], int(args[-2]), args[-1])
    elif cmd == 'imax':
        cmd_imax(args)
    elif cmd == 'lobe4':
        cmd_lobe4(args)
    elif cmd == 'decide':
        cmd_decide(args[1:], float(args[0]))
    elif cmd == 'ricci':
        cmd_ricci(args[0], args[1], [m if m in ('lo', 'hi') else int(m) for m in args[2].split(',')],
                  [float(t) for t in args[3].split(',')] if len(args) > 3 else (None,))
    elif cmd == 'axisah':
        cmd_axisah(args[0], args[1], [int(m) for m in args[2].split(',')], [float(t) for t in args[3].split(',')])
    elif cmd == 'constraint':
        cmd_constraint(args[0], args[1], [m if m in ('lo', 'hi') else int(m) for m in args[2].split(',')])
    elif cmd in ('curvature', 'horizons'):
        f = cmd_curvature if cmd == 'curvature' else cmd_horizons
        f(args[0], args[1], args[2], [float(t) for t in args[3].split(',')])
    else:
        raise SystemExit(f'unknown command {cmd}')


if __name__ == '__main__':
    main()
