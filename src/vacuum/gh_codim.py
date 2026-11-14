#!/usr/bin/env python3
"""Direct codimension test of the centred near-DSS (Sec. V of the paper): add to the initial data of the threshold trajectory multiples of
the input combinations of the untuned growing components found by the
linear response (gh_analysis.py growth), bisect A again at every point, and
see whether the centred phase gets longer.  If those components are
unstable modes of the near-DSS, removing their content from the data must
let the trajectory stay with it for more echoes; if their growth along the
trajectory was a transient, the centred phase does not depend on them.

    python src/vacuum/gh_codim.py knobs out/gh/resp/t24d_T8{,r,ax,dT}_round0.json \
        --out out/gh/codim/knobs.npz
    python src/vacuum/gh_codim.py pair out/gh/codim/knobs.npz --p 1,0 --width 2e-6 \
        --out out/gh/codim/p+1+0_init.npz
    sbatch -J cp10 jobs/gh_bisect.sh out/gh/codim/p+1+0 --init out/gh/codim/p+1+0_init.npz \
        --track out/gh/bis_gpu/t24d_round0.json,out/gh/bis_gpu/t24d_round1.json \
        --rounds 2 --Tend 70 --set hold=40 --set cbad=0.5 --set ckpt=2.0
    python src/vacuum/gh_codim.py report out/gh/codim/p*_round1.json

knobs: rebuilds every direction of the response batches at their T0 (the
same base, family midpoints and seeded random fields as gh_response.py),
takes the joint components of the central response over the last three
echoes, and stores per component j its input combination as a state
perturbation k_j (the right singular vector applied to the unit-RMS
directions), its output norm per echo relative to the base (amp), and its
tuned content (the coefficient of the bracket direction's response in it).
The unit of p_j is 1 / amp_j(last echo): p_j = 1 changes component j's
output in the last centred echo by the base's own norm there.

pair: writes a <out>_pair.npz-format file at T0 for gh_bisect.py --init:
the t24d pair line through u_base + sum_j p_j k_j / amp_j, centred on the
linear prediction of the threshold (the tuned content cancelled by a shift
along the bracket direction), of the given width in A.  The A of a member is
the amplitude of the t24d line it would have without the offset.

Batches of Brill shape directions (gh_response.py --nq or --qcoef, T0 = 0)
give knobs that are initial data: the pair is Brill data with q = q_holz(A)
+ sum_k c_k dq_k = q_holz(A) + q_poly(sum_k c_k coef_k) (exact solutions of
the constraints at any amplitude), the c_k the knobs' weights on the shape
directions, the A content of the knobs and the
tuned-content cancellation both moved into A.  The random-field knobs
violate the constraints at their own amplitude and are kept only for the
record.

report: per round json, the flipping pair's main negative lobes of the
central E^Z_Z (records), the time the curvature peak leaves the centre, and
where the pair separates."""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/


def rebuild(rjs):
    """The base at T0 and every direction of the batches, unit scaled RMS,
    keyed as gh_analysis.response_members (dT left out), with the grid and
    the base's frame and detector state at T0."""
    from vacuum import gh_sim as g
    from vacuum.gh_analysis import load_round
    from vacuum.gh_bisect import CONFIG
    from vacuum.gh_linmap import field_scales
    from vacuum.gh_response import pair_state, random_field
    d0 = load_round(rjs[0])
    a0 = d0['args']
    T0 = d0['T0']
    G = g.grid_from_config(CONFIG)
    lo, hi, fr, pk, tp = pair_state(a0['pair'], T0)
    ub, dA = lo + d0['s_base'] * (hi - lo), hi - lo
    sc = field_scales(G, ub, g.active(G, g.flat_state(G, fr[0])))
    win = np.tile((G.R < a0['rw']).ravel(), g.NA)
    snorm = lambda du: float(np.sqrt(np.mean((du / sc)[win] ** 2)))
    keys, D, qkeys, qcoef, qnorm = ['A'], [dA / snorm(dA)], [], [], []
    for rj in rjs:                                      # in gh_response.py's order of directions
        d = load_round(rj)
        a = d['args']
        assert a['pair'] == a0['pair'] and a['round'] == a0['round'] and d['T0'] == T0, f'{rj}: another base'
        rng = np.random.default_rng(a['seed'])
        for f in a['dir']:
            name = os.path.basename(f).replace('_pair.npz', '')
            if name in keys:
                continue
            l2, h2 = pair_state(f, T0)[:2]
            du = 0.5 * (l2 + h2) - ub
            keys.append(name)
            D.append(du / snorm(du))
        for k in range(a['nrand']):
            du = sc * random_field(G, rng, a['nc'], a['rw'])
            key = f'rand{k}:{a["seed"]}'
            if key not in keys:
                keys.append(key)
                D.append(du / snorm(du))
        qs = slice(*map(int, a['qsel'].split(':'))) if a.get('qsel') else slice(None)
        for nq, cf in ((a.get('nq', 0), None), (0, a.get('qcoef', ''))):
            if not (nq or cf):
                continue
            from vacuum import gh_data as gdat
            from vacuum.gh_response import shape_dirs
            _, gh = g.make(CONFIG)
            Amid = d0['lo'] + d0['s_base'] * (d0['hi'] - d0['lo'])
            if cf:
                cs = np.load(cf)['coef']
            else:
                r2 = np.random.default_rng((a['seed'], 1))
                cs = [gdat.shape_coef(r2, a['qdeg']) for _ in range(nq)]
            for (name, du), cq in zip(shape_dirs(G, gh, CONFIG, Amid, T0, nq, a['qdeg'], a['qh'], a['seed'],
                                                 None if not cf else cs)[qs], list(cs)[qs]):
                key = f'{name}:{a["seed"]}' if name.startswith('bq') else name
                if key in keys:
                    continue
                keys.append(key)
                qkeys.append(key)
                qcoef.append(cq)
                qnorm.append(snorm(du))
                D.append(du / qnorm[-1])
        print(f'  {rj}: {len(keys)} directions so far', flush=True)
    return dict(G=G, T0=T0, ub=ub, dA=dA, nA=snorm(dA), lo=lo, hi=hi, fr=fr, peak=pk, Tpk=tp, keys=keys, D=D,
                pair=a0['pair'], round=a0['round'], s_base=d0['s_base'], qkeys=np.array(qkeys, dtype=str),
                qcoef=np.array(qcoef), qnorm=np.array(qnorm))


def cmd_knobs(rjs, out, ncomp=4):
    from vacuum.gh_analysis import joint_components, load_round, main_lobes, response_members, trivial_basis
    names, tb, e0, R, S, lin = response_members(rjs)
    phys = [i for i, n in enumerate(names) if n != 'dT']
    L = main_lobes(tb, e0)
    ech = list(zip(L[:-1], L[1:]))[-3:]
    vt, amp = joint_components(tb, e0, R[:, phys], ech)
    # tuned content: coefficient of R_A in each direction's response over the last two echoes
    F = trivial_basis(tb, e0)
    k = (tb >= ech[-2][0]) & (tb < ech[-1][1])
    Q = np.linalg.qr(F[k])[0]
    Rk = R[k] - Q @ (Q.T @ R[k])
    iA = names.index('A')
    c = Rk[:, iA] @ Rk / (Rk[:, iA] @ Rk[:, iA])
    b = rebuild(rjs)
    assert [names[i] for i in phys] == b['keys'], f'direction order: {[names[i] for i in phys]} vs {b["keys"]}'
    D = np.array(b.pop('D'))
    K = vt[:ncomp] @ D                                  # input combinations as state perturbations
    tuned = vt[:ncomp] @ c[phys]
    print('echoes ' + ' '.join(f'[{a:.4f},{b_:.4f})' for a, b_ in ech))
    for j in range(ncomp):
        print(f'  component {j}: output / base norm per echo ' + ' '.join(f'{v:.3g}' for v in amp[j])
              + f'; growth ' + ' '.join(f'{amp[j, n + 1] / amp[j, n]:.2f}' for n in range(amp.shape[1] - 1))
              + f'; tuned content {tuned[j]:+.3g}; the knob p = 1 is {1 / amp[j, -1]:.3e} of unit scaled RMS')
    G = b.pop('G')
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    np.savez(out, K=K, amp=amp, tuned=tuned, vt=vt[:ncomp], names=np.array([names[i] for i in phys]),
             ech=np.array(ech), shape=np.array(G.shape), rjs=np.array(rjs),
             **{k: np.asarray(v) for k, v in b.items() if k != 'keys'})


def cmd_pair(knobs, p, width, out, shift=0.0, grid=None):
    from vacuum.gh_analysis import load_round
    z = np.load(knobs)
    K, tuned = z['K'], z['tuned']
    amp = z['amp'][:len(K)]
    p = np.array([0.0] + p + [0.0] * (len(K) - 1 - len(p)))   # component 0 is the tuned one: not a knob
    q = p / amp[:, -1]
    # the tuned content of the offset, cancelled along the bracket direction dA (unit RMS dA / nA)
    ds = -(q @ tuned) / float(z['nA'])
    d = load_round(str(z['round']))
    Alo, Ahi = d['lo'], d['hi']                         # the t24d pair line: u(A) = lo + (A - Alo) / (Ahi - Alo) dA
    w = Ahi - Alo
    half = 0.5 * width / abs(w)
    fr = z['fr']
    if 'qcoef' in z.files and len(z['qcoef']):         # Brill knobs: exact initial data
        from vacuum import gh_data as gdat
        from vacuum import gh_sim as g
        from vacuum.gh_bisect import CONFIG
        if grid:                                        # the same data on another grid (resolution checks)
            CONFIG = dict(CONFIG, n=grid[0], nz=grid[1])
        names = list(z['names'])
        ck = (q @ z['vt'][:, [names.index(k) for k in z['qkeys']]]) / z['qnorm']   # weights of the dq_k
        ds += (q @ z['vt'][:, names.index('A')]) / float(z['nA'])
        s0 = float(z['s_base']) + ds + shift / w
        A0 = Alo + s0 * w
        G, gh = g.make(CONFIG)
        Aq = Alo + float(z['s_base']) * w                 # the A of the shape directions (gh_response.shape_dirs)
        extra = ((1.0, gdat.q_poly(np.tensordot(ck, z['qcoef'], 1), Aq, CONFIG['sigma'])),)
        bd = lambda A: g.active(G, gdat.brill_data(G, A, CONFIG['sigma'], ell=CONFIG['ell0'], gh=gh, nr=CONFIG['nr'],
                                                   nmu=CONFIG['nmu'], extra=extra)[0])
        ulo, uhi = bd(Alo + (s0 - half) * w), bd(Alo + (s0 + half) * w)
        print('shape weights ' + ' '.join(f'{c:+.2e}' for c in ck))
    else:
        off = q @ K
        s0 = float(z['s_base']) + ds + shift / w
        A0 = Alo + s0 * w
        ulo = z['lo'] + (s0 - half) * z['dA'] + off
        uhi = z['lo'] + (s0 + half) * z['dA'] + off
    np.savez(out, T=np.array([float(z['T0'])]), ic=0, lo=A0 - 0.5 * width * np.sign(w), hi=A0 + 0.5 * width * np.sign(w),
             round=-1, u_lo=ulo[None], u_hi=uhi[None], peak=np.array([float(z['peak'])]), Tpk=np.array([float(z['Tpk'])]),
             rel=np.array([0.0]), p=p, **{f'fr_{k}': np.array([float(fr[j])]) for j, k in enumerate(('ell', 'lnL', 'tc', 'c'))})
    print(f'p = {p.tolist()}: offset scaled weights {q.tolist()}; predicted threshold A = {A0:.13f} '
          f'(shift {ds * w + shift:+.3e} from the base, {shift:+.1e} of it given); bracket width {width:.1e}', flush=True)


def centred_end(tc, e, t0=1.6, t1=np.inf):
    """Where the growth of the central curvature stops (the criterion of
    Sec. V of the paper): among the record negative lobes of e_c(tc)
    after t0 (|e| above every earlier negative extremum), the first one, R,
    that no later negative extremum up to t1 exceeds by a factor 2.  Returns
    R, the chain of echo lobes before it (each record at most half the next
    one in the chain), and the largest negative extremum after R up to t1."""
    from vacuum.gh_analysis import extrema
    ex = extrema(tc, e)
    neg = ex[(ex[:, 1] < 0) & (ex[:, 0] > t0) & (ex[:, 0] <= t1)]
    rec, top = [], 0.0
    for t, v in neg:
        if abs(v) > top:
            rec.append((t, v))
            top = abs(v)
    if not rec:
        return None, [], None
    k = next(i for i, (t, v) in enumerate(rec) if not (np.abs(neg[neg[:, 0] > t, 1]) > 2 * abs(v)).any())
    chain = [rec[k]]
    for t, v in reversed(rec[:k]):
        if abs(v) <= 0.5 * abs(chain[0][1]):
            chain.insert(0, (t, v))
    after = neg[neg[:, 0] > rec[k][0]]
    later = tuple(after[np.argmax(np.abs(after[:, 1]))]) if len(after) else None
    return rec[k], chain[:-1], later


def separation(sa, sb, t0=1.6):
    """The central proper time at which two members' e_c (splines) differ by
    0.3 of their local RMS (window 0.008), or the end of the shorter record
    (a member's decision) if they never do before it."""
    tg = np.linspace(max(sa.x[0], sb.x[0], t0), min(sa.x[-1], sb.x[-1]), 6000)
    a, b = sa(tg), sb(tg)
    ref = 0.5 * (a + b)
    env = np.array([np.sqrt(np.mean(ref[(tg > t - 0.004) & (tg < t + 0.004)] ** 2)) for t in tg])
    k = np.abs(a - b) / env > 0.3
    return tg[np.argmax(k)] if k.any() else tg[-1]


def pair_members(arg, quiet=False):
    """The (round, member) list of a report argument: 'file' (the flip pair),
    'file:i,j' or 'file:i@file2:j'."""
    from vacuum.gh_analysis import load_round
    specs = []
    for part in arg.split('@'):
        rj, _, ms = part.partition(':')
        d = load_round(rj)
        st = [m[1] for m in d['members']]
        fl = [k for k in range(len(st) - 1) if st[k] == 'collapse' and st[k + 1] == 'disperse']
        if not quiet:
            print(f'{rj}: ' + ''.join(x[0] for x in st) + (f'  flip {fl[0]}, {fl[0] + 1}: A* = '
                  f'{0.5 * (d["members"][fl[0]][0] + d["members"][fl[0] + 1][0]):.13f}' if len(fl) == 1
                  else '  (no single flip)'))
        mem = [int(x) for x in ms.split(',')] if ms else (fl[0], fl[0] + 1) if len(fl) == 1 else ()
        specs += [(d, m) for m in mem]
    return specs


def pair_curves(specs):
    """Central (tc, e_c, T) of the members and where the pair separates."""
    from scipy.interpolate import CubicSpline
    from vacuum.gh_analysis import central
    cur = [central(d, m) for d, m in specs]
    tsep = separation(*[CubicSpline(tc, e) for tc, e, _ in cur]) if len(cur) == 2 else np.inf
    return cur, tsep


def scale_factor(lobes, ref):
    """The factor lam with lobe times ~ lam x the reference's (least squares
    over the last three lobes of each; (t, e) rows)."""
    t, r = np.asarray(lobes)[-3:, 0], np.asarray(ref)[-3:, 0]
    return float(t @ r / (r @ r))


def cmd_report(rjs, ref=''):
    """Per round json: the flip pair's central negative lobes after tc = 1.9
    (|e| > 1e3), where the pair separates, the lobe where the growth of the
    central curvature stops before that (centred_end), the ratio of the
    largest later negative lobe before the separation to it (the continuous
    readout) and of the last two spacings of the echo lobes, and when the
    K_ij K^ij maximum first leaves the centre (r > 0.005) after tc = 1.95.
    'file:i,j' reads the members i, j instead of the flip pair, and
    'file:i@file2:j' pairs members of two rounds (when every member of a
    round went one way, the threshold lies between its last member and an
    endpoint, a member of the round before).  With ref (the same syntax, its
    first member the reference, e.g. the t24d flip pair): the factor lam that
    maps the last three echo lobes onto the reference's by tc -> tc / lam
    (least squares; the data's scale enters the evolution only this way), the
    lobe times divided by it, and the scale-free heights lam^2 e / e_ref."""
    from vacuum.gh_analysis import extrema
    rch = None
    if ref:
        cr, ts = pair_curves(pair_members(ref, quiet=True))
        last, chain, _ = centred_end(cr[0][0], cr[0][1], t1=ts)
        rch = np.array(chain + [last])[-3:]
        print('reference lobes ' + ' '.join(f'{t:.4f}:{v:.4g}' for t, v in rch))
    for arg in rjs:
        specs = pair_members(arg)
        if not specs:
            continue
        cur, tsep = pair_curves(specs)
        for (d, m), (tc, e, Tm) in zip(specs, cur):
            h = d['hist']
            T = np.array(h['T'])
            ex = extrema(tc, e)
            neg = ex[(ex[:, 1] < -1e3) & (ex[:, 0] > 1.9)]
            last, chain, later = centred_end(tc, e, t1=tsep)
            if last is None:
                print(f'   member {m}: no negative lobe after tc = 1.6 before the separation')
                continue
            rec = chain + [last]
            ratio = f'{later[1] / last[1]:.3f} at {later[0]:.4f}' if later else '-'
            srat = f'{(rec[-1][0] - rec[-2][0]) / (rec[-2][0] - rec[-3][0]):.3f}' if len(rec) >= 3 else '-'
            grow = f'{last[1] / rec[-2][1]:.2f}' if len(rec) >= 2 else '-'
            off = '-'
            if 'kkloc' in h:
                # (in the earlier echoes the K_ij K^ij maximum moves between the centre and the axis)
                r = np.hypot(*np.array([x[m] for x in h['kkloc']]).T)
                tcT = np.interp(T, Tm, tc)
                ii = [k for k in range(1, len(T)) if r[k] > 0.005 and tcT[k] > 1.95 and T[k] <= Tm[-1]]
                off = f'{tcT[ii[0]]:.4f}' if ii else '-'
            st = d['members'][m]
            print(f'   member {m} (A = {st[0]:.12f}, {st[1]} at T = {st[2]:.1f}): negative lobes '
                  + ' '.join(f'{t:.4f}:{v:.3g}' for t, v in neg[:8]))
            print(f'      growth stops at {last[0]:.4f}:{last[1]:.4g} (x {grow} over the echo before); largest later '
                  f'negative lobe / it {ratio}; last spacing ratio {srat}; K peak first at r > 0.005 after tc 1.95 '
                  f'at tc {off}')
            if rch is not None and len(rec) >= 3:
                t3, e3 = np.array(rec[-3:]).T
                lam = scale_factor(rec, rch)
                print(f'      lam = {lam:.4f}: lobes / lam ' + ' '.join(f'{t:.4f}' for t in t3 / lam)
                      + ' (largest offset from the reference ' + f'{np.max(np.abs(t3 / lam - rch[:, 0])):.4f}), '
                      + 'lam^2 e / e_ref ' + ' '.join(f'{x:.2f}' for x in lam ** 2 * e3 / rch[:, 1]))
        print(f'   pair separates (0.3 of the local RMS) at tc {tsep:.4f}' if len(cur) == 2 else
              '   (one member: lobes up to the end of its record)')


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = p.add_subparsers(dest='cmd', required=True)
    k = sub.add_parser('knobs')
    k.add_argument('rjs', nargs='+')
    k.add_argument('--ncomp', type=int, default=4)
    k.add_argument('--out', required=True)
    q = sub.add_parser('pair')
    q.add_argument('knobs')
    q.add_argument('--p', default='', help='knob amplitudes of components 1, 2, ... (comma separated)')
    q.add_argument('--width', type=float, default=2e-6, help='bracket width in A')
    q.add_argument('--shift', type=float, default=0.0,
                   help='extra shift of the bracket centre in A (the second-order threshold shift, which the '
                        'linear cancellation leaves out: at the knob amplitudes of the Brill knobs it is 1e-4)')
    q.add_argument('--grid', default='', metavar='N,NZ',
                   help='Brill knobs: build the pair on an N x NZ grid instead of the production one (run it with '
                        'gh_bisect.py --set n=N --set nz=NZ)')
    q.add_argument('--out', required=True)
    r = sub.add_parser('report')
    r.add_argument('rjs', nargs='+')
    r.add_argument('--ref', default='', help='reference pair for the time-scale factor (e.g. the t24d flip pair)')
    a = p.parse_args()
    import warnings
    warnings.simplefilter('ignore')
    if a.cmd == 'knobs':
        cmd_knobs(a.rjs, a.out, a.ncomp)
    elif a.cmd == 'pair':
        cmd_pair(a.knobs, [float(x) for x in a.p.split(',')] if a.p else [], a.width, a.out, a.shift,
                 [int(x) for x in a.grid.split(',')] if a.grid else None)
    else:
        cmd_report(a.rjs, a.ref)


if __name__ == '__main__':
    main()
