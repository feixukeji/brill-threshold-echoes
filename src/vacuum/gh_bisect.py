#!/usr/bin/env python3
"""Bisection of a Brill family toward the black-hole threshold on one GPU:
B members evolved together as one batch (gh_cuda.CudaGH), each round
replacing the bracket by the adjacent pair of members with different
outcomes.

    python src/vacuum/gh_bisect.py --eta 0.85 --lo -3.6215 --hi -3.616 \
        --rounds 4 --out out/gh/bis_gpu/h085

Round 0 starts from the Brill data q_holz(A, sigma, eta) at T = 0 (or from
the pair of --init).  Every later round starts from the bracketing pair's
states at the last checkpoint Tc where they still differ by less than dtol
(relative, active vector): the members are the straight line
S_lo + s (S_hi - S_lo), s = k / (B + 1), so the depth of the bisection is not
limited by the precision of A and a round only evolves from Tc on.  The
bracket is reported as the amplitude A_lo + s (A_hi - A_lo) (to linear order
the same family) and its width w.

Frame and grid (CONFIG): zoom rate ell0 up to T = Ts, then locked to the
central proper time tc = int L alpha_c dT (alpha_c the median central lapse
of the active members): L = c (tc* - tc) with c fixed at Ts, i.e.
ell = L alpha_c / (tc* - tc) + k ln(L / (c (tc* - tc))), clipped to
[lmin, lmax] (lmin raised whenever the outflow margin falls below mmin), each
change an exact reframe.  From T = hold on (if set) the clock no longer
focuses: ell = lmin, the slowest zoom that keeps the outer face spacelike,
so a phase at a fixed physical scale stays inside the box after tc passes
tc* (a --track's frames from hold on are not replayed).  At T = Tg every
member moves to a grid with a finer axis (X map ax2).

Determinism: a member's evolution must not depend on the batch it is in or
on the time its round started.  Three things otherwise would: the frame (the
median central lapse), the step (the CFL limit is the batch minimum) and the
age of the frozen LU that preconditions the fixed-count maximal-lapse solve.
So every round records the zoom rate and the substep count of each step and
the next round replays them (the controller takes over where the record
ends; a recorded count is used while it is within 3% of the CFL need), and
the LU is refrozen on an absolute schedule: every relu refreshes from the
start time of round 0 (from Tg after the regrid).  Restarts are taken only
at checkpoints on that schedule.

Outcomes, from quantities the GPU has at every refresh: 'collapse' if
max |C_a| > cmax, min alpha < acoll or the state (or its CFL step) is not
finite, each logged with the criterion and min alpha; a non-finite member
whose last finite max K_ij K^ij sat at x > xfail times the box radius is
'fail' instead (the blow-up of the X lines' outer end); 'disperse'
once max K_ij K^ij has stayed below kdisp times its peak for dhold (and at
least 2 after the peak).  The new bracket is the pair of decided members
(endpoints included) where the outcome flips; undecided and failed members
are passed over.  The bisection stops once the bracketing pair's
max |C_a| exceeded cbad up to 1 before the dispersing member's decision or
2.5 before the collapsing member's.

Writes <out>_round<k>.json (rewritten at every progress line while the round
runs, undecided members as 'active'): args, template (the configuration), bracket,
members [(A, outcome, T_end)] and hist, per refresh (every mrefresh): T,
ell, lnL, tc of the frame; per member amin, a0 (central lapse), C (max |C_a|),
KK (max K_ij K^ij), ez (central E^Z_Z at L = 1, physical value / L^2), kkloc
and aloc ((L X, L Z) of max K_ij K^ij and of min alpha); steps [(T, substeps)];
with --set efine=K also fT, flnL, fez, fa0: ez and a0 every K substeps and at
the end of every step (the refresh samples are too sparse for d e_c / d tc in
the last centred echoes).
<out>_pair.npz: the bracketing pair's checkpoints (T, u_lo, u_hi active
vectors) with the frame state (fr_ell, fr_lnL, fr_tc, fr_c) of the last
round that had a single flip."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/

# the production configuration: L = 24 (60 sigma) box, n x nz = 400 x 960,
# damped-wave gauge with the maximal-lapse driver, tclock frame focused on
# tc* = 2.147 from T = 12, axis refinement at T = 20
CONFIG = dict(
    n=400, nz=960, L=24.0, ax=6.33, az=5.37, b=0.995, filtp=16,                 # grid
    gamma0=20.0, g0p=-1.0, muL=3.0, muS=15.0, pL=0.0, qL=-1.0,                   # gauge
    maximal=3.0, cK=1.0, mgrid=96, mfixed=40, afloor=1e-3,                       # maximal-lapse target
    sigma=0.4, nr=48, nmu=20,                                                    # Brill data
    ell0=0.0965, cfl=1.2, dT=0.05, mrefresh=0.1, relu=8,                         # stepping
    tclock=[12.0, 2.147, 0.2, 0.042, 0.4, 0.05],                                 # Ts, tc*, k, lmin, lmax, mmin
    regrid=[20.0, 400, 8.0],                                                     # Tg, n2, ax2
    regrid2=None,                   # [T2, n3, nz3]: from T2 on the n3 x nz3 grid (event 4 needs X nodes)
    gauge2=None,                    # with regrid2: gauge overrides from T2 on, e.g. {"muS": 3.0} (P_T re-solved there)
    acoll=1e-4, cmax=3.0, kdisp=3e-3, dhold=3.0, xfail=0.5,                      # outcomes
    ckpt=0.5, dtol=1e-4, cbad=0.05,                                              # restarts
    hold=None,                                                                   # T: ell = lmin from then on
    mchk=20, ltrack=False,          # outflow check every mchk refreshes; ltrack: lmin = the smallest outflow rate
    kphys=False,                    # dispersal test on K_ij K^ij / L^2 (a zooming cut box, --recentre BOX)
    efine=None,                     # K: also record e_c and the central lapse every K substeps
    kret=None,                      # K: also record max / min of the Kretschmann scalar over the grid every K refreshes
    lines=None)                     # K: also record g, d_T g and the Kretschmann scalar on the axis and the equator every K refreshes


class Track:
    """Recorded zoom rates {T: ell} (from Ts on) and substep counts {T: n}
    of the steps from T, replayed by later rounds."""

    def __init__(self):
        self.ell, self.nsub = {}, {}

    def add(self, hist, Ts):
        self.ell.update({round(T, 6): e for T, e in zip(hist['T'], hist['ell']) if T >= Ts - 1e-9})
        self.nsub.update({round(T, 6): int(n) for T, n in hist.get('steps', [])})


def regridder(E, E2, cfg, gauge=None):
    """The change to the grid of E2 (same chart): spectral interpolation of
    every member's active fields (gh_sim.transfer_fields); with gauge (E2 has
    other gauge parameters) P_T is re-solved from C_a = 0."""
    from vacuum import gh_sim as g

    def to_new(S):
        t1 = time.time()
        ell = E.ell
        U = E.to_active(S).reshape(S.shape[0], g.NA, *E.G.shape)
        V = np.array([g.transfer_fields(E.G, F, E2.G, ell) for F in U])
        E2.set_ell(ell)
        S2 = E2.from_active(V.reshape(S.shape[0], -1))
        E2.freeze_lu(S2)
        if gauge:                       # a new gauge source: P_T from C_a = 0 in it
            S2 = E2.solve_PT(S2, E2.maximal(S2))
        cm = float(E2.constraint(S2, E2.maximal(S2)).abs().amax())
        print(f'   regridded to {E2.G.shape} (ax {E2.G.ax.a}), ell {ell:.4f}' + (f', gauge {gauge}' if gauge else '')
              + f': max|C| {cm:.2e} [{time.time() - t1:.0f}s]', flush=True)
        return S2
    return to_new


def evolve_round(E, S, T0, fr, peak0, Tpk0, cfg, Tend, track, sched, switch=None, axis=None, dump=None,
                 recentre=None, save=None):
    """Evolve the batch S (CudaGH E) from T0 until every member is decided
    (or Tend).  fr = (ell, lnL, tc, c) the frame state at T0, (peak0, Tpk0)
    the outcome detector's peak so far; track: the replay (Track);
    sched(T): True on the LU refreeze schedule; switch = [(Tg, E2, f), ...]:
    at T = Tg the batch moves to E2 as f(S); axis(G): the Z node indices at
    which hist['eax'] records E^Z_Z on the axis (CudaGH.axis_E) at every
    refresh (hist['eax_rc'] after a recentring);
    dump(status, tend, hist): called at every progress line (partial output);
    recentre = (Trc, f, tcs'): at T = Trc the batch moves to the chart
    centred on an axis point, (E, S, zc) = f(E, S, L) with zc its physical z;
    the frame clock restarts there (tc = 0 on the new centre, focus tcs',
    the track's later frames and substeps dropped); kkloc / aloc stay in the
    old physical z; save = (Ts, f): f(T, E, S, peak, Tpk, frame) at the
    refresh at T = Ts, after any reframe.
    Returns status, tend, hist, the
    checkpoints {member: [(T, u, peak, Tpk)]} taken every ckpt while the
    member is active, and the frame states {T: (ell, lnL, tc, c)} there."""
    from vacuum import gh_sim as g
    mref, Ts, tcs, kc, lmin, lmax, mmin = cfg['mrefresh'], *cfg['tclock']
    dT = cfg['dT']
    B = S.shape[0]
    ell, lnL, tc, c = fr
    E.set_ell(ell)
    E.freeze_lu(S)
    status, tend = [''] * B, np.full(B, np.nan)
    peak, Tpk = np.full(B, peak0), np.full(B, Tpk0)
    if cfg.get('kphys'):                # the restart peak is in frame units (L at T0 stands for L at its time)
        peak = peak / np.exp(2 * lnL)
    dlow = np.full(B, np.nan)           # start of the current stretch with KK < kdisp peak
    hist = {k: [] for k in ('T', 'amin', 'a0', 'C', 'KK', 'ell', 'lnL', 'tc', 'ez', 'steps', 'kkloc', 'aloc')}
    fine = cfg.get('efine')             # also e_c and the central lapse every efine substeps (and at every step's end)
    if fine:
        hist.update(fT=[], flnL=[], fez=[], fa0=[])
    ck = {b: [] for b in range(B)}
    frames = {}
    T, lastm, nref, lam, prev = T0, -1e9, 0, None, None
    zc, eax = 0.0, 'eax'                # physical z of the frame centre; key of the axis record
    cut = False                         # in a cut box (--recentre BOX)
    t0 = time.time()

    def died(b, why):                   # a non-finite member: collapse, or a blow-up at the outer end
        j = next((j for j in range(len(hist['KK']) - 1, -1, -1) if np.isfinite(hist['KK'][j][b])), None)
        x = hist['kkloc'][j][b][0] / (np.exp(hist['lnL'][j]) * E.G.L) if j is not None else 0.0
        status[b], tend[b] = ('fail' if x > cfg['xfail'] else 'collapse'), T
        print(f'  T = {T:.2f}: member {b} {status[b]} ({why}; last finite K_ij K^ij peak at x / box {x:.2f}), '
              f'min alpha at the last refresh {d["amin"][b]:.4f}', flush=True)

    while T < Tend - 1e-9 and '' in status:
        act = [b for b in range(B) if status[b] == '']
        for sw in switch or ():
            if abs(T - sw[0]) < 0.5 * dT:
                E, S = sw[1], sw[2](S)
                lastm, nref = -1e9, 0
                print(f'  T = {T:.2f}: grid {E.G.shape} [{time.time() - t0:.0f}s]', flush=True)
        if recentre is not None and abs(T - recentre[0]) < 0.5 * dT:
            L0 = E.G.L
            E, S, zc, *lrc = recentre[1](E, S, np.exp(lnL))
            lastm, nref, eax = -1e9, 0, 'eax_rc'
            tc, tcs, c, prev = 0.0, recentre[2], float('nan'), None
            # the outer face of the smaller box stays spacelike (a cut box: the rate found there)
            lmin = lrc[0] if lrc else lmin * L0 / E.G.L
            cut = bool(lrc)
            track.ell = {k: v for k, v in track.ell.items() if k < T - 1e-9}
            track.nsub = {k: v for k, v in track.nsub.items() if k < T - 1e-9}
            print(f'  T = {T:.2f}: recentred on z = {zc:.5f}: grid {E.G.shape}, box {E.G.L}, lmin {lmin:.4f}, '
                  f'clock from 0 on the new centre, focus {tcs} [{time.time() - t0:.0f}s]', flush=True)
        if T - lastm >= mref - 1e-9:
            if nref and sched(T):
                E.freeze_lu(S)
            lam = E.maximal(S)
            lastm, nref = T, nref + 1
            d = {k: v.cpu().numpy() for k, v in E.diag.items()}
            ac = float(np.median(d['a0'][act]))
            if prev is not None:
                tc += 0.5 * (prev[1] + np.exp(lnL) * ac) * (T - prev[0])
            prev = (T, np.exp(lnL) * ac)
            if T >= Ts - 1e-9:
                if not np.isfinite(c):
                    c = np.exp(lnL) / (tcs - tc)
                    print(f'  T = {T:.2f}: tclock on, tc = {tc:.5f}, c = {c:.4f} (focus tc* = {tcs})', flush=True)
                if (nref - 1) % cfg.get('mchk', 20) == 0:
                    ug = g.full_state(E.G, E.to_active(S[act[0]:act[0] + 1])[0], E.ell)
                    mg = g.outflow_margin(E.G, ug)
                    if cut and cfg.get('ltrack'):   # follow the faces' light speed both ways
                        lnew = g.outflow_rate(E.G, ug, 2 * mmin)
                        if abs(lnew / lmin - 1) > 0.05:
                            print(f'  T = {T:.2f}: outflow margin {mg:.3f}: lmin {lmin:.4f} -> {lnew:.4f}', flush=True)
                        lmin = lnew
                    elif mg < mmin:
                        lmin = max(lmin, 1.05 * E.ell)
                        print(f'  T = {T:.2f}: outflow margin {mg:.3f} < {mmin}: lmin -> {lmin:.4f}', flush=True)
                rem = max(tcs - tc, 1e-12)
                ell_new = float(np.clip(np.exp(lnL) * ac / rem + kc * (lnL - np.log(c * rem)), lmin, lmax))
                if cfg.get('hold') is not None and T >= cfg['hold'] - 1e-9:
                    ell_new = lmin
                rec = track.ell.get(round(T, 6))
                if rec is not None and rec != E.ell or rec is None and abs(ell_new - E.ell) > 1e-5:
                    S = E.reframe(S, lam, ell_new if rec is None else rec)
            for k in ('amin', 'a0', 'C', 'KK'):
                hist[k].append(d[k].tolist())
            try:
                hist['ez'].append(E.central_E(S, lam).tolist())
            except Exception:           # a failed member makes the batch's Weyl assembly singular
                hist['ez'].append([float('nan')] * B)
            if axis is not None:
                iz = axis(E.G)
                try:
                    ea = E.axis_E(S, lam, iz)
                except Exception:
                    ea = np.full((B, len(iz)), np.nan)
                hist.setdefault(eax, []).append(ea)
                hist.setdefault(eax + '_T', []).append(T)
            Xf, Zf, Lf = E.G.X.ravel(), E.G.Z.ravel(), np.exp(lnL)
            if cfg.get('lines') and (nref - 1) % cfg['lines'] == 0:   # frame values (L = 1)
                try:
                    Il = E.kretschmann(S, lam).reshape((B,) + E.G.shape)
                    Il = (Il[:, 0, :].cpu().numpy(), Il[:, :, E.ic].cpu().numpy())
                except Exception as exc:
                    print(f'  T = {T:.2f}: Kretschmann failed: {exc!r}', flush=True)
                    Il = (np.full((B, E.G.shape[1]), np.nan), np.full((B, E.G.shape[0]), np.nan))
                Sl = S[:, :2 * g.NG]
                for k, v in (('T', T), ('lnL', lnL), ('ell', E.ell), ('tc', tc), ('X', E.G.ax.x.copy()),
                             ('ax', Sl[:, :, 0, :].cpu().numpy()), ('eq', Sl[:, :, :, E.ic].cpu().numpy()),
                             ('Iax', Il[0]), ('Ieq', Il[1])):
                    hist.setdefault('eaxl_' + k, []).append(v)
            if cfg.get('kret') and (nref - 1) % cfg['kret'] == 0:     # physical values
                import torch
                try:
                    I = E.kretschmann(S, lam) / Lf ** 4
                    I = torch.where(torch.isfinite(I), I, torch.zeros_like(I))
                    imx, imn = I.max(dim=1), I.min(dim=1)
                    hist.setdefault('Imax', []).append(imx.values.cpu().numpy().tolist())
                    hist.setdefault('Imin', []).append(imn.values.cpu().numpy().tolist())
                    hist.setdefault('Iloc', []).append([[Lf * float(Xf[i]), zc + Lf * float(Zf[i])]
                                                        for i in imx.indices.cpu().numpy()])
                except Exception as exc:
                    print(f'  T = {T:.2f}: Kretschmann failed: {exc!r}', flush=True)
                    for k in ('Imax', 'Imin'):
                        hist.setdefault(k, []).append([float('nan')] * B)
                    hist.setdefault('Iloc', []).append([[float('nan')] * 2] * B)
                hist.setdefault('IT', []).append(T)
            for k, kk in (('kkloc', 'kki'), ('aloc', 'ami')):
                hist[k].append([[Lf * float(Xf[i]), zc + Lf * float(Zf[i])] for i in d[kk].astype(int)])
            for k, v in (('T', T), ('ell', E.ell), ('lnL', lnL), ('tc', tc)):
                hist[k].append(v)
            kk = d['KK'] / (np.exp(2 * lnL) if cfg.get('kphys') else 1.0)
            for b in act:
                if not np.isfinite(d['C'][b]):
                    died(b, 'not finite')
                elif d['C'][b] > cfg['cmax'] or d['amin'][b] < cfg['acoll']:
                    status[b], tend[b] = 'collapse', T
                    why = 'alpha' if d['amin'][b] < cfg['acoll'] else 'C'
                    print(f'  T = {T:.2f}: member {b} collapse ({why}): max|C| {d["C"][b]:.3g}, '
                          f'min alpha {d["amin"][b]:.4f}, KK {d["KK"][b]:.4g}', flush=True)
                elif kk[b] > peak[b]:
                    peak[b], Tpk[b], dlow[b] = kk[b], T, np.nan
                elif T > 3 and kk[b] < cfg['kdisp'] * peak[b] and T > Tpk[b] + 2:
                    # dispersal once KK has stayed low for dhold (a dip between
                    # echoes, then a later collapse, is not a dispersal)
                    if not np.isfinite(dlow[b]):
                        dlow[b] = T
                    if T >= dlow[b] + cfg['dhold'] - 1e-9:
                        status[b], tend[b] = 'disperse', T
                else:
                    dlow[b] = np.nan
            act = [b for b in range(B) if status[b] == '']
            if not act:
                break
            if abs(T / cfg['ckpt'] - round(T / cfg['ckpt'])) < 1e-6:
                u = E.to_active(S)
                for b in act:
                    ck[b].append((T, u[b].copy(), peak[b], Tpk[b]))
                frames[round(T, 6)] = (E.ell, lnL, tc, c)
            if save is not None and abs(T - save[0]) < 0.5 * dT:
                save[1](T, E, S, peak, Tpk, (E.ell, lnL, tc, c))
            for b in range(B):          # finished members follow an active one (keeps the batch finite)
                if status[b] != '':
                    S[b] = S[act[0]]
                    lam[b] = lam[act[0]]
            if int(round(T / mref)) % 20 == 0:
                print(f'  T = {T:.2f}: active {len(act)}, amin ' + ' '.join(f'{x:.3f}' for x in d['amin'][act])
                      + f' [{time.time() - t0:.0f}s]', flush=True)
                if dump is not None:
                    try:
                        dump(status, tend, hist)
                    except Exception as err:    # partial output only; never stop the run for it
                        print(f'  partial dump failed: {err!r}', flush=True)
        dts = E.cfl_dt(S, cfg['cfl']).cpu().numpy()
        for b in [b for b in act if not np.isfinite(dts[b])]:   # failed between refreshes
            died(b, 'step not finite')
        act = [b for b in range(B) if status[b] == '']
        if not act:
            break
        for b in range(B):
            if status[b] != '':
                S[b] = S[act[0]]
                lam[b] = lam[act[0]]
        nsub = max(1, int(np.ceil(dT / float(np.min(dts[act])))))
        rec = track.nsub.get(round(T, 6))
        if rec is not None and rec < np.ceil(0.97 * nsub):
            print(f'  T = {T:.2f}: CFL needs {nsub} substeps, the track has {rec}', flush=True)
        elif rec is not None:
            nsub = rec
        hist['steps'].append((T, nsub))
        for i in range(nsub):
            S = E.step(S, lam, dT / nsub)
            if fine and ((i + 1) % fine == 0 or i == nsub - 1):
                try:
                    ez, a0 = E.central_E(S, lam, alpha=True)
                except Exception:       # a failed member makes the batch's Weyl assembly singular
                    ez = a0 = np.full(B, np.nan)
                hist['fT'].append(T + (i + 1) * dT / nsub)
                hist['flnL'].append(lnL - E.ell * (i + 1) * dT / nsub)
                hist['fez'].append(ez.tolist())
                hist['fa0'].append(a0.tolist())
        lnL -= E.ell * dT
        T += dT
    status = [s or 'undecided' for s in status]
    return status, tend, hist, ck, frames


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--lo', type=float, help='amplitude on the collapse side')
    p.add_argument('--hi', type=float, help='amplitude on the dispersal side')
    p.add_argument('--eta', type=float, default=1.0, help='Brill shape: axial over radial width (gh_data.q_holz)')
    p.add_argument('--eps', type=float, default=0.0, help='Brill shape: factor 1 + eps z/sigma (needs --full); '
                   'to first order a z-translation, not a physical odd direction')
    p.add_argument('--full', action='store_true', help='full-Z grid (no equatorial symmetry), 2 nz nodes on [-L, L]')
    p.add_argument('--B', type=int, default=16)
    p.add_argument('--rounds', type=int, default=6)
    p.add_argument('--Tend', type=float, default=150.0)
    p.add_argument('--set', action='append', default=[], metavar='KEY=VALUE',
                   help='override a CONFIG entry (JSON value), e.g. --set cbad=0.2 --set "regrid=[20, 400, 8.0]"')
    p.add_argument('--init', default='', help='round 0 from the restart checkpoint of a <out>_pair.npz (and its '
                   'bracket; the npz may be written for the purpose, e.g. a perturbed pair)')
    p.add_argument('--sub', default='', metavar='S0,S1', help='with --init: round 0 on the part [S0, S1] of the '
                   'pair\'s line (bracket A_lo + S (A_hi - A_lo)), e.g. the next round of a run stopped at cbad')
    p.add_argument('--init-T', type=float, default=None, help='with --init: restart from the pair checkpoint '
                   'nearest this T instead of the stored one; states on another grid than this run\'s are '
                   'interpolated from the production grid of that time')
    p.add_argument('--init-grid', default='', metavar='N,NZ', help='with --init (T >= Tg): the pair states are on '
                   'the N x NZ grid of the regridded production configuration (default: CONFIG regrid)')
    p.add_argument('--track', default='', help='round json(s), comma separated: round 0 replays their frame and '
                   'substeps (with --init: the run the pair comes from)')
    p.add_argument('--axis', default='', metavar='ZMAX,STEP', help='also record E^Z_Z on the axis at every '
                   'STEP-th Z node below ZMAX (frame units) at every refresh, to <out>_round<k>_axis.npz')
    p.add_argument('--members', default='', help='explicit amplitudes (comma separated): one round, every '
                   'checkpoint of every member to <out>_members.npz, no bracket update')
    p.add_argument('--save-T', type=float, default=None, help='write every member\'s state at this T (after the '
                   'refresh there) to <out>_T<T>.npz, for --init-members')
    p.add_argument('--init-members', default='', help='round 0 from the member states of a --save-T npz '
                   '(their amplitudes, frame and outcome detector) and its bracket')
    p.add_argument('--recentre', default='', metavar='T,ZC,TCS[,BOX,AX,AZ,NZ]', help='at T move to the chart '
                   'centred on the axis point z = ZC (physical, as kkloc), on a full-Z grid with 2 nz + 1 nodes in '
                   'a box L - ZC/L(T) (rounded down to 1/4); the frame clock restarts at 0 on the new centre with '
                   'focus TCS.  With BOX: a box of physical half-size BOX (cut inside the strong field, so that '
                   'the mirror point stays outside), X / Z maps AX, AZ, 2 NZ + 1 Z nodes, alpha_M on the faces = '
                   'the lapse there, and the zoom rate the smallest one with outflow margin 2 mmin (gh_sim.outflow_rate)')
    p.add_argument('--out', required=True)
    a = p.parse_args()
    import warnings
    warnings.simplefilter('ignore')
    from vacuum import gh_sim as g
    from vacuum import gh_data as gdat
    from vacuum.gh_cuda import CudaGH
    cfg = dict(CONFIG)
    for kv in a.set:
        k, v = kv.split('=', 1)
        assert k in cfg, f'unknown CONFIG key {k}'
        cfg[k] = json.loads(v)
    if a.full:
        cfg.update(zsym=0, nz=2 * cfg['nz'])
    G, gh = g.make(cfg)
    E = CudaGH(G, gh)
    Tg, n2, ax2 = cfg['regrid']
    G2, gh2 = g.make(dict(cfg, n=n2, nz=G.shape[1], ax=ax2))
    E2 = CudaGH(G2, gh2)
    E3 = None
    if cfg['regrid2']:                  # a second, finer grid from T2 on (same X map)
        T2, n3, nz3 = cfg['regrid2']
        G3, gh3 = g.make(dict(cfg, n=n3, nz=nz3, ax=ax2, **(cfg['gauge2'] or {})))
        E3 = CudaGH(G3, gh3)
    Ts = cfg['tclock'][0]
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    print(f'grid {G.shape} (ax {cfg["ax"]}), from T = {Tg} {G2.shape} (ax {ax2}); eta {a.eta}; '
          f'tclock from T = {Ts} on tc* = {cfg["tclock"][1]}', flush=True)

    track = Track()
    for fn in a.track.split(',') if a.track else []:
        with open(fn) as f:
            track.add(json.load(f)['hist'], Ts)
    if cfg['hold'] is not None:
        track.ell = {k: v for k, v in track.ell.items() if k < cfg['hold'] - 1e-9}
    lo, hi = a.lo, a.hi
    pair = None                         # (Tc, u_lo, u_hi, peak, Tpk, frame) to restart from
    saved = None
    if a.init_members:
        saved = np.load(a.init_members)
        lo, hi = float(saved['lo']), float(saved['hi'])
        print(f'round 0 from the members of {a.init_members}: T = {float(saved["T"]):.2f}', flush=True)
    if a.init:
        z = np.load(a.init)
        i = int(z['ic']) if a.init_T is None else int(np.argmin(np.abs(z['T'] - a.init_T)))
        pair = (float(z['T'][i]), z['u_lo'][i], z['u_hi'][i], float(z['peak'][i]), float(z['Tpk'][i]),
                tuple(float(z['fr_' + k][i]) for k in ('ell', 'lnL', 'tc', 'c')))
        lo, hi = float(z['lo']), float(z['hi'])
        if a.sub:
            s0, s1 = (float(x) for x in a.sub.split(','))
            du = pair[2] - pair[1]
            pair = (pair[0], pair[1] + s0 * du, pair[1] + s1 * du) + pair[3:]
            lo, hi = lo + s0 * (hi - lo), lo + s1 * (hi - lo)
        print(f'round 0 from {a.init}: T = {pair[0]:.2f}, bracket [{lo:.15f}, {hi:.15f}]', flush=True)
    Tf = pair[0] if pair else 0.0       # phase of the LU refreeze schedule

    def sched(T):
        q = (T - (Tg if T >= Tg - 1e-9 else Tf)) / (cfg['relu'] * cfg['mrefresh'])
        return abs(q - round(q)) < 1e-6

    w = hi - lo
    B = a.B
    edge = None                         # the outcome of every member of the last round, if all agreed
    for rnd in range(a.rounds):
        t0 = time.time()
        if saved is not None:
            amps = saved['amps']
            s = (amps - lo) / (hi - lo)
        elif a.members:
            amps = np.array([float(x) for x in a.members.split(',')])
            s = (amps - lo) / (hi - lo)
        else:
            s = np.arange(1, B + 1) / (B + 1)
            amps = lo + (hi - lo) * s
        B = len(amps)
        if saved is not None:
            U = saved['u']
            T0, pk0, tp0, fr0 = float(saved['T']), saved['peak'], saved['Tpk'], tuple(float(x) for x in saved['fr'])
            saved = None                # later rounds restart from the pair as usual
        elif pair is None:
            U = np.array([g.active(G, gdat.brill_data(G, float(A), cfg['sigma'], ell=cfg['ell0'], gh=gh, nr=cfg['nr'],
                                                      nmu=cfg['nmu'], eta=a.eta, eps=a.eps)[0]) for A in amps])
            T0, pk0, tp0, fr0 = 0.0, 0.0, 0.0, (cfg['ell0'], 0.0, 0.0, float('nan'))
        else:
            T0, ulo, uhi, pk0, tp0, fr0 = pair
            U = ulo[None] + s[:, None] * (uhi - ulo)[None]
        Er, switch = (E2, []) if T0 >= Tg - 1e-9 else (E, [(Tg, E2, regridder(E, E2, cfg))])
        if E3 is not None:
            if T0 > T2 + 0.5 * cfg['dT']:      # (from T2 itself: through the switch, i.e. its gauge re-solve)
                Er = E3
            else:
                switch.append((T2, E3, regridder(E2, E3, cfg, cfg['gauge2'])))
        if U.shape[1] != g.NA * Er.G.N:
            # pair states of a production run: its grid at T0 (before or after its regrid, or --init-grid)
            if a.init_grid:
                n0, nz0 = (int(x) for x in a.init_grid.split(','))
                Gp = g.grid_from_config(dict(CONFIG, n=n0, nz=nz0, ax=CONFIG['regrid'][2]))
            else:
                Gp = g.grid_from_config(dict(cfg, n=cfg['regrid'][1], ax=cfg['regrid'][2])
                                        if T0 >= cfg['regrid'][0] - 1e-9 else cfg)
            assert U.shape[1] == g.NA * Gp.N, 'pair states on an unknown grid'
            U = np.array([g.transfer_fields(Gp, v.reshape(g.NA, *Gp.shape), Er.G, fr0[0]).ravel() for v in U])
            print(f'   members interpolated from {Gp.shape} to {Er.G.shape}', flush=True)
        axis = None
        if a.axis:
            zmax, step = a.axis.split(',')
            axis = lambda G, zmax=float(zmax), step=int(step): np.flatnonzero(np.abs(G.az.x) < zmax)[::step]
        rc = {}                         # the recentred engine, once made
        recentre = None
        if a.recentre:
            rcs = [float(x) for x in a.recentre.split(',')]
            Trc, zcp, tcs2 = rcs[:3]
            box = rcs[3:]                   # BOX, AX, AZ, NZ of a cut box

            def to_centre(Eo, So, L, zcp=zcp):
                t1 = time.time()
                zeta = zcp / L
                if box:
                    Lb = box[0] / L
                    G3, gh3 = g.make(dict(cfg, n=Eo.G.shape[0], nz=2 * int(box[3]) + 1, L=Lb, ax=box[1],
                                          az=box[2], zsym=0, lbnd='state'))
                else:
                    Lb = float(np.floor(4 * (Eo.G.L - abs(zeta))) / 4)
                    G3, gh3 = g.make(dict(cfg, n=Eo.G.shape[0], nz=2 * Eo.G.shape[1] + 1, L=Lb, ax=Eo.G.ax.a,
                                          az=Eo.G.az.a, zsym=0))
                E3 = rc['E'] = CudaGH(G3, gh3)
                E3.set_ell(Eo.ell)
                Uo = Eo.to_active(So).reshape(So.shape[0], g.NA, *Eo.G.shape)
                S3 = E3.from_active(np.array([g.recentre_fields(Eo.G, F, G3, Eo.ell, zeta) for F in Uo])
                                    .reshape(So.shape[0], -1))
                E3.freeze_lu(S3)
                S3 = E3.solve_PT(S3, E3.maximal(S3))
                lam3 = E3.maximal(S3)
                cm = float(E3.constraint(S3, lam3).abs().amax())
                print(f'   recentred: zeta {zeta:.4f}, v = ell zeta {Eo.ell * zeta:.4f}, box {Lb:.4f} '
                      f'(physical {Lb * L:.4f}), max|C| {cm:.2e} [{time.time() - t1:.0f}s]', flush=True)
                if not box:
                    return E3, S3, zcp
                # the slowest zoom that makes every outer face spacelike, with margin 2 mmin
                ell = g.outflow_rate(G3, g.full_state(G3, E3.to_active(S3[:1])[0], Eo.ell), 2 * cfg['tclock'][5])
                S1 = E3.reframe(S3[:1].clone(), lam3[:1], ell)
                mg = g.outflow_margin(G3, g.full_state(G3, E3.to_active(S1)[0], ell))
                E3.set_ell(Eo.ell)
                print(f'   cut box: zoom rate {ell:.4f} (outflow margin there {mg:.3f})', flush=True)
                return E3, S3, zcp, ell
            recentre = (Trc, to_centre, tcs2)
        save = None
        if a.save_T is not None:
            def save_fn(T, Eo, So, peak, Tpk, fr):
                np.savez(f'{a.out}_T{T:g}.npz', T=T, u=Eo.to_active(So), amps=amps, lo=lo, hi=hi, peak=peak,
                         Tpk=Tpk, fr=np.array(fr), shape=np.array(Eo.G.shape))
                print(f'  T = {T:.2f}: member states written to {a.out}_T{T:g}.npz', flush=True)
            save = (a.save_T, save_fn)
        Er.set_ell(fr0[0])
        S = Er.from_active(U)
        print(f'round {rnd}: from T = {T0:.2f}, grid {Er.G.shape}, ell {fr0[0]}, ln L {fr0[1]:.4f}, tc {fr0[2]:.5f}',
              flush=True)
        def dump(st, te, h, rnd=rnd, T0=T0):     # the round so far, undecided members 'active' up to now
            m = [(float(A), x or 'active', float(t if x else h['T'][-1])) for A, x, t in zip(amps, st, te)]
            with open(f'{a.out}_round{rnd}.json', 'w') as f:
                json.dump(dict(args=vars(a), template=cfg, lo=lo, hi=hi, T0=T0, members=m,
                               hist={k: v for k, v in h.items() if not k.startswith('eax')}), f)

        status, tend, hist, ck, frames = evolve_round(Er, S, T0, fr0, pk0, tp0, cfg, a.Tend, track, sched, switch,
                                                      axis=axis, dump=dump, recentre=recentre, save=save)
        if axis is not None:
            ax = {}
            if hist.get('eax'):         # on the last grid only (a regrid2 in Z changes the axis nodes)
                ea, Ta = hist.pop('eax'), hist.pop('eax_T')
                keep = [j for j, e in enumerate(ea) if np.shape(e) == np.shape(ea[-1])]
                Ga = E3.G if E3 is not None and len(axis(E3.G)) == np.shape(ea[-1])[-1] else E2.G
                ax.update(eax=np.array([ea[j] for j in keep]), zax=Ga.az.x[axis(Ga)], T=np.array([Ta[j] for j in keep]))
            if hist.get('eax_rc'):
                ax.update(eax_rc=np.array(hist.pop('eax_rc')), zax_rc=rc['E'].G.az.x[axis(rc['E'].G)],
                          T_rc=np.array(hist.pop('eax_rc_T')))
            np.savez(f'{a.out}_round{rnd}_axis.npz', **ax)
        if hist.get('eaxl_T'):          # --set lines: the axis (X node 0) and equator (Z node ic) records
            ln = {k[5:]: hist.pop(k) for k in [k for k in hist if k.startswith('eaxl_')]}
            np.savez(f'{a.out}_round{rnd}_lines.npz', Z=Er.G.az.x, amps=amps, tend=tend,
                     **{k: np.array(v) for k, v in ln.items()})
        track.add(hist, Ts)             # the next round replays this one
        out = [(float(A), st, float(t)) for A, st, t in zip(amps, status, tend)]
        print(f'round {rnd} [{time.time() - t0:.0f}s]:', flush=True)
        for (A, st, t), sk in zip(out, s):
            print(f'   s = {sk:.4f}  A = {A:.15f}: {st} at T = {t:.2f}', flush=True)
        rec = dict(args=vars(a), template=cfg, lo=lo, hi=hi, T0=T0, members=out, hist=hist)
        if a.recentre:
            rec['recentre'] = dict(T=Trc, zc=zcp, tcs=tcs2, grid=list(rc['E'].G.shape), box=rc['E'].G.L) if rc else None
        if a.recentre:
            with open(f'{a.out}_round{rnd}.json', 'w') as f:
                json.dump(rec, f)
            break
        if a.members:
            with open(f'{a.out}_round{rnd}.json', 'w') as f:
                json.dump(rec, f)
            ck = {b: [c for c in ck[b] if c[1].size == ck[b][-1][1].size] if ck[b] else [] for b in range(B)}   # the last grid's
            np.savez(f'{a.out}_members.npz', amps=amps, status=np.array(status), tend=tend,
                     **{f'T{b}': np.array([c[0] for c in ck[b]]) for b in range(B)},
                     **{f'u{b}': np.array([c[1] for c in ck[b]]) for b in range(B)},
                     frT=np.array(sorted(frames)),
                     **{f'fr_{k}': np.array([frames[T][j] for T in sorted(frames)])
                        for j, k in enumerate(('ell', 'lnL', 'tc', 'c'))})
            break
        # new bracket: the pair of decided members (endpoints included) where the outcome flips;
        # undecided or failed members between or beside them are passed over
        allA = [lo] + list(amps) + [hi]
        alls = ['collapse'] + status + ['disperse']
        dec = [i for i in range(B + 2) if alls[i] in ('collapse', 'disperse')]
        flips = [(i, k) for i, k in zip(dec, dec[1:]) if alls[i] != alls[k]]
        sd = {alls[i] for i in dec[1:-1]}
        ok = len(flips) == 1 and alls[flips[0][0]] == 'collapse' and bool(sd)
        if ok and len(dec) < B + 2:
            print('   passed over: ' + ', '.join(f'{b} {status[b]}' for b in range(B) if b + 1 not in dec), flush=True)
        if ok and len(sd) == 1:
            # every member on one side: the threshold is in the end segment next to the endpoint of the
            # other outcome (a chance of 2 / (B + 1) per round), or that endpoint's outcome is not
            # reproduced.  One such round is followed on the end segment; a second in a row at the same
            # end is taken as the latter
            if edge == status[0]:
                print(f'   every member {min(sd)} again: the bracket endpoint is not confirmed', flush=True)
                ok = False
            else:
                print(f'   every member {min(sd)}: the threshold is next to the bracket endpoint', flush=True)
            edge = min(sd)
        else:
            edge = None
        if ok:
            # reliability: the pair's max|C_a| up to 1 before the dispersing member's decision and
            # 2.5 before the collapsing one's (a collapse itself drives C up over its last ~2)
            i, k = flips[0]
            Th, Ch = np.array(hist['T']), np.array(hist['C'])
            cut = {b: tend[b] - (2.5 if status[b] == 'collapse' else 1.0) for b in (i - 1, k - 1) if 0 <= b < B}
            cpair = max(float(np.max(Ch[Th <= t, b], initial=0.0)) for b, t in cut.items())
            print('   pair max|C| before ' + ', '.join(f'T = {t:.1f}' for t in cut.values()) + f': {cpair:.2e}',
                  flush=True)
            if cpair > cfg['cbad']:
                print(f'   above cbad = {cfg["cbad"]}: outcomes no longer trustworthy; stopping', flush=True)
                ok = False
        if ok:
            i, k = flips[0]
            lo, hi = allA[i], allA[k]
            w = w * (k - i) / (B + 1)
            if i >= 1 and k <= B:
                cl, ch = {c[0]: c for c in ck[i - 1]}, {c[0]: c for c in ck[k - 1]}
                Tc = sorted(set(cl) & set(ch))
                rel = [np.linalg.norm(ch[T][1] - cl[T][1]) / np.linalg.norm(cl[T][1]) for T in Tc]
                good = [j for j, r in enumerate(rel) if r < cfg['dtol'] and sched(Tc[j])]
                ic = good[-1] if good else 0
                print('   pair separation: ' + ' '.join(f'{T:.1f}:{r:.1e}' for T, r in zip(Tc, rel)
                                                      if abs(T - round(T)) < 1e-6), flush=True)
                if Tc:                  # regrid2: keep the checkpoints on the restart's grid
                    keep = [j for j, T in enumerate(Tc) if cl[T][1].size == cl[Tc[ic]][1].size]
                    ic = keep.index(ic)
                    Tc, rel = [Tc[j] for j in keep], [rel[j] for j in keep]
                # no common checkpoint after T = 0: the next round from the Brill data of the family, unless
                # round 0 came from --init / --init-members (data that brill_data does not reproduce)
                pair = None
                if Tc:
                    np.savez(f'{a.out}_pair.npz', T=np.array(Tc), rel=np.array(rel), ic=ic, lo=lo, hi=hi,
                             round=rnd, u_lo=np.array([cl[T][1] for T in Tc]), u_hi=np.array([ch[T][1] for T in Tc]),
                             peak=np.array([max(cl[T][2], ch[T][2]) for T in Tc]),
                             Tpk=np.array([max(cl[T][3], ch[T][3]) for T in Tc]),
                             **{f'fr_{k}': np.array([frames[round(T, 6)][j] for T in Tc])
                                for j, k in enumerate(('ell', 'lnL', 'tc', 'c'))})
                    Tr = Tc[ic]
                    if Tr > 1e-9 or a.init or a.init_members:
                        pair = (Tr, cl[Tr][1], ch[Tr][1], max(cl[Tr][2], ch[Tr][2]), max(cl[Tr][3], ch[Tr][3]),
                                frames[round(Tr, 6)])
                        print(f'   restart at T = {Tr:.2f} (separation {rel[ic]:.2e})', flush=True)
            elif pair is not None:
                # flip next to an endpoint of the bracket: the sub-segment of the same line at T0
                print('   flip next to an endpoint of the bracket: restart at T0 on the sub-segment', flush=True)
                allS = [0.0] + list(s) + [1.0]
                dd = pair[2] - pair[1]
                pair = (T0, pair[1] + allS[i] * dd, pair[1] + allS[k] * dd, pk0, tp0, fr0)
        rec['w'] = w
        with open(f'{a.out}_round{rnd}.json', 'w') as f:
            json.dump(rec, f)
        if not ok:
            print(f'   outcomes not a single flip ({flips}); stopping', flush=True)
            break
        print(f'   new bracket [{lo:.15f}, {hi:.15f}] (width {w:.3e}, ln w = {np.log(w):.3f})', flush=True)


if __name__ == '__main__':
    main()
