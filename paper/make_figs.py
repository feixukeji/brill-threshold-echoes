#!/usr/bin/env python3
"""Figures of the paper, from the archived runs under out/gh/ (no evolution).

    python paper/make_figs.py

centred.pdf    central E^Z_Z of near-threshold members; departure times
axis.pdf       E^Z_Z on the axis around the end of the centred phase
events.pdf     the invariant E^Z_Z on the axis after the centred phase, in two slicings; position of max K_ij K^ij
imax.pdf       largest Kretschmann scalar of dispersing members against |A - A*|, with Fig. 1 of Baumgarte et al. 2023;
               central proper time and place of the maximum
null.pdf       rescaled Kretschmann scalar along outgoing null rays on the axis and in the equatorial plane,
               against T_null and lambda (the coordinates of Baumgarte et al. 2023, Fig. 2)
codim.pdf      the knob points and the other families: negative central lobes against rescaled time, one row each;
               lobe heights against the linear response
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm, SymLogNorm

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
from vacuum.gh_analysis import central, departures, imax_rows, load_round  # noqa: E402
from vacuum.gh_codim import centred_end, pair_curves, pair_members, scale_factor  # noqa: E402

OUT = os.path.join(HERE, 'figs')
GH = lambda p: os.path.join(ROOT, 'out', 'gh', p)

INK, INK2, GRID = '#0b0b0b', '#52514e', '#d9d8d4'
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
SEQ = ['#86b6ef', '#3987e5', '#1c5cab', '#0d366b']           # light -> dark: farther -> closer to A*
DIV = LinearSegmentedColormap.from_list('div', ['#104281', '#3987e5', '#cde2fb', '#f0efec',
                                               '#f6cccb', '#e34948', '#8f1d1c'])
LOBES = (1.7011, 1.8949, 1.9828, 2.0235)                     # main negative lobes of the threshold trajectory
W1 = 3.4

plt.rcParams.update({
    'font.size': 9, 'font.family': 'serif', 'mathtext.fontset': 'cm', 'axes.labelsize': 9,
    'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5, 'legend.fontsize': 8.5, 'axes.linewidth': 0.6,
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'xtick.minor.width': 0.4, 'ytick.minor.width': 0.4,
    'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.top': True, 'ytick.right': True,
    'lines.linewidth': 1.0, 'axes.edgecolor': INK, 'axes.labelcolor': INK, 'xtick.color': INK,
    'ytick.color': INK, 'legend.frameon': False, 'savefig.dpi': 600, 'pdf.fonttype': 42,
})


def label(ax, s):
    ax.text(0.02, 0.95, s, transform=ax.transAxes, ha='left', va='top', fontsize=9)


def fig_centred():
    r0, r1, rw = GH('bis_gpu/t24d_round0.json'), GH('bis_gpu/t24d_round1.json'), GH('thr/wide_round0.json')
    Astar, ip, tpair, rows, tg, eref = departures(r1, [r0])
    wide = [r for r in departures(r0, [rw])[3] if r[5] == rw]
    # the other gauge and grid: distances from their own threshold, departures from their own threshold trajectory
    alt = {k: departures(GH(f'thr/{k}_round0.json'), [])[3] for k in ('g1', 'r500')}
    d0, d1, dw = load_round(r0), load_round(r1), load_round(rw)
    tstar = 2.06
    fig = plt.figure(figsize=(W1, 5.0))
    outer = fig.add_gridspec(2, 1, height_ratios=(1.25, 1), hspace=0.32)
    top = outer[0].subgridspec(4, 1, hspace=0.08)
    bot = outer[1].subgridspec(1, 2, wspace=0.06)
    # threshold trajectory: round 0's dispersing flip member before tc = 1.7, round 1's after
    t0_, e0_, _ = central(d0, 5)
    t1_, e1_, _ = central(d1, 6)
    tr = np.r_[t0_[t0_ < 1.7], t1_[t1_ >= 1.7]]
    er = np.r_[e0_[t0_ < 1.7], e1_[t1_ >= 1.7]]
    sc = lambda t, e: (-np.log(tstar - t), e * (tstar - t) ** 2)
    k = (tr > 1.2) & (tr < tstar - 0.022)
    xr, yr = sc(tr[k], er[k])
    for n, (d, m) in enumerate([(dw, 13), (dw, 11), (dw, 9), (d0, 13)]):
        a = fig.add_subplot(top[n])
        tc, e, _ = central(d, m)
        k = (tc > 1.2) & (tc < tstar - 0.022)
        a.plot(xr, yr, color='#a8a7a3', lw=1.6, solid_capstyle='butt', label='threshold trajectory')
        a.plot(*sc(tc[k], e[k]), color=BLUE, lw=0.9, ls=(0, (3.5, 1.5)), label='member')
        for t in LOBES[:3]:
            a.axvline(-np.log(tstar - t), color=GRID, lw=0.6, zorder=0)
        a.set_xlim(-np.log(tstar - 1.2), -np.log(0.022))
        a.set_ylim(-140, 110)
        a.set_yticks([-100, 0])
        dA = abs(d['members'][m][0] - Astar)
        a.text(0.985, 0.9, rf'$|A-A_\star|=10^{{{int(round(np.log10(dA)))}}}$', transform=a.transAxes,
               fontsize=8, ha='right', va='top')
        if n == 0:
            a.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2, handlelength=2.4)
            a.text(-0.2, 1.0, '(a)', transform=a.transAxes, ha='left', va='bottom', fontsize=9)
        if n < 3:
            plt.setp(a.get_xticklabels(), visible=False)
        else:
            a.set_xlabel(r'$-\ln(\tau_\star-\tau_c)$')
        if n == 1:
            a.set_ylabel(r'$(\tau_\star-\tau_c)^2\,e_c$', y=0.0)
    R = np.array([(r[0], r[1], r[4]) for r in rows + wide if np.isfinite(r[4])])
    Q = {key: np.array([(r[0], r[1], r[4]) for r in v if np.isfinite(r[4])]) for key, v in alt.items()}
    axs = []
    for col, side in enumerate((-1, 1)):
        b = fig.add_subplot(bot[col], sharey=axs[0] if axs else None)
        axs.append(b)
        k = R[:, 1] == side
        b.plot(R[k, 0], R[k, 2], 'o', ms=2.4, color=INK, mew=0, zorder=4, label='bisection')
        for (key, lab), mk, ms, c in zip((('g1', r'damping $\times2$'), ('r500', r'$500\times1200$')), ('o', 's'),
                                       (4.4, 6.4), (BLUE, ORANGE)):
            q = Q[key][Q[key][:, 1] == side]
            b.plot(q[:, 0], q[:, 2], mk, ms=ms, mfc='none', mec=c, mew=0.6, zorder=3, label=lab)
        b.plot([3.46e-9], [tpair], '*', ms=6, color=INK, zorder=5, label='flip pair')
        for n, t in enumerate(LOBES):
            b.axhline(t, color=GRID, lw=0.6, zorder=0)
            if col == 1 and n < 3:
                b.text(1.02, t, str(n + 1), transform=b.get_yaxis_transform(), ha='left', va='center', fontsize=7.5,
                       color=INK2)
        b.set_xscale('log')
        b.set_xlim(2e-9, 2e-3) if side == 1 else b.set_xlim(2e-3, 2e-9)
        b.set_xticks([1e-8, 1e-6, 1e-4])
        b.set_ylim(1.3, 2.16)
        b.set_xlabel(r'$|A-A_\star|$, ' + ('collapse' if side == -1 else 'dispersal'))
        if col == 0:
            b.set_ylabel(r'departure $\tau_c$')
            b.text(0.04, 0.96, '(b)', transform=b.transAxes, ha='left', va='top', fontsize=9)
            b.legend(loc='lower right', handletextpad=0.2, borderaxespad=0.3, fontsize=7.5)
        else:
            plt.setp(b.get_yticklabels(), visible=False)
    fig.savefig(os.path.join(OUT, 'centred.pdf'), bbox_inches='tight')
    plt.close(fig)


def fig_axis():
    rj = GH('resp/t24d_T8ax_round0.json')
    d = load_round(rj)
    z = np.load(rj.replace('_round0.json', '_states.npz'))
    labs = list(z['labels'])
    b = labs.index('base')
    h = d['hist']
    T, L = np.array(h['T']), np.exp(np.array(h['lnL']))
    a0 = np.array(h['a0'])[:, b]
    tc = h['tc'][0] + np.concatenate([[0.0], np.cumsum(0.5 * (L[1:] * a0[1:] + L[:-1] * a0[:-1]) * np.diff(T))])
    E = np.array(z['eax'])[:, b] / L[:len(z['eax']), None] ** 2
    tc, L = tc[:len(E)], L[:len(E)]
    zp = np.array(z['zax'])[None, :] * L[:, None]               # physical z
    k = (tc > 1.86) & (tc < 2.075)
    fig, ax = plt.subplots(figsize=(W1, 2.5))
    pc = ax.pcolormesh(np.broadcast_to(tc[k, None], zp[k].shape), zp[k], E[k], cmap=DIV, shading='gouraud',
                       norm=SymLogNorm(linthresh=1e3, linscale=0.5, vmin=-3e5, vmax=3e5, base=10), rasterized=True)
    ax.contour(np.broadcast_to(tc[k, None], zp[k].shape), zp[k], E[k], levels=[-1e4, 1e4],
               colors=INK, linewidths=0.6, linestyles=['--', '-'])
    ax.set_ylim(0, 0.05)
    ax.set_xlabel(r'central proper time $\tau_c$')
    ax.set_ylabel(r'$z$ on the axis')
    cb = fig.colorbar(pc, ax=ax, pad=0.02, ticks=[-1e5, -1e4, -1e3, 0, 1e3, 1e4, 1e5])
    cb.ax.set_yticklabels([r'$-10^5$', r'$-10^4$', r'$-10^3$', '0', r'$10^3$', r'$10^4$', r'$10^5$'])
    cb.set_label(r'$E^z{}_z$', labelpad=1)
    cb.ax.tick_params(labelsize=8.5)
    cb.ax.minorticks_off()
    for t in LOBES[1:]:
        ax.axvline(t, color=INK2, lw=0.5, ls=':')
    fig.savefig(os.path.join(OUT, 'axis.pdf'), bbox_inches='tight')
    plt.close(fig)


def axis_peak(stem, m, Tmax=np.inf):
    """(tc, max over the axis of |E^Z_Z|, the centre included) of member m of a
    bis_res round written with --axis, up to its decision (and T < Tmax)."""
    d = load_round(GH(f'bis_res/{stem}.json'))
    z = np.load(GH(f'bis_res/{stem}_axis.npz'))
    T, L = np.array(d['hist']['T']), np.exp(np.array(d['hist']['lnL']))
    tc, e, Tm = central(d, m)
    n = len(z['T'])
    E = np.abs(np.array(z['eax'])[:, m]) / L[:n, None] ** 2
    k = np.isin(T[:n], Tm) & (T[:n] < Tmax)
    j = np.isin(Tm, T[:n][k])
    return tc[j], np.maximum(np.abs(e[j]), np.nanmax(E[k], axis=1)), d


def fig_events():
    fig, (a, b) = plt.subplots(2, 1, figsize=(W1, 3.6), sharex=True, gridspec_kw=dict(hspace=0.60))
    for m, c, ls, on_marker, off_marker, lab in (
            (5, ORANGE, '--', 'v', '+', 'collapses'), (6, BLUE, '-', 'o', 'x', 'disperses')):
        tc, I, d = axis_peak('dhold_round0', m)
        a.plot(tc, I, color=c, ls=ls, label=lab)
        T, L = np.array(d['hist']['T']), np.exp(np.array(d['hist']['lnL']))
        tcm, _, Tm = central(d, m)
        x, zz = np.array(d['hist']['kkloc'])[np.isin(T, Tm), m].T
        on = x < 2e-4
        for mask, position, marker, coord in ((on, zz, on_marker, 'z'), (~on, x, off_marker, r'\rho')):
            # Keep every sample visible; stagger the larger identifying symbols.
            b.scatter(tcm[mask], position[mask], s=1.6, marker='.', linewidths=0, color=c, zorder=2)
            b.plot(tcm[mask], position[mask], marker, ms=3.0, mfc='none', mew=0.6, color=c,
                   markevery=(0.0225 * (m - 5), 0.045), zorder=3, label=rf'${coord}$, {lab}')
    tc, I, _ = axis_peak('r400_round0', 6, Tmax=63.0)
    a.plot(tc, I, color=INK, lw=0.7, ls=':', label='other slicing')
    a.set_yscale('log')
    a.set_ylim(6e2, 6e6)
    a.set_ylabel(r'max $|E^z{}_z|$ on the axis')
    a.legend(loc='lower center', ncol=3, handlelength=1.5, columnspacing=0.4, handletextpad=0.2)
    label(a, '(a)')
    b.set_ylim(0, 0.035)
    b.set_xlim(2.045, 2.165)
    b.set_xlabel(r'central proper time $\tau_c$')
    b.set_ylabel(r'position of max $K_{ij}K^{ij}$')
    b.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2,
             handlelength=1.0, columnspacing=0.7, handletextpad=0.2)
    label(b, '(b)')
    fig.savefig(os.path.join(OUT, 'events.pdf'), bbox_inches='tight')
    plt.close(fig)


def fig_imax():
    rows = np.array([r[:1] + r[2:] for r in imax_rows([GH('thr/imaxf_round0.json'), GH('thr/imaxn_round0.json')])])
    dA, I, tc, rho, z = rows[:, 0], rows[:, 1], rows[:, 3], rows[:, 4], rows[:, 5]
    ref = np.loadtxt(GH('thr/bbc23_fig1_Aneg.txt'))
    where = np.where(z > 5 * rho, 1, np.where(rho > 5 * z, 2, 0))         # centre, axis, equatorial plane
    fig, (a, b) = plt.subplots(2, 1, figsize=(W1, 3.5), sharex=True,
                               gridspec_kw=dict(height_ratios=(1.7, 1), hspace=0.08))
    a.fill_between(ref[:, 0], ref[:, 2], ref[:, 3], color='#cfcdc8', lw=0)
    a.plot(ref[:, 0], ref[:, 1], color='#6f6e69', lw=1.0, ls=(0, (3, 1.5)))
    a.fill_between([], [], [], color='#cfcdc8', lw=0, label='three codes')
    a.plot(dA, I, color=INK, lw=0.5, zorder=3)
    for y, txt in ((1.24e8, 'last centered lobe'), (2.82e11, 'first off-center event')):     # threshold trajectory
        a.plot([2e-9, 3e-5], [y, y], color=INK2, lw=0.7, ls='-.', zorder=2)
        a.text(3.5e-5, 1.4 * y, txt, fontsize=7, color=INK2, va='bottom', ha='left')
    b.plot(dA, tc, color=INK, lw=0.5, zorder=3)
    for w, mk, lab in ((0, 'o', 'center'), (1, '^', 'axis'), (2, 's', 'equatorial plane')):
        k = where == w
        for ax, y in ((a, I), (b, tc)):
            ax.plot(dA[k], y[k], mk, ms=3.6, mfc='white' if w else INK, mec=INK if w != 1 else BLUE, mew=0.7,
                    zorder=4, label=lab if ax is a else None)
    a.set_xscale('log')
    a.set_yscale('log')
    a.set_xlim(2e-9, 2)
    a.set_ylim(5e2, 3e12)
    a.set_ylabel(r'$I_{\max}\,\sigma^4$')
    a.legend(loc='lower left', handletextpad=0.3, borderaxespad=0.3)
    a.text(0.97, 0.95, '(a)', transform=a.transAxes, ha='right', va='top', fontsize=9)
    b.set_ylim(0.75, 2.25)
    b.set_xlabel(r'$A-A_\star$')
    b.set_ylabel(r'$\tau_c$ at $I_{\max}$')
    b.text(0.03, 0.07, '(b)', transform=b.transAxes, ha='left', va='bottom', fontsize=9)
    fig.savefig(os.path.join(OUT, 'imax.pdf'), bbox_inches='tight')
    plt.close(fig)


def null_map(f):
    """(T_null, lambda grid, (tau* - tau_e) |I|^(1/4)) of a gh_null.py output."""
    z = np.load(f)
    ts, sg, te = float(z['tstar']), float(z['sigma']), z['tau_e']
    n = len(te)
    lg = np.arange(0.0, max(z[f'lam{i}'][-1] for i in range(n)) + 1e-9, 0.005)
    M = np.full((n, len(lg)), np.nan)
    for i in range(n):
        v = (ts - te[i]) * np.abs(z[f'I{i}']) ** 0.25
        M[i, :len(v)] = v[:len(lg)]
    return -np.log((ts - te) / sg), lg, M, ts, sg


def fig_null(tag=''):
    """(a, b) maps for the dispersing member of the flip pair; (c, d) cuts at lambda = 1 for members at
    1e-5, 1e-6 and the flip pair."""
    fig = plt.figure(figsize=(W1, 5.6))
    gs = fig.add_gridspec(4, 2, width_ratios=(1, 0.035), height_ratios=(1, 1, 0.8, 0.8), hspace=0.1, wspace=0.03)
    cm = LinearSegmentedColormap.from_list('seq', ['#f7f7f5', '#cde2fb', '#86b6ef', '#3987e5', '#1c5cab', '#0d366b',
                                                   '#051a36'])
    axs = [fig.add_subplot(gs[r, 0]) for r in range(4)]
    for a in axs[1:]:
        a.sharex(axs[0])
    for n, (w, lab) in enumerate((('axis', 'axis'), ('equator', 'equatorial plane'))):
        a = axs[n]
        Tn, lg, M, ts, sg = null_map(GH(f'null/m1_{w}{tag}.npz'))
        pc = a.pcolormesh(Tn, lg, M.T, shading='nearest', cmap=cm, norm=LogNorm(vmin=0.1, vmax=30), rasterized=True)
        a.set_ylim(0, 2.5)
        a.set_yticks([0, 1, 2])
        a.set_ylabel(r'$\lambda$')
        a.text(0.98, 0.95, f'({"ab"[n]}) {lab}', transform=a.transAxes, ha='right', va='top', fontsize=8.5,
               bbox=dict(fc='white', ec='none', pad=1.0))
    cb = fig.colorbar(pc, cax=fig.add_subplot(gs[0:2, 1]))
    cb.set_label(r'$(\tau_\star-\tau_c)\,|I|^{1/4}$', labelpad=1)
    cb.ax.tick_params(labelsize=8.5)
    styles = ((3, 'C', '#86b6ef', (0, (1, 1.2)), r'$10^{-5}$'), (2, 'C', '#3987e5', (0, (4, 1.5)), r'$10^{-6}$'),
              (0, 'C', ORANGE, '-', r'$-3.5\times10^{-9}$'), (1, 'C', INK, '-', r'$3.5\times10^{-9}$'))
    for n, (w, lab) in enumerate((('axis', 'axis'), ('equator', 'equatorial plane'))):
        a = axs[2 + n]
        for m, _, c, ls, lab_m in styles:
            Tn, lg, M, ts, sg = null_map(GH(f'null/m{m}_{w}{tag}.npz'))
            a.plot(Tn, M[:, np.searchsorted(lg, 1.0)], color=c, ls=ls, lw=1.3 if m == 0 else 0.8,
                   label=rf'$A-A_\star={lab_m[1:-1]}$' if n == 0 else None)
        a.set_yscale('log')
        a.set_ylim(1e-3, 80)
        a.set_yticks([1e-2, 1, 1e2])
        a.set_ylabel(r'$(\tau_\star-\tau_c)\,|I|^{1/4}$', fontsize=8)
        a.text(0.98, 0.95, f'({"cd"[n]}) {lab}', transform=a.transAxes, ha='right', va='top', fontsize=8.5,
               bbox=dict(fc='white', ec='none', pad=1.0))
    axs[2].legend(loc='lower left', fontsize=7, handlelength=2.2, borderaxespad=0.2, ncol=2, columnspacing=0.6,
                  labelspacing=0.2)
    for a in axs:
        for t in LOBES:
            a.axvline(-np.log((ts - t) / sg), color=INK2, lw=0.5, ls=':', zorder=0)
        if a is not axs[3]:
            plt.setp(a.get_xticklabels(), visible=False)
    axs[3].set_xlim(Tn.min() if not tag else -1.6, 6.0)
    axs[3].set_xlabel(r'$T_{\rm null}=-\ln[(\tau_\star-\tau_c)/\sigma]$')
    fig.savefig(os.path.join(OUT, f'null{tag}.pdf'), bbox_inches='tight')
    plt.close(fig)

# the knob points of Sec. V, (p1, p2) -> the flip pair (dispersing member first) of their deepest round
# ('a:i@b:j' when every member of the last round went one way); (-2, 0) is not one of them (Sec. V)
KNOBS = {(0.0, 0.0): 'bis_gpu/t24d_round1.json:6,5',
         (0.4, 0.0): 'codim/qf+4_0b_round2.json:3,2', (-0.4, 0.0): 'codim/qf-4_0b_round0.json:4@codim/qf-4_0b_round1.json:15',
         (0.0, 0.4): 'codim/qf0_+4b_round1.json:6@codim/qf0_+4b_round2.json:15', (0.0, -0.4): 'codim/qf0_-4b_round3.json:4,3',
         (1.0, 1.0): 'codim/qf+10_+10_round4.json:2,1', (1.0, -1.0): 'codim/qf+10_-10_round4.json:3,2',
         (-1.0, 1.0): 'codim/qf-10_+10_round3.json:11,10', (-1.0, -1.0): 'codim/qf-10_-10c_round0.json:4,3',
         (2.0, 0.0): 'codim/qf+20_0_round4.json:9,8', (0.0, 2.0): 'codim/qf0_+20_round4.json:8,7',
         (0.0, -2.0): 'codim/qf0_-20_round3.json:5,4'}


def knob_slopes():
    """d ln|e_c| / dp_j at the lobes 1.8949 and 1.9828 of the base (the last
    centred lobe and the echo before it), from the linear response of the
    tuned-free Brill batches (the knobs' units)."""
    from scipy.interpolate import CubicSpline
    from vacuum.gh_analysis import response_members
    z = np.load(GH('codim/qfknobs.npz'))
    names, tb, e0, R, _, _ = response_members([GH('resp/t24d_T0qfaf_round0.json'), GH('resp/t24d_T0qfbf_round0.json')])
    phys = [i for i, n in enumerate(names) if n != 'dT']
    s0 = CubicSpline(tb, e0)
    tn = []
    for t in LOBES[1:3]:
        g = np.linspace(t - 0.004, t + 0.004, 2001)
        tn.append(g[np.argmin(s0(g))])
    return {j: CubicSpline(tb, R[:, phys] @ z['vt'][j] / z['amp'][j, -1])(tn) / s0(tn) for j in (1, 2)}


FAMILIES = ['bis_gpu/e090t_round0.json:9,8', 'bis_gpu/e095r_round0.json:4,3', 'bis_gpu/e105r_round0.json:9,8',
            'bis_gpu/e110t_round0.json:10,9', 'bis_gpu/e120t_round2.json:5,4']


def lobe_sequence(spec):
    """Echo lobes (t, e) up to the one where the growth of the central
    curvature stops, the later negative lobes before the flip pair
    separates, and the separation time, of the first member of a pair spec."""
    from vacuum.gh_analysis import extrema
    cur, tsep = pair_curves(pair_members('@'.join(GH(x) for x in spec.split('@')), quiet=True))
    tc, e, _ = cur[0]
    end, chain, _ = centred_end(tc, e, t1=tsep)
    ex = extrema(tc, e)
    later = ex[(ex[:, 1] < 0) & (ex[:, 0] > end[0]) & (ex[:, 0] <= tsep) & (np.abs(ex[:, 1]) > 1e3)]
    return np.array(chain + [end]), later, tsep


XMAX = 2.2


def fig_codim():
    seq = {p: lobe_sequence(spec) for p, spec in KNOBS.items()}
    fam = [lobe_sequence(spec) for spec in FAMILIES]
    ref = seq[(0.0, 0.0)][0]
    slope = knob_slopes()
    fig = plt.figure(figsize=(W1, 5.6))
    outer = fig.add_gridspec(2, 1, height_ratios=(2.2, 1), hspace=0.3)
    a = fig.add_subplot(outer[0])
    bot = outer[1].subgridspec(1, 2, wspace=0.08)
    rows = [((0.0, 0.0), seq[(0.0, 0.0)])] + [(p, v) for p, v in seq.items() if p != (0.0, 0.0)] \
        + list(zip((0.9, 0.95, 1.05, 1.1, 1.2), fam))
    cmap = LinearSegmentedColormap.from_list('seq', ['#e3eefc', '#86b6ef', '#3987e5', '#1c5cab', '#0b2447'])
    norm = LogNorm(3e2, 2e4)
    fmt = lambda x: '0' if x == 0 else (f'{x:+.1f}' if abs(x) < 1 else f'{x:+.0f}')
    labels = []
    for r, (key, (lob, later, tsep)) in enumerate(rows):
        lam = scale_factor(lob, ref)
        y = -r
        a.plot([lob[0, 0] / lam, min(tsep / lam, XMAX)], [y, y], color=GRID, lw=0.8, zorder=1)
        if tsep / lam < XMAX:
            a.plot([tsep / lam] * 2, [y - 0.3, y + 0.3], color=INK2, lw=0.7, zorder=1)
        else:                                       # separation beyond the plotted range
            a.plot([XMAX - 0.006], [y], marker='>', ms=3.5, color=INK2, zorder=2, clip_on=False)
            a.text(XMAX - 0.012, y + 0.12, f'{tsep / lam:.2f}', fontsize=6.5, color=INK2, ha='right', va='bottom')
        a.scatter(lob[:, 0] / lam, [y] * len(lob), c=np.abs(lob[:, 1]) * lam ** 2, cmap=cmap, norm=norm, s=16,
                  marker='o', linewidths=0, zorder=3)
        a.scatter(lob[-1:, 0] / lam, [y], s=34, facecolors='none', edgecolors=INK, linewidths=0.7, zorder=4)
        if len(later):
            a.scatter(later[:, 0] / lam, [y] * len(later), c=np.abs(later[:, 1]) * lam ** 2, cmap=cmap, norm=norm,
                      s=12, marker='s', linewidths=0, zorder=3)
        labels.append(f'$({fmt(key[0])},{fmt(key[1])})$' if isinstance(key, tuple) else rf'$\eta={key}$')
    for yb in (-0.5, -4.5, -11.5):
        a.axhline(yb, color=INK2, lw=0.4, zorder=0)
    for t in LOBES[:3]:
        a.axvline(t, color=GRID, lw=0.6, zorder=0, ls='--')
    a.set_yticks(-np.arange(len(rows)))
    a.set_yticklabels(labels, fontsize=7.5)
    a.tick_params(axis='y', length=0, right=False)
    a.set_ylim(-len(rows) + 0.4, 0.6)
    a.set_xlim(1.6, XMAX)
    a.set_xlabel(r'rescaled central proper time $\tau_c/\kappa$')
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cb = fig.colorbar(sm, ax=a, orientation='horizontal', location='top', fraction=0.05, pad=0.02, aspect=40)
    cb.set_label(r'$\kappa^2|e_c|$ at the negative lobes', fontsize=8.5)
    cb.ax.tick_params(labelsize=7.5)
    a.text(-0.02, 1.0, '(a)', transform=a.transAxes, ha='right', va='bottom', fontsize=9)
    e0 = {n: ref[-2 + n][1] for n in (0, 1)}
    b0 = None
    for col, name in enumerate(('echo before\nthe last', 'last centered\nlobe')):
        b = fig.add_subplot(bot[col], sharey=b0)
        b0 = b0 or b
        pg = np.array([-2.2, 2.2])
        for jj, c, mk, ls in ((1, BLUE, 'o', ':'), (2, ORANGE, 's', '-.')):
            b.plot(pg, slope[jj][col] * pg, color=c, lw=0.7, ls=ls)
            P = np.array(sorted((p[jj - 1], np.log(v[0][-2 + col][1] / e0[col])) for p, v in seq.items()
                                if p[2 - jj] == 0))
            b.plot(P[:, 0], P[:, 1], mk, ms=4.2 if jj == 1 else 2.6,
                   mfc='none' if jj == 1 else c, mec=c, mew=0.6,
                   zorder=3 if jj == 1 else 2, label=f'$k_{jj}$')
        b.axhline(0, color=GRID, lw=0.6, zorder=0)
        b.set_xlim(-2.2, 2.2)
        b.set_ylim(-2.2, 2.2)
        b.set_xlabel(r'$p_j$')
        if col == 0:
            b.set_ylabel(r'$\ln|e/e_{p=0}|$')
            b.legend(loc='lower left', handletextpad=0.2)
        else:
            plt.setp(b.get_yticklabels(), visible=False)
        b.text(0.04, 0.95, f'({"bc"[col]}) {name}', transform=b.transAxes, ha='left', va='top', fontsize=8.5)
    fig.savefig(os.path.join(OUT, 'codim.pdf'), bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    fig_centred()
    fig_axis()
    fig_events()
    fig_imax()
    fig_null()
    fig_codim()
