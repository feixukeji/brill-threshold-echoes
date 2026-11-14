#!/usr/bin/env python3
"""Outgoing null rays from the centre along the axis or in the equatorial
plane, in the coordinates of Baumgarte et al. 2023 (Fig. 2): the emission time
T_null = -ln((tau* - tau_e) / sigma) at the centre and the affine parameter
lambda, with lambda = 0 and d lambda / d tau = 1 / (tau* - tau_e) at the
centre; along every ray the rescaled Kretschmann scalar (tau* - tau_e) |I|^(1/4).

    python src/vacuum/gh_null.py <out>_round0_lines.npz <out>_round0.json <member> axis|equator <tau*> <out.npz>

The input is a gh_bisect run with --set lines=1: at every refresh the metric
g_hat, its T derivative and the Kretschmann scalar on the axis (X node 0) and
on the equator (Z node 0).  The axis and the equatorial line {Y = Z = 0} are
fixed-point sets of isometries (the rotations about the axis; the reflections
z -> -z and y -> -y), hence totally geodesic: a null geodesic that starts
along one stays on it, and the 2-metric induced there fixes it.  The rays are
traced in (T, x = L q), q = Z or X, where the metric
    G_TT = L^2 (g_TT + 2 ell g_Tq q + ell^2 g_qq q^2),
    G_Tx = L (g_Tq + ell g_qq q),   G_xx = g_qq
does not jump when the zoom rate ell changes (a reframe keeps the lapse, the
spatial metric and the shift relative to x)."""
from __future__ import annotations

import json
import sys

import numpy as np
from scipy.interpolate import CubicSpline

COMP = {'axis': (0, 2, 5), 'equator': (0, 1, 3)}     # (TT, Tq, qq) among the 7 stored components


class Line:
    """The 2-metric G_AB(T, x) and the Kretschmann scalar I(T, x) (physical)
    on the axis or the equator, interpolated between the refresh records:
    cubic splines in x (extended to x < 0 by parity), 4-point Lagrange in T."""

    def __init__(self, z, m, which, Tmax):
        self.T = np.asarray(z['T'], dtype=float)
        k = self.T <= Tmax + 1e-9
        self.T = self.T[k]
        L = np.exp(np.asarray(z['lnL'], dtype=float)[k])
        ell = np.asarray(z['ell'], dtype=float)[k]
        self.lnL = np.log(L)
        rec = np.asarray(z['ax' if which == 'axis' else 'eq'][k][:, m], dtype=float)     # (nT, 14, nq)
        Ih = np.asarray(z['Iax' if which == 'axis' else 'Ieq'][k][:, m], dtype=float)     # (nT, nq) at L = 1
        iTT, iTq, iqq = COMP[which]
        self.sp, self.xmax = [], []
        for i in range(len(self.T)):
            q = np.asarray(z['Z'] if which == 'axis' else z['X'][np.flatnonzero(k)[i]], dtype=float)
            gTT, gTq, gqq = rec[i, iTT], rec[i, iTq], rec[i, iqq]
            GTT = L[i] ** 2 * (gTT + 2 * ell[i] * gTq * q + ell[i] ** 2 * gqq * q ** 2)
            GTx = L[i] * (gTq + ell[i] * gqq * q)
            Gxx = gqq
            I = Ih[i] / L[i] ** 4
            x = L[i] * q
            xs = np.r_[-x[::-1], x]
            ev = lambda f: np.r_[f[::-1], f]
            od = lambda f: np.r_[-f[::-1], f]
            self.sp.append(CubicSpline(xs, np.stack([ev(GTT), od(GTx), ev(Gxx), ev(I)], axis=1)))
            self.xmax.append(0.9 * x[-1])

    def _weights(self, T):
        i = int(np.clip(np.searchsorted(self.T, T) - 2, 0, len(self.T) - 4))
        t = self.T[i:i + 4]
        w, dw = np.zeros(4), np.zeros(4)
        for a in range(4):
            o = [b for b in range(4) if b != a]
            den = np.prod([t[a] - t[b] for b in o])
            w[a] = np.prod([T - t[b] for b in o]) / den
            dw[a] = sum(np.prod([T - t[b] for b in o if b != c]) for c in o) / den
        return i, w, dw

    def metric(self, T, x):
        """G (2, 2), dG[c] (c = T, x) and I at (T, x)."""
        i, w, dw = self._weights(T)
        v = np.array([self.sp[i + a](x) for a in range(4)])                          # (4, 4)
        vx = np.array([self.sp[i + a](x, 1)[:3] for a in range(4)])                  # (4, 3)
        f, fT, fx = w @ v, dw @ v[:, :3], w @ vx
        G = np.array([[f[0], f[1]], [f[1], f[2]]])
        dG = np.array([[[fT[0], fT[1]], [fT[1], fT[2]]], [[fx[0], fx[1]], [fx[1], fx[2]]]])
        return G, dG, f[3]

    def inside(self, T, x):
        i, _, _ = self._weights(T)
        return abs(x) < min(self.xmax[i:i + 4])


def ray(line, Te, tau_e, tstar, lmax=2.5, dl=0.005, hl=0.00125, dT=0.005):
    """The ray emitted at T = Te from x = 0: lambda, T, x, I along it, on a
    grid of step dl in lambda.  Classical RK4 with step min(hl, dT / k^T), so
    that a step never spans more than dT in T (the metric is interpolated
    between records 0.1 apart; an adaptive integrator stalls on the kinks).
    Halving hl and dT changes T at lambda = 2 by <= 0.03 and I there by <= 10%."""
    G, _, _ = line.metric(Te, 0.0)
    N = np.sqrt(-1.0 / np.linalg.inv(G)[0, 0])
    kT = (tstar - tau_e) / N
    a, b, c = G[1, 1], 2 * G[0, 1] * kT, G[0, 0] * kT ** 2
    kx = (-b + np.sqrt(b * b - 4 * a * c)) / (2 * a)

    def rhs(y):
        G, dG, I = line.metric(y[0], y[1])
        k = y[2:]
        # Gamma^A_BC k^B k^C = G^AD (dG[B,D,C] - dG[D,B,C] / 2) k^B k^C
        t = np.einsum('bdc,b,c->d', dG, k, k) - 0.5 * np.einsum('dbc,b,c->d', dG, k, k)
        return np.r_[k, -np.linalg.solve(G, t)], I

    y, lam = np.array([Te, 0.0, kT, kx]), 0.0
    L, Y, Is = [], [], []
    while lam < lmax and line.inside(y[0], y[1]) and y[0] < line.T[-2]:
        k1, I = rhs(y)
        L.append(lam), Y.append(y[:2].copy()), Is.append(I)
        h = min(hl, dT / max(abs(y[2]), 1e-12))
        k2, _ = rhs(y + 0.5 * h * k1)
        k3, _ = rhs(y + 0.5 * h * k2)
        k4, _ = rhs(y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        lam += h
    L, Y, Is = np.array(L), np.array(Y), np.array(Is)
    lg = np.arange(0.0, L[-1], dl)
    return lg, np.interp(lg, L, Y[:, 0]), np.interp(lg, L, Y[:, 1]), np.interp(lg, L, Is)


def main():
    lines, rj, m, which, tstar, out = sys.argv[1:7]
    m, tstar = int(m), float(tstar)
    z = np.load(lines, allow_pickle=True)
    d = json.load(open(rj))
    h = d['hist']
    sigma = d['template']['sigma']
    T = np.array(h['T'])
    L = np.exp(np.array(h['lnL']))
    a0 = np.array(h['a0'], dtype=float)[:, m]
    tc = h['tc'][0] + np.r_[0.0, np.cumsum(0.5 * (L[1:] * a0[1:] + L[:-1] * a0[:-1]) * np.diff(T))]
    Tend = float(d['members'][m][2])
    line = Line(z, m, which, Tend)
    Te = np.arange(line.T[1], Tend - 0.2, 0.1)
    tau_e = np.interp(Te, T, tc)
    Te = Te[tau_e < tstar - 1e-3]
    res = []
    for T0 in Te:
        te = float(np.interp(T0, T, tc))
        lam, Tr, xr, Ir = ray(line, T0, te, tstar)
        res.append((T0, te, lam, Tr, xr, Ir))
        print(f'T_e {T0:6.2f} tau_e {te:.5f} T_null {-np.log((tstar - te) / sigma):+.3f}: '
              f'lambda to {lam[-1]:.3f}, x to {xr[-1]:.4f}', flush=True)
    np.savez(out, which=which, tstar=tstar, sigma=sigma, Te=np.array([r[0] for r in res]),
             tau_e=np.array([r[1] for r in res]),
             **{f'{k}{i}': r[j] for i, r in enumerate(res) for j, k in ((2, 'lam'), (3, 'T'), (4, 'x'), (5, 'I'))})


if __name__ == '__main__':
    main()
