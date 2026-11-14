#!/usr/bin/env python3
"""Apparent horizons of axisymmetric slices of a gh_sim state.

A star-shaped surface r = h(theta) about the origin, theta in [0, pi/2]
(equatorial symmetry), h = sum_k a_k cos(2 k theta), or about an axis point
(0, zc), theta in [0, pi], h = sum_k a_k cos(k theta), is an apparent
horizon when the outgoing expansion

    Theta = D_i s^i - K + K_ij s^i s^j,    s^i = gamma^ij d_j F / |dF|,
    F = r - h(theta),

vanishes.  Everything is computed in the rescaled frame (horizons are
scale-free).  The 3-metric and extrinsic curvature are interpolated from the
grid with quintic splines on the parity-extended data and turned into
(r, theta, phi) components; D_i s^i = (1/sqrt(gamma)) d_i (sqrt(gamma) s^i)
is formed with a small central-difference stencil around each surface point.
This is a classifier, not a precision tool: second-order accurate in the
stencil, spline-limited in the fields.

Newton on the coefficients a_k from spheres of several radii; the outermost
converged solution is reported with its areal mass sqrt(A / 16 pi).

    python src/vacuum/gh_ah.py           # check on a time-symmetric slice"""
from __future__ import annotations

import os
import sys

import numpy as np
from scipy.interpolate import RectBivariateSpline

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/
from vacuum import gh_sim as g


def slice_data(G, gh, u):
    """3-metric and K_ij (Cartesian components XX, XZ, ZZ, YY) on the grid,
    K_ij = -(d_t gamma_ij - D_i beta_j - D_j beta_i) / (2 alpha) from the
    physical 4-metric derivatives at L = 1."""
    g7, P7, phi, Pi, ell = g.unpack(G, u)
    X = G.X
    gam = {'XX': g7[3], 'XZ': g7[4], 'ZZ': g7[5], 'YY': g7[3] - X ** 2 * g7[6]}
    du, aux = gh.rhs(u, full=True)
    alpha = 1 / np.sqrt(-aux['gi'][0, 0])
    dg = aux['dg']                      # (c, a, b, N), physical at L = 1
    g4 = aux['g']
    gi3 = np.moveaxis(np.linalg.inv(np.moveaxis(g4[1:, 1:], -1, 0)), 0, -1)
    d3 = dg[1:, 1:, 1:]                 # d_k gamma_ij
    G3l = 0.5 * (np.einsum('jkiN->kijN', d3) + np.einsum('ikjN->kijN', d3) - d3)  # Gamma_kij
    G3u = np.einsum('klN,lijN->kijN', gi3, G3l)
    b3 = g4[0, 1:]                      # beta_i
    # D_i beta_j = d_i beta_j - Gamma^k_ij beta_k, with dg[1:, 0, 1:][i, j] = d_i beta_j
    Db = dg[1:, 0, 1:] - np.einsum('kijN,kN->ijN', G3u, b3)
    Kij = -(dg[0, 1:, 1:] - Db - np.einsum('ijN->jiN', Db)) / (2 * alpha)
    K = {}
    for key, (i, j) in {'XX': (0, 0), 'XZ': (0, 2), 'ZZ': (2, 2), 'YY': (1, 1)}.items():
        K[key] = Kij[i, j].reshape(G.shape)
    return gam, K


class Interp:
    """Quintic splines of parity-extended grid functions."""

    def __init__(self, G, fields, parities):
        x, z = G.ax.x, G.az.x
        xe = np.concatenate([-x[::-1], x])
        ze = np.concatenate([-z[::-1], z]) if G.zsym else z
        self.sp = {}
        for k, f in fields.items():
            px, pz = parities[k]
            fe = np.concatenate([px * f[::-1], f], axis=0)
            if G.zsym:
                fe = np.concatenate([pz * fe[:, ::-1], fe], axis=1)
            self.sp[k] = RectBivariateSpline(xe, ze, fe, kx=5, ky=5)

    def __call__(self, k, X, Z):
        return self.sp[k].ev(X, Z)


PARS = {'XX': (1, 1), 'XZ': (-1, -1), 'ZZ': (1, 1), 'YY': (1, 1)}


def spherical(I, r, th, pre, zc=0.0):
    """(rr, rth, thth, phph) components of metric (pre='g') or K (pre='K')
    in spherical coordinates centred at (0, zc) on the axis."""
    s, c = np.sin(th), np.cos(th)
    X, Z = r * s, zc + r * c
    xx, xz, zz, yy = (I(pre + k, X, Z) for k in ('XX', 'XZ', 'ZZ', 'YY'))
    rr = s * s * xx + 2 * s * c * xz + c * c * zz
    rt = r * (s * c * (xx - zz) + (c * c - s * s) * xz)
    tt = r * r * (c * c * xx - 2 * s * c * xz + s * s * zz)
    pp = r * r * s * s * yy
    return rr, rt, tt, pp


def expansion(I, a, th, eps=1e-4, zc=None):
    """Theta at theta nodes.  zc None: surface centred at the origin with
    equatorial symmetry, h = sum a_k cos(2 k theta), theta in (0, pi/2);
    zc a number: centred at (0, zc), h = sum a_k cos(k theta), theta in
    (0, pi)."""
    k = np.arange(len(a))
    m = 2 if zc is None else 1
    z0 = 0.0 if zc is None else zc

    def h(t):
        return np.cos(m * np.outer(t, k)) @ a

    def hp(t):
        return -(m * k * np.sin(m * np.outer(t, k))) @ a

    def field(r, t):
        """sqrt(gamma) s^r, sqrt(gamma) s^th, and pieces for K terms."""
        grr, grt, gtt, gpp = spherical(I, r, t, 'g', z0)
        det2 = grr * gtt - grt ** 2
        irr, irt, itt = gtt / det2, -grt / det2, grr / det2
        dF = (np.ones_like(t), -hp(t))
        ur = irr * dF[0] + irt * dF[1]
        ut = irt * dF[0] + itt * dF[1]
        N = np.sqrt(ur * dF[0] + ut * dF[1])
        sq = np.sqrt(det2 * gpp)
        return sq * ur / N, sq * ut / N, sq, ur / N, ut / N

    r0 = h(th)
    er = eps * np.maximum(r0, 1e-3)
    fr_p = field(r0 + er, th)[0]
    fr_m = field(r0 - er, th)[0]
    ft_p = field(r0, th + eps)[1]
    ft_m = field(r0, th - eps)[1]
    _, _, sq, sr, st = field(r0, th)
    div = ((fr_p - fr_m) / (2 * er) + (ft_p - ft_m) / (2 * eps)) / sq
    Krr, Krt, Ktt, Kpp = spherical(I, r0, th, 'K', z0)
    grr, grt, gtt, gpp = spherical(I, r0, th, 'g', z0)
    det2 = grr * gtt - grt ** 2
    irr, irt, itt = gtt / det2, -grt / det2, grr / det2
    trK = irr * Krr + 2 * irt * Krt + itt * Ktt + Kpp / gpp
    Kss = Krr * sr * sr + 2 * Krt * sr * st + Ktt * st * st
    return div - trK + Kss


def area(I, a, nq=64, zc=None):
    k = np.arange(len(a))
    t, w = np.polynomial.legendre.leggauss(nq)
    if zc is None:
        th, w, m, z0, sym = 0.25 * np.pi * (t + 1), 0.25 * np.pi * w, 2, 0.0, 2.0
    else:
        th, w, m, z0, sym = 0.5 * np.pi * (t + 1), 0.5 * np.pi * w, 1, zc, 1.0
    h = np.cos(m * np.outer(th, k)) @ a
    hp = -(m * k * np.sin(m * np.outer(th, k))) @ a
    grr, grt, gtt, gpp = spherical(I, h, th, 'g', z0)
    # induced 2-metric: q_thth = grr h'^2 + 2 grt h' + gtt, q_phph = gpp
    q = grr * hp ** 2 + 2 * grt * hp + gtt
    return sym * 2 * np.pi * np.sum(w * np.sqrt(q * gpp))


def find_ah(G, gh, u, radii=None, nk=12, tol=1e-9, maxit=40, accept=1e-3,
            a_start=None, zc=None, interp=None):
    """Outermost MOTS r = h(theta), h = sum_{k<nk} a_k cos(2k theta).

    Gauss-Newton on 3 nk collocation points (more equations than unknowns,
    so a shape that zeroes Theta only at the nodes is not a solution), from
    spheres of several radii (and a_start, e.g. the previous horizon).  A
    candidate is ACCEPTED only if max |Theta| h_mean < accept on a fine
    theta grid and its cosine coefficients decay (|a_{nk-1}| < 1e-3 a_0);
    checking only at the collocation nodes let oscillating non-solutions
    through."""
    I = interp if interp is not None else make_interp(G, gh, u)
    m = 3 * nk
    top = np.pi / 2 if zc is None else np.pi
    mult = 2 if zc is None else 1
    th = 0.5 * top * (1 - np.cos(np.pi * (np.arange(m) + 0.5) / m))       # in (0, top)
    thf = np.linspace(1e-3, top - 1e-3, 200)
    rmax = 0.9 * G.L if zc is None else 0.9 * min(G.L - abs(zc), G.L)
    # geometric starts down to 0.01: a horizon forming in a collapsing
    # centre can be much smaller than the frame's unit length
    radii = list(radii) if radii is not None else list(np.geomspace(0.01, rmax, 18))
    starts = []
    if a_start is not None:
        a0 = np.zeros(nk)
        a0[:min(nk, len(a_start))] = a_start[:nk]
        starts.append(a0)
    for r0 in radii:
        a0 = np.zeros(nk)
        a0[0] = r0
        starts.append(a0)
    found = []
    for a in starts:
        a = a.copy()
        for it in range(maxit):
            Th = expansion(I, a, th, zc=zc)
            if not np.all(np.isfinite(Th)):
                break
            if np.max(np.abs(Th)) * a[0] < tol:
                break
            J = np.zeros((m, nk))
            for j in range(nk):
                da = np.zeros(nk)
                da[j] = 1e-6
                J[:, j] = (expansion(I, a + da, th, zc=zc) - Th) / 1e-6
            step = np.linalg.lstsq(J, -Th, rcond=None)[0]
            lam = 1.0
            n0 = np.linalg.norm(Th)
            while lam > 1e-3:
                an = a + lam * step
                hn = np.cos(mult * np.outer(thf, np.arange(nk))) @ an
                if hn.min() > 0.02 and hn.max() < rmax:
                    Tn = expansion(I, an, th, zc=zc)
                    if np.all(np.isfinite(Tn)) and np.linalg.norm(Tn) < n0:
                        break
                lam *= 0.5
            if lam <= 1e-3:
                break
            a = a + lam * step
        Thf = expansion(I, a, thf, zc=zc)
        hf = np.cos(mult * np.outer(thf, np.arange(nk))) @ a
        if (np.all(np.isfinite(Thf)) and hf.min() > 0.02
                and np.max(np.abs(Thf)) * np.mean(hf) < accept
                and abs(a[-1]) < 1e-3 * abs(a[0])):
            found.append((a.copy(), area(I, a, zc=zc), float(np.max(np.abs(Thf)) * np.mean(hf))))
    if not found:
        return None
    a, A, res = max(found, key=lambda f: f[0][0])
    return dict(a=a, area=A, mass=np.sqrt(A / (16 * np.pi)), n=len(found), resid=res, zc=zc)


def make_interp(G, gh, u):
    gam, K = slice_data(G, gh, u)
    fields = {('g' + k): v for k, v in gam.items()}
    fields.update({('K' + k): v for k, v in K.items()})
    pars = {('g' + k): PARS[k] for k in PARS}
    pars.update({('K' + k): PARS[k] for k in PARS})
    return Interp(G, fields, pars)


def find_any_ah(G, gh, u, Kgrid=None, nk=12):
    """Centred search, then off-centre searches about the axis points where
    the Kretschmann scalar and the lapse have their extrema (near-threshold
    A < 0 Brill waves peak on the axis away from the origin)."""
    I = make_interp(G, gh, u)
    res = find_ah(G, gh, u, nk=nk, interp=I)
    if res is not None:
        return res
    if Kgrid is None:
        from vacuum.gh_sim import riemann_invariant
        Kgrid = riemann_invariant(G, gh, u)
    zs = G.az.x
    Ka = np.abs(Kgrid[0])
    cands = [float(zs[i]) for i in range(1, len(zs) - 1) if Ka[i] >= Ka[i - 1] and Ka[i] >= Ka[i + 1]]
    alpha = (1 / np.sqrt(-g.inv_metric(G, u)[0, 0])).reshape(G.shape)
    cands.append(float(zs[np.argmin(alpha[0])]))
    cands += list(np.linspace(0.1, 0.8, 8) * G.L)
    seen = []
    for zc in cands:
        if any(abs(zc - q) < 0.03 * G.L for q in seen):
            continue
        seen.append(zc)
    for zc in seen:
        if zc < 0.05 * G.L:
            continue
        rmax = 0.9 * min(zc, G.L - zc)
        r = find_ah(G, gh, u, nk=nk, zc=zc, interp=I,
                    radii=np.linspace(0.05 * rmax, rmax, 10))
        if r is not None:
            return r
    return None


class Stretched:
    """Quintic splines of the parity-extended slice data on X < xm, Z < zm,
    seen in the coordinate X' = s X (components rescaled), so that a
    horizon next to the axis, where g_XX >> g_ZZ, is nearly round."""

    def __init__(self, G, gam, K, xm, zm, s):
        x, z = G.ax.x, G.az.x
        i, j = x < xm, z < zm
        xe = np.concatenate([-x[i][::-1], x[i]])
        ze = np.concatenate([-z[j][::-1], z[j]])
        self.s = s
        self.sp = {}
        for pre, F in (('g', gam), ('K', K)):
            for k, f in F.items():
                px, pz = PARS[k]
                fe = f[np.ix_(i, j)]
                fe = np.concatenate([px * fe[::-1], fe], axis=0)
                fe = np.concatenate([pz * fe[:, ::-1], fe], axis=1)
                self.sp[pre + k] = RectBivariateSpline(xe, ze, fe, kx=5, ky=5)
        self.fac = {'XX': s ** -2, 'YY': s ** -2, 'XZ': 1 / s, 'ZZ': 1.0}

    def __call__(self, k, X, Z):
        return self.sp[k].ev(X / self.s, Z) * self.fac[k[1:]]


def axis_search(G, gh, u, xm=2.5, zm=3.0, nk=12, nr=40):
    """Horizon search for a structure on the axis near the center: about the
    origin and about the axis minimum zc of the lapse, in X and in
    X' = s X with s = sqrt(g_XX / g_ZZ) at zc.  For each of the four,
    the sphere (r = const) with the smallest max Theta r, the smallest
    min Theta r over all spheres (flat space: 2), and find_ah's MOTS or
    None.  Returns (zc, s, rows)."""
    gam, K = slice_data(G, gh, u)
    alpha = (1 / np.sqrt(-g.inv_metric(G, u)[0, 0])).reshape(G.shape)
    ja = int(np.argmin(alpha[0]))
    zc0 = float(G.az.x[ja])
    s0 = float(np.sqrt(gam['XX'][0, ja] / gam['ZZ'][0, ja]))
    rows = []
    for s in (1.0, s0):
        I = Stretched(G, gam, K, xm, zm, s)
        for zc in (None, zc0):
            th = np.linspace(1e-3, (np.pi / 2 if zc is None else np.pi) - 1e-3, 200)
            rmax = 1.5 if zc is None else min(1.2, 0.95 * zc0)
            best, low = (np.inf, None), (np.inf, None)
            for r in np.geomspace(0.02, rmax, nr):
                a = np.zeros(nk)
                a[0] = r
                Th = expansion(I, a, th, zc=zc)
                if np.all(np.isfinite(Th)):
                    best = min(best, (Th.max() * r, r))
                    low = min(low, (Th.min() * r, r))
            res = find_ah(G, gh, u, nk=nk, zc=zc, interp=I, radii=np.geomspace(0.02, rmax, 14))
            rows.append(dict(s=s, zc=zc, best=best, low=low, mots=res))
    return zc0, s0, rows


def check_spherical(a=1.2, sw=0.6, n=32):
    """Smooth time-symmetric conformally flat slice, psi = 1 + a exp(-r^2/sw^2):
    spheres have Theta ~ psi + 2 r psi', whose outermost root r_h gives the
    horizon and area 4 pi r_h^2 psi(r_h)^4.  Also checks K_ij = 0."""
    from scipy.optimize import brentq
    G = g.Grid(n, n, 3.0, 1.5, 1.5, b=0.9)
    gh = g.GH(G, gamma0=0.0, scalar=False)
    psi = 1 + a * np.exp(-G.R ** 2 / sw ** 2)
    p4 = psi ** 4
    u = g.from_3p1(G, {'XX': p4, 'ZZ': p4, 'W': 0 * p4}, {}, np.ones_like(p4), 0.45, gh=gh)
    gam, K = slice_data(G, gh, u)
    Kmax = max(np.abs(v).max() for v in K.values())
    f = lambda r: (1 + a * np.exp(-r ** 2 / sw ** 2)) + 2 * r * (-2 * r / sw ** 2) * a * np.exp(-r ** 2 / sw ** 2)
    rs = np.linspace(0.01, 2.5, 500)
    roots = [brentq(f, rs[i], rs[i + 1]) for i in range(len(rs) - 1) if f(rs[i]) * f(rs[i + 1]) < 0]
    rh = max(roots) if roots else None
    Ah = 4 * np.pi * rh ** 2 * (1 + a * np.exp(-rh ** 2 / sw ** 2)) ** 4 if rh else None
    res = find_ah(G, gh, u)
    return Kmax, rh, Ah, res


if __name__ == '__main__':
    Kmax, rh, Ah, res = check_spherical()
    print(f"time-symmetric slice: max|K_ij| = {Kmax:.1e}; expected horizon r = {rh:.5f}, "
          f"area {Ah:.5f}; found", res and dict(a=np.round(res['a'][:3], 6), area=res['area'], resid=res['resid']))
