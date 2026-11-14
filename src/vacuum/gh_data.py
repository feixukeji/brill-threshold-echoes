#!/usr/bin/env python3
"""Initial data for gh_sim: Brill waves (time symmetric vacuum).

    python src/vacuum/gh_data.py         # solver check

    gamma = psi^4 [ e^{2q} (drho^2 + dz^2) + rho^2 dphi^2 ],   K_ij = 0,

with the Hamiltonian constraint, linear in psi,

    flat_laplacian(psi) + psi (q_,rho rho + q_,zz) / 4 = 0,   psi -> 1.

It is solved by collocation on all of R^3: r = r_s eta / (1 - eta^2) with
eta in (0, 1] on the positive half of a CGL grid (psi even in eta, so the
centre is regular without a node there), and mu = cos(theta) on the positive
half of a Gauss-Legendre grid (equatorial symmetry, psi even in mu).  psi = 1
is imposed at eta = 1 (spatial infinity).  The dense linear system is small.
"""
from __future__ import annotations

import os
import sys

import numpy as np
from numpy.polynomial import legendre as Lg

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/
from vacuum.gh_sim import _cgl_full


def q_holz(A, sigma, eta=1.0, eps=0.0):
    """q = A (rho/sigma)^2 exp(-rho^2/sigma^2 - z^2/(eta sigma)^2) (1 + eps z/sigma)
    and its rho,rho + z,z Laplacian (as functions of rho, z).  eta = 1 is
    Holz's form (sigma then only sets the scale); eta != 1 (axial over radial
    width) is a second shape parameter; eps != 0 breaks the equatorial
    symmetry (an odd-in-z deformation)."""
    def q(rho, z):
        return (A * rho ** 2 / sigma ** 2 * np.exp(-rho ** 2 / sigma ** 2 - z ** 2 / (eta * sigma) ** 2)
                * (1 + eps * z / sigma))

    def lap2(rho, z):          # q_,rho rho + q_,zz
        s2 = sigma ** 2
        t2 = (eta * sigma) ** 2
        e = np.exp(-rho ** 2 / s2 - z ** 2 / t2)
        o = 1 + eps * z / sigma
        # d_rho^2 [rho^2 e] = e (2 - 10 rho^2/s2 + 4 rho^4/s2^2)
        # d_z^2   [rho^2 e o] = rho^2 e ((-2/t2 + 4 z^2/t2^2) o - 4 z eps / (t2 sigma))
        return A / s2 * e * ((2 - 10 * rho ** 2 / s2 + 4 * rho ** 4 / s2 ** 2) * o
                             + rho ** 2 * ((-2 / t2 + 4 * z ** 2 / t2 ** 2) * o - 4 * z * eps / (t2 * sigma)))
    return q, lap2


def shape_coef(rng, deg=3):
    """The coefficients of a random q_shape deformation: c[i + 1, j]
    multiplies x^(i+1) y^j (i + j <= deg), drawn N(0, 1)/(i+j)!."""
    from math import factorial
    c = np.zeros((deg + 2, deg + 1))
    for i in range(deg + 1):
        for j in range(deg + 1 - i):
            c[i + 1, j] = rng.standard_normal() / factorial(i + j)
    return c


def q_poly(c, A, sigma, eta=1.0):
    """dq = A Q(x, y) exp(-x - y/eta^2), x = rho^2/sigma^2, y = z^2/sigma^2,
    Q = sum c[i, j] x^i y^j, and its rho,rho + z,z Laplacian."""
    from numpy.polynomial import polynomial as P2
    b = 1.0 / eta ** 2
    cx, cy = P2.polyder(c, axis=0), P2.polyder(c, axis=1)
    cxx, cyy = P2.polyder(c, 2, axis=0), P2.polyder(c, 2, axis=1)

    def q(rho, z):
        x, y = np.broadcast_arrays(rho ** 2 / sigma ** 2, z ** 2 / sigma ** 2)
        return A * np.exp(-x - b * y) * P2.polyval2d(x, y, c)

    def lap2(rho, z):          # (2/sigma^2) (F_x + 2x F_xx + F_y + 2y F_yy) of F(x, y)
        x, y = np.broadcast_arrays(rho ** 2 / sigma ** 2, z ** 2 / sigma ** 2)
        v = lambda k: P2.polyval2d(x, y, k)
        Q, Qx, Qy, Qxx, Qyy = v(c), v(cx), v(cy), v(cxx), v(cyy)
        return (2 * A / sigma ** 2 * np.exp(-x - b * y)
                * (Qx - Q + 2 * x * (Qxx - 2 * Qx + Q) + Qy - b * Q + 2 * y * (Qyy - 2 * b * Qy + b * b * Q)))
    return q, lap2


def q_shape(rng, A, sigma, eta=1.0, deg=3):
    """A random equatorially symmetric deformation of q_holz(A, sigma, eta):
    q_poly with Q = x P(x, y), P a polynomial of total degree <= deg
    (shape_coef).  Brill data with q_holz + h dq solve the constraints
    exactly, so their differences are physical directions.  The deformations
    of one degree span a space of dimension (deg + 1)(deg + 2)/2 that
    contains q_holz itself (P = 1) and its eta derivative (P = y)."""
    return q_poly(shape_coef(rng, deg), A, sigma, eta)


class BrillSolver:
    def __init__(self, nr=48, nmu=16, rs=1.0, full=False):
        """full: no equatorial symmetry (all of mu in [-1, 1], every l)."""
        m = 2 * nr
        self.full = full
        s, D = _cgl_full(m)
        idx = np.arange(nr, m)
        mir = m - 1 - idx
        eta = s[idx]
        self.eta, self.rs = eta, rs
        with np.errstate(divide='ignore'):         # (eta = 1 is spatial infinity)
            r = rs * eta / (1 - eta ** 2)
            dr_de = rs * (1 + eta ** 2) / (1 - eta ** 2) ** 2
            d2r_de2 = rs * 2 * eta * (eta ** 2 + 3) / (1 - eta ** 2) ** 3
        # psi(-r, mu) = psi(r, -mu): the mirror radial node is the point in the
        # opposite direction (equatorial symmetry: the same direction)
        De = (D[np.ix_(idx, idx)], D[np.ix_(idx, mir)])
        D2e = ((D @ D)[np.ix_(idx, idx)], (D @ D)[np.ix_(idx, mir)])
        self.r = r
        # Gauss-Legendre in mu, positive half, even functions
        xg, wg = Lg.leggauss(2 * nmu)
        pos = np.ones_like(xg, dtype=bool) if full else xg > 0
        self.mu, self.wmu = xg[pos], wg[pos]
        ls = np.arange(0, 2 * nmu, 1 if full else 2)
        self.ls = ls
        Pm = np.array([Lg.legval(self.mu, np.eye(l + 1)[l]) for l in ls]).T   # (nmu, nl)
        # even extension: int_{-1}^{1} = 2 int_0^1
        norm = (2 * ls + 1) / 2.0
        Pinv = (norm[:, None] * Pm.T * self.wmu[None, :] * (1.0 if full else 2.0))   # (nl, nmu)
        Lang = Pm @ np.diag(-ls * (ls + 1.0)) @ Pinv
        self.Pm, self.Pinv = Pm, Pinv
        self.De, self.D2e, self.dr_de, self.d2r_de2 = De, D2e, dr_de, d2r_de2
        self.Lang = Lang
        self.nr, self.nmu = nr, len(self.mu)
        nm = len(self.mu)
        self.flip = np.eye(nm)[::-1] if full else np.eye(nm)     # mu -> -mu (Gauss nodes are symmetric)

    def solve(self, lap2q):
        nr, nm = self.nr, self.nmu
        r = self.r
        inner = slice(0, nr - 1)                 # eta = 1 is the last node
        re, dre, d2re = r[inner], self.dr_de[inner], self.d2r_de2[inner]
        Rad = []                                 # same-direction and mirror parts, (nr-1, nr) each
        for De, D2e in zip(self.De, self.D2e):
            Dr = De[inner] / dre[:, None]
            Drr = D2e[inner] / dre[:, None] ** 2 - (d2re / dre ** 3)[:, None] * De[inner]
            Rad.append(Drr + (2.0 / re)[:, None] * Dr)
        R2, MU = np.meshgrid(re, self.mu, indexing='ij')
        rho, z = R2 * np.sqrt(1 - MU ** 2), R2 * MU
        V = 0.25 * lap2q(rho, z)                 # (nr-1, nm)
        # unknown psi[i, j], i over all nr radial nodes, j over mu
        N = nr * nm
        A = np.zeros(((nr - 1) * nm, N))
        Im = np.eye(nm)
        A += np.kron(Rad[0], Im) + np.kron(Rad[1], self.flip)
        E = np.zeros((nr - 1, nr))
        E[np.arange(nr - 1), np.arange(nr - 1)] = 1.0
        A += np.kron(E / re[:, None] ** 2, self.Lang)
        A += np.kron(E, Im) * V.ravel()[:, None]
        # boundary rows psi(eta = 1) = 1
        B = np.zeros((nm, N))
        B[:, (nr - 1) * nm:] = np.eye(nm)
        M = np.vstack([A, B])
        rhs = np.concatenate([np.zeros((nr - 1) * nm), np.ones(nm)])
        psi = np.linalg.solve(M, rhs).reshape(nr, nm)
        self.psi = psi
        return psi

    def interp(self, rho, z):
        """psi at arbitrary (rho, z) (spectral: Chebyshev in eta, Legendre in mu)."""
        r = np.hypot(rho, z)
        mu = np.where(r > 0, (z if self.full else np.abs(z)) / np.where(r > 0, r, 1), 1.0)
        # eta(r) inverting r = rs eta/(1-eta^2)
        x = r / self.rs
        eta = np.where(x > 0, (np.sqrt(1 + 4 * x ** 2) - 1) / (2 * np.where(x > 0, x, 1)), 0.0)
        # Legendre coefficients per radial node
        cl = self.psi @ self.Pinv.T              # (nr, nl)
        # Chebyshev coefficients in eta of the even extension, per l
        m = 2 * self.nr
        s, _ = _cgl_full(m)
        sg = (-1.0) ** self.ls if self.full else np.ones(len(self.ls))
        full = np.vstack([cl[::-1] * sg, cl])    # values at s ascending (parity (-1)^l)
        th = np.arccos(np.clip(s, -1, 1))
        V = np.cos(np.outer(th, np.arange(m)))
        ck = np.linalg.solve(V, full)            # (m, nl)
        Te = np.cos(np.outer(np.arccos(np.clip(eta.ravel(), -1, 1)), np.arange(m)))
        fl = Te @ ck                             # (npts, nl)
        Pl = np.array([Lg.legval(mu.ravel(), np.eye(l + 1)[l]) for l in self.ls]).T
        return np.sum(fl * Pl, axis=1).reshape(np.shape(rho))

    def adm_mass(self):
        """From the l = 0 mode: psi_0 -> 1 + M/(2r)."""
        cl = self.psi @ self.Pinv.T
        # near eta = 1: psi_0 - 1 ~ M/(2r); r(eta) large
        i = self.nr - 3
        return 2 * self.r[i] * (cl[i, 0] - 1.0)


def brill_data(G, A, sigma, ell=1.0, gh=None, nr=48, nmu=16, rs=None, eta=1.0, eps=0.0, extra=()):
    """The gh_sim state of the Brill wave q_holz(A, sigma, eta, eps) on G at
    zoom rate ell (unit lapse, zero physical shift), and the solver.
    eps != 0 needs a full-Z grid (G.zsym False); the Hamiltonian constraint
    is then solved without equatorial symmetry, with 2 nmu Gauss-Legendre
    nodes in mu.  extra: (h, (dq, lap2 dq)) terms added to q (q_shape; each
    dq must vanish like rho^2 on the axis)."""
    from vacuum import gh_sim
    if eps and G.zsym:
        raise ValueError('eps != 0 breaks the equatorial symmetry: needs a full-Z grid (zsym = 0)')
    q0, l0 = q_holz(A, sigma, eta, eps)
    q = lambda rho, z: q0(rho, z) + sum(h * f[0](rho, z) for h, f in extra)
    lap2 = lambda rho, z: l0(rho, z) + sum(h * f[1](rho, z) for h, f in extra)
    S = BrillSolver(nr, nmu, rs if rs else sigma, full=not G.zsym)
    S.solve(lap2)
    X, Z = G.X, G.Z
    psi = S.interp(X, Z)
    p4 = psi ** 4
    e2q = np.exp(2 * q(X, Z))
    # (e^{2q} - 1)/X^2 without cancellation: q = A X^2 E / sigma^2
    qq = q(X, Z)
    W = p4 * np.expm1(2 * qq) / X ** 2
    gam = {'XX': p4 * e2q, 'ZZ': p4 * e2q, 'XZ': np.zeros_like(X), 'W': W}
    return gh_sim.from_3p1(G, gam, {}, np.ones_like(X), ell, gh=gh), S


def check_brill_solver():
    """Weak field: psi = 1 + psi1 with lap psi1 = -lap2 q / 4; compare two
    resolutions and the known ADM-mass scaling M ~ A^2 (quadratic, since the
    l = 0 part of lap2 q integrates to zero at first order)."""
    out = []
    for nr, nm in ((32, 10), (48, 16)):
        S = BrillSolver(nr, nm, 1.0)
        q, lap2 = q_holz(1.0, 1.0)
        S.solve(lap2)
        out.append((S, S.interp(np.array([0.3, 1.0]), np.array([0.2, 0.5]))))
    d = np.max(np.abs(out[0][1] - out[1][1]))
    masses = []
    for A in (0.5, 1.0):
        S = BrillSolver(48, 16, 1.0)
        S.solve(q_holz(A, 1.0)[1])
        masses.append(S.adm_mass())
    return d, masses


if __name__ == '__main__':
    d, m = check_brill_solver()
    print(f"psi, two resolutions: max diff {d:.2e};  M(A=0.5) = {m[0]:.6f}, "
          f"M(A=1) = {m[1]:.6f}, ratio {m[1]/m[0]:.3f} (quadratic: 4)")
