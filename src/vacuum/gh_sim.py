#!/usr/bin/env python3
"""Generalized-harmonic evolution of axisymmetric spacetimes in similarity
coordinates, with an outer boundary that needs no data.

Coordinates and variables
-------------------------
x^a = (T, X, Y, Z), Cartesian and rescaled: the physical metric is
g = L(T)^2 ghat with d ln L / dT = -ell, so at constant ell = 1 the physical
coordinates t = -e^{-T}, x^i = e^{-T} X^i put the accumulation point at the
origin of T -> infinity, and a discretely self-similar spacetime whose
coordinates respect the symmetry is a periodic orbit of this system.  ell is
piecewise constant; a change of ell is an exact change of chart (`reframe`).

Every term of the generalized-harmonic (GH) equations is homogeneous of degree
zero under a constant rescaling of g, so the equations for ghat carry no
explicit T once the gauge source H_a and the damping rate are rescaled the same
way.  We therefore evaluate the PHYSICAL GH equations at the instant where
L = 1, with the substitutions

    d_T g   = P - 2 ell ghat,
    d_T^2 g = d_T P - 4 ell P + 4 ell^2 ghat,   P = d_T ghat,

and solve for d_T P.

Gauge
-----
The gauge source

    H_a = ell g_ab c_b g^{bT},   c = (1, 2, 2, 2),

says that the underlying (t, x^i) are harmonic, box t = box x^i = 0; the flat
metric ghat = -dT^2 + (dX - ell X dT)^2 is then a static solution for every
constant ell.  On top of it the damped-wave terms of Lindblom & Szilagyi
(2009) in the rescaled frame (`GH.hfunc`): the shift term damps
b = beta + ell X (the shift relative to the underlying coordinates), the lapse
term drives log alpha to a target, which with the maximal-lapse driver is an
evolved field lA (the phi slot) relaxing to the log of the lapse that keeps
the slices maximal (`maximal_lapse`, re-solved every refresh).  Constraint
damping follows Gundlach et al. (2005) with a rate gamma0 alpha^g0p that is
constant in the rescaled frame (1/(t*-t) physically).

Axisymmetry (cartoon)
---------------------
Fields live on the half plane Y = 0, X >= 0, and (by default) Z >= 0 with
reflection symmetry.  Cartesian components are smooth with parity
(-1)^{#X+#Y} in X and (-1)^{#Z} in Z; components with an odd number of Y
indices vanish on the plane.  Y-derivatives follow from invariance under
rotations about Z (`_cartoon`).  The one component with no parity protection
at the axis is g_YY, which must equal g_XX there; we evolve
W = (g_XX - g_YY) / X^2 instead, so regularity holds by construction.  Every
derivative of g_YY is formed by the product rule from g_XX and W, never by
differentiating nodal values of g_YY.

Grid: in each direction the positive half of a 2n-point Chebyshev-Gauss-
Lobatto grid on [-1, 1] (no node at 0), mapped by x = L sinh(a k(s)) / sinh(a);
derivatives act on the parity-extended function.  The outer faces X = L and
Z = L are grid points.  With ell L > 1 they are spacelike in flat space
(the similarity horizon is at |X| = 1/ell), so every characteristic leaves
and no boundary condition is imposed; `outflow_margin` monitors this.

    python src/vacuum/gh_sim.py          # self-checks, ~30 s
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/

T_, X_, Y_, Z_ = 0, 1, 2, 3
# evolved metric components (a, b) and their parities (in X, in Z); 'W' is
# (g_XX - g_YY) / X^2
COMPS = [(T_, T_), (T_, X_), (T_, Z_), (X_, X_), (X_, Z_), (Z_, Z_), 'W']
NAMES = ['TT', 'TX', 'TZ', 'XX', 'XZ', 'ZZ', 'W']


def _parity(c):
    if c == 'W':
        return 1, 1
    a, b = c
    nx = sum(i in (X_, Y_) for i in (a, b))
    nz = sum(i == Z_ for i in (a, b))
    return (-1) ** nx, (-1) ** nz


PAR = [_parity(c) for c in COMPS]
NG = len(COMPS)
# state layout: g(7), P(7), phi, Pi (all on the grid), then ell.  In vacuum
# with the maximal-lapse driver the phi slot holds the lapse target lA and Pi
# is unused; the ACTIVE fields are the first NA (g, P, lA), the part that a
# GPU batch (gh_cuda) and the stored checkpoints carry
NF = 2 * NG + 2
NA = 2 * NG + 1
FPAR = PAR + PAR + [(1, 1), (1, 1)]


# ----------------------------------------------------------------------------
# grid
# ----------------------------------------------------------------------------
def _cgl_full(m):
    """m-point CGL nodes ascending on [-1, 1] and d/ds."""
    k = np.arange(m)
    s = -np.cos(np.pi * k / (m - 1))
    c = np.ones(m)
    c[0] = c[-1] = 2.0
    c *= (-1.0) ** k
    S = np.tile(s, (m, 1)).T
    dS = S - S.T
    D = np.outer(c, 1.0 / c) / (dS + np.eye(m))
    D -= np.diag(D.sum(axis=1))
    return s, D


class Axis:
    """Half of a 2n-point CGL grid, s in (0, 1], mapped to x in (0, L] by

        x = L sinh(a k(s)) / sinh(a),   k(s) = arcsin(b s) / arcsin(b),

    both odd in s so parity is kept.  a > 0 pulls nodes toward the centre;
    b in (0, 1) (Kosloff-Tal-Ezer) spreads the O(1/n^2) clustering at the
    outer edge, where the characteristic speeds are largest.

    D[p], D2[p]: d/dx and d^2/dx^2 acting on nodal values of a function with
    parity p (the result has parity -p resp. p).  F[p]: exponential filter.
    If sym is False the axis is the full n-point grid on [-L, L] and the
    parity argument is ignored."""

    def __init__(self, n, L, a=0.0, sym=True, filt=(36.0, 16), b=0.0):
        self.n, self.L, self.a, self.b, self.sym = n, L, a, b, sym
        m = 2 * n if sym else n
        s, D = _cgl_full(m)
        self.s_full = s
        if b > 0:
            asb = np.arcsin(b)
            k = np.arcsin(b * s) / asb
            ks = b / (asb * np.sqrt(1 - (b * s) ** 2))
            kss = b ** 3 * s / (asb * (1 - (b * s) ** 2) ** 1.5)
        else:
            k, ks, kss = s, np.ones(m), np.zeros(m)
        if a > 0:
            sa = np.sinh(a)
            x = L * np.sinh(a * k) / sa
            xs = L * a * np.cosh(a * k) * ks / sa
            xss = L * a * (a * np.sinh(a * k) * ks ** 2 + np.cosh(a * k) * kss) / sa
        else:
            x, xs, xss = L * k, L * ks, L * kss
        Dx = D / xs[:, None]
        Dxx = (D @ D) / (xs ** 2)[:, None] - (xss / xs ** 3)[:, None] * D
        # exponential filter in Chebyshev coefficients (in s) of the full grid
        V = np.cos(np.outer(np.arccos(np.clip(s, -1, 1)), np.arange(m)))
        kk = np.arange(m) / (m - 1)
        sig = np.exp(-filt[0] * kk ** filt[1])
        Ff = V @ np.diag(sig) @ np.linalg.inv(V)
        if sym:
            idx = np.arange(n, m)
            mir = m - 1 - idx
            self.x = x[idx]
            self.D, self.D2, self.F = {}, {}, {}
            for p in (1, -1):
                self.D[p] = Dx[np.ix_(idx, idx)] + p * Dx[np.ix_(idx, mir)]
                self.D2[p] = Dxx[np.ix_(idx, idx)] + p * Dxx[np.ix_(idx, mir)]
                self.F[p] = Ff[np.ix_(idx, idx)] + p * Ff[np.ix_(idx, mir)]
        else:
            self.x = x
            self.D = {1: Dx, -1: Dx}
            self.D2 = {1: Dxx, -1: Dxx}
            self.F = {1: Ff, -1: Ff}
        # quadrature weights: Clenshaw-Curtis in s times dx/ds
        from vacuum.cone_numerics import cc_weights
        wcc = cc_weights(m) * xs
        self.w = wcc[idx] if sym else wcc

    def s_of_x(self, x):
        k = np.arcsinh(x * np.sinh(self.a) / self.L) / self.a if self.a > 0 else x / self.L
        return np.sin(k * np.arcsin(self.b)) / self.b if self.b > 0 else k

    def interp(self, xnew, p=1):
        """Matrix taking nodal values (parity p) to the polynomial interpolant
        in s at the points xnew (barycentric, full CGL grid)."""
        s = self.s_full
        m = s.size
        w = (-1.0) ** np.arange(m)
        w[0] *= 0.5
        w[-1] *= 0.5
        sn = self.s_of_x(np.asarray(xnew, float))
        diff = sn[:, None] - s[None, :]
        hit = np.abs(diff) < 1e-14
        diff[hit] = 1.0
        B = w[None, :] / diff
        B /= B.sum(axis=1, keepdims=True)
        rows = hit.any(axis=1)
        B[rows] = hit[rows].astype(float)
        if not self.sym:
            return B
        idx = np.arange(self.n, m)
        return B[:, idx] + p * B[:, m - 1 - idx]


class Grid:
    def __init__(self, nx=24, nz=24, L=3.0, ax=1.5, az=1.5, zsym=True,
                 filt=(36.0, 16), b=0.0):
        self.ax = Axis(nx, L, ax, True, filt, b)
        self.az = Axis(nz, L, az, zsym, filt, b)
        self.zsym = zsym
        self.L = L
        self.X, self.Z = np.meshgrid(self.ax.x, self.az.x, indexing='ij')
        self.shape = self.X.shape
        self.N = self.X.size
        self.R = np.hypot(self.X, self.Z)

    # derivatives of a 2-D field with parity (px, pz)
    def dx(self, f, px):
        return self.ax.D[px] @ f

    def dz(self, f, pz):
        return f @ self.az.D[pz].T

    def dxx(self, f, px):
        return self.ax.D2[px] @ f

    def dzz(self, f, pz):
        return f @ self.az.D2[pz].T

    def dxz(self, f, px, pz):
        return self.ax.D[px] @ f @ self.az.D[pz].T

    def filt(self, f, px, pz):
        return self.ax.F[px] @ f @ self.az.F[pz].T

    def integrate(self, f):
        """int f dV over the 3-D region the half plane represents
        (2 pi X dX dZ, doubled if the Z >= 0 half stands for both)."""
        w = np.outer(self.ax.w * 2 * np.pi * self.ax.x, self.az.w)
        return (2.0 if self.zsym else 1.0) * np.sum(w * f)



def grid_from_config(cfg, n=None):
    """The Grid of a configuration dict (keys n, nz, L, ax, az, b, filtp,
    zsym; nz / az absent or 0: same as n / ax).  n overrides the X resolution,
    and nz then scales with it."""
    nx = n or cfg['n']
    nz = cfg.get('nz') or cfg['n']
    nz = int(round(nz * nx / cfg['n']))
    return Grid(nx, nz, cfg['L'], cfg['ax'], cfg.get('az') or cfg['ax'], b=cfg['b'],
                filt=(36.0, cfg.get('filtp', 16)), zsym=bool(cfg.get('zsym', 1)))


def make(cfg, n=None):
    """Grid and vacuum GH operator of a configuration dict (as
    gh_bisect.CONFIG; other keys are ignored)."""
    G = grid_from_config(cfg, n)
    mx = (cfg['maximal'], cfg['cK']) if cfg.get('maximal', 0) > 0 else None
    gh = GH(G, gamma0=cfg['gamma0'], scalar=False, muL=cfg['muL'], muS=cfg['muS'], pL=cfg['pL'],
            qL=cfg['qL'], g0p=cfg.get('g0p', 0.0), maximal=mx, mgrid=cfg.get('mgrid', 0),
            mfixed=cfg.get('mfixed', 0), afloor=cfg.get('afloor', 1e-3), lbnd=cfg.get('lbnd', 'one'))
    return G, gh


# ----------------------------------------------------------------------------
# state helpers
# ----------------------------------------------------------------------------
def fields(G, u):
    """The grid fields of u as an (NF, nx, nz) array."""
    return u[:-1].reshape(NF, *G.shape)


def full_state(G, v, ell):
    """The state u of an active vector v (NA fields; phi = lA, Pi = 0)."""
    F = np.zeros((NF, G.N))
    F[:NA] = np.asarray(v, dtype=np.float64).reshape(NA, G.N)
    return np.concatenate([F.ravel(), [ell]])


def active(G, u):
    """The active vector (NA fields) of a state u."""
    return fields(G, u)[:NA].ravel().copy()


def transfer_fields(G_src, F, G_dst, ell):
    """Fields F (k <= NF, nx, nz), in the layout of a state, interpolated to
    G_dst (spectrally, on the deviation of the metric from the flat
    similarity metric, every field with its parities; G_dst must lie inside
    G_src)."""
    fs, _ = _flat_parts(G_src, ell)
    fd, _ = _flat_parts(G_dst, ell)
    out = np.empty((F.shape[0],) + G_dst.shape)
    for i in range(F.shape[0]):
        px, pz = FPAR[i]
        f = F[i] - fs[i] if i < NG else F[i]
        out[i] = G_src.ax.interp(G_dst.ax.x, px) @ f @ G_src.az.interp(G_dst.az.x, pz).T + (fd[i] if i < NG else 0.0)
    return out


def recentre_fields(G_src, F, G_dst, ell, zeta):
    """The active fields F (NA, nx, nz) of a state on G_src, in the chart
    centred on the axis point Z = zeta of the old one (z' = z - zeta L at
    the same instant, zoom rate and slicing), on the full-Z grid G_dst
    (|Z'| <= G_dst.L must lie inside G_src: G_dst.L <= G_src.L - |zeta|).

    A constant translation keeps the physical coordinates harmonic and the
    physical shift b = beta + ell X^k unchanged, so the chart is again one
    of this code's.  At fixed Z' the old Z = Z' + zeta L0 / L moves at
    v = ell zeta: d_T' = d_T + v d_Z, hence ghat_TT += 2 v ghat_TZ +
    v^2 ghat_ZZ, ghat_TX += v ghat_XZ, ghat_TZ += v ghat_ZZ, and the
    spatial P += v d_Z ghat.  P_TT, P_TX, P_TZ are returned as zero: they
    follow from C'_a = 0 (CudaGH.solve_PT), as in reframe."""
    assert not G_dst.zsym and G_dst.L <= G_src.L - abs(zeta) + 1e-12
    fs, _ = _flat_parts(G_src, ell)
    Ix = {p: G_src.ax.interp(G_dst.ax.x, p) for p in (1, -1)}
    Iz = {p: G_src.az.interp(G_dst.az.x + zeta, p) for p in (1, -1)}
    X, Zs = G_dst.X, G_dst.Z + zeta                      # the new nodes in the old chart
    one, zero = np.ones(G_dst.shape), np.zeros(G_dst.shape)
    fl = [-(1 - ell ** 2 * (X ** 2 + Zs ** 2)), -ell * X, -ell * Zs, one, zero, one, zero]
    g7, dz7, P7 = [], [], []
    for i, (px, pz) in enumerate(PAR):
        h = F[i] - fs[i]
        g7.append(Ix[px] @ h @ Iz[pz].T + fl[i])
        dz7.append(Ix[px] @ G_src.dz(h, pz) @ Iz[-pz].T)  # spatial components: flat part constant
        P7.append(Ix[px] @ F[NG + i] @ Iz[pz].T)
    v = ell * zeta
    gTT = g7[0] + 2 * v * g7[2] + v * v * g7[5]
    g7[1], g7[2] = g7[1] + v * g7[4], g7[2] + v * g7[5]
    g7[0] = gTT
    for k in (3, 4, 5, 6):
        P7[k] = P7[k] + v * dz7[k]
    P7[0], P7[1], P7[2] = zero, zero, zero
    phi = Ix[1] @ F[2 * NG] @ Iz[1].T
    return np.stack(g7 + P7 + [phi])


def transfer(G_src, u, G_dst):
    """The state u on grid G_src, interpolated to G_dst (transfer_fields)."""
    return np.concatenate([transfer_fields(G_src, fields(G_src, u), G_dst, u[-1]).ravel(), [u[-1]]])


def pack(g7, P7, phi, Pi, ell):
    return np.concatenate([np.stack(list(g7) + list(P7) + [phi, Pi]).ravel(), [ell]])


def unpack(G, u):
    F = fields(G, u)
    return list(F[:NG]), list(F[NG:2 * NG]), F[2 * NG], F[2 * NG + 1], u[-1]


def to4(G, c7):
    """7 evolved components -> symmetric 4x4 array (4, 4, nx, nz)."""
    g = np.zeros((4, 4) + G.shape, dtype=c7[0].dtype)
    for c, f in zip(COMPS[:6], c7[:6]):
        a, b = c
        g[a, b] = g[b, a] = f
    g[Y_, Y_] = c7[3] - G.X ** 2 * c7[6]
    return g


def flat_state(G, ell=1.0):
    X, Z = G.X, G.Z
    one, zero = np.ones(G.shape), np.zeros(G.shape)
    r2 = X ** 2 + Z ** 2
    g7 = [-(1 - ell ** 2 * r2), -ell * X, -ell * Z, one.copy(), zero.copy(),
          one.copy(), zero.copy()]
    P7 = [zero.copy() for _ in range(NG)]
    return pack(g7, P7, zero.copy(), zero.copy(), ell)


# ----------------------------------------------------------------------------
# derivatives of the full 4x4 metric on the plane
# ----------------------------------------------------------------------------
def _tensor_derivs(G, c7):
    """Spatial derivatives of the 7 components: dict name -> list of arrays."""
    out = {'x': [], 'z': [], 'xx': [], 'zz': [], 'xz': []}
    for f, (px, pz) in zip(c7, PAR):
        out['x'].append(G.dx(f, px))
        out['z'].append(G.dz(f, pz))
        out['xx'].append(G.dxx(f, px))
        out['zz'].append(G.dzz(f, pz))
        out['xz'].append(G.dxz(f, px, pz))
    return out


def _yy_from(G, t, W, dt=None, dW=None, kind=None):
    """Derivatives of t_YY = t_XX - X^2 W by the product rule."""
    X = G.X
    if kind is None:
        return t - X ** 2 * W
    if kind == 'x':
        return dt - 2 * X * W - X ** 2 * dW
    if kind == 'z':
        return dt - X ** 2 * dW
    raise ValueError(kind)


def _cartoon(G, t4, W, dxt4, dxW, dzt4, dzW):
    """Y-derivatives on the plane of an axisymmetric covariant 2-tensor.

    t4: (4,4,...) components on the plane (odd-Y components zero), W its
    (t_XX - t_YY)/X^2; dxt4, dzt4 the X and Z derivatives of the components
    (with the YY entry already formed by the product rule), dxW, dzW those of
    W.  Returns dY[a,b], dYY[a,b], dXY[a,b], dZY[a,b]."""
    X = G.X
    iX = 1.0 / X
    sh = t4.shape
    dY = np.zeros(sh, dtype=t4.dtype)
    dYY = np.zeros(sh, dtype=t4.dtype)
    dXY = np.zeros(sh, dtype=t4.dtype)
    dZY = np.zeros(sh, dtype=t4.dtype)

    def sym(A, a, b, v):
        A[a, b] = v
        A[b, a] = v

    # first derivatives: components with one Y index
    sym(dY, T_, Y_, t4[T_, X_] * iX)
    sym(dY, X_, Y_, X * W)
    sym(dY, Z_, Y_, t4[Z_, X_] * iX)
    # d_Y d_Y: components with zero or two Y indices
    sym(dYY, T_, T_, dxt4[T_, T_] * iX)
    sym(dYY, T_, X_, (dxt4[T_, X_] - t4[T_, X_] * iX) * iX)
    sym(dYY, T_, Z_, dxt4[T_, Z_] * iX)
    sym(dYY, X_, X_, dxt4[X_, X_] * iX - 2 * W)
    sym(dYY, X_, Z_, (dxt4[X_, Z_] - t4[X_, Z_] * iX) * iX)
    sym(dYY, Z_, Z_, dxt4[Z_, Z_] * iX)
    sym(dYY, Y_, Y_, dxt4[Y_, Y_] * iX + 2 * W)
    # d_X d_Y: one Y index
    sym(dXY, T_, Y_, (dxt4[T_, X_] - t4[T_, X_] * iX) * iX)
    sym(dXY, X_, Y_, W + X * dxW)
    sym(dXY, Z_, Y_, (dxt4[Z_, X_] - t4[Z_, X_] * iX) * iX)
    # d_Z d_Y
    sym(dZY, T_, Y_, dzt4[T_, X_] * iX)
    sym(dZY, X_, Y_, X * dzW)
    sym(dZY, Z_, Y_, dzt4[Z_, X_] * iX)
    return dY, dYY, dXY, dZY


def _flat_parts(G, ell):
    """The 7 components of the flat similarity metric
    -dT^2 + (dX - ell X dT)^2 and their analytic X, Z derivatives."""
    X, Z = G.X, G.Z
    one, zero = np.ones(G.shape), np.zeros(G.shape)
    fl = [-(1 - ell ** 2 * (X ** 2 + Z ** 2)), -ell * X, -ell * Z, one, zero, one, zero]
    d = {'x': [2 * ell ** 2 * X, -ell * one, zero, zero, zero, zero, zero],
         'z': [2 * ell ** 2 * Z, zero, -ell * one, zero, zero, zero, zero],
         'xx': [2 * ell ** 2 * one] + [zero] * 6,
         'zz': [2 * ell ** 2 * one] + [zero] * 6,
         'xz': [zero] * 7}
    return fl, d


def metric_derivs(G, g7, P7, ell, ddT=None):
    """g[a,b], dg[c,a,b], ddg[c,d,a,b] of the PHYSICAL metric at L = 1.

    ddg[T,T] is set to ddT (default zero): the caller treats it as unknown."""
    g = to4(G, g7)
    # physical first time derivative d_T g = P - 2 ell ghat (all 7 comps, W too)
    gT7 = [p - 2 * ell * f for p, f in zip(P7, g7)]
    gT = to4(G, gT7)
    # spectral derivatives act on the deviation from the flat similarity
    # metric; the flat part (a quadratic in X, not a polynomial in the mapped
    # grid coordinate) is differentiated analytically
    fl, dfl = _flat_parts(G, ell)
    Dg = _tensor_derivs(G, [f - b for f, b in zip(g7, fl)])
    for k in dfl:
        for i in range(NG):
            Dg[k][i] = Dg[k][i] + dfl[k][i]
    hT7 = [p - 2 * ell * (f - b) for p, f, b in zip(P7, g7, fl)]
    DgT = {'x': [G.dx(f, px) - 2 * ell * dfl['x'][i] for i, (f, (px, pz)) in enumerate(zip(hT7, PAR))],
           'z': [G.dz(f, pz) - 2 * ell * dfl['z'][i] for i, (f, (px, pz)) in enumerate(zip(hT7, PAR))]}

    def full(c7d, W, dW, kind):
        t = to4(G, c7d[:6] + [np.zeros(G.shape)])
        t[Y_, Y_] = _yy_from(G, None, W, c7d[3], dW, kind)
        return t

    W, WT = g7[6], gT7[6]
    dxg = full(Dg['x'], W, Dg['x'][6], 'x')
    dzg = full(Dg['z'], W, Dg['z'][6], 'z')
    dxgT = full(DgT['x'], WT, DgT['x'][6], 'x')
    dzgT = full(DgT['z'], WT, DgT['z'][6], 'z')
    X = G.X
    # second derivatives of g_YY by the product rule
    dxxg = to4(G, Dg['xx'][:6] + [np.zeros(G.shape)])
    dxxg[Y_, Y_] = Dg['xx'][3] - 2 * W - 4 * X * Dg['x'][6] - X ** 2 * Dg['xx'][6]
    dzzg = to4(G, Dg['zz'][:6] + [np.zeros(G.shape)])
    dzzg[Y_, Y_] = Dg['zz'][3] - X ** 2 * Dg['zz'][6]
    dxzg = to4(G, Dg['xz'][:6] + [np.zeros(G.shape)])
    dxzg[Y_, Y_] = Dg['xz'][3] - 2 * X * Dg['z'][6] - X ** 2 * Dg['xz'][6]

    dYg, dYYg, dXYg, dZYg = _cartoon(G, g, W, dxg, Dg['x'][6], dzg, Dg['z'][6])
    dYgT, _, _, _ = _cartoon(G, gT, WT, dxgT, DgT['x'][6], dzgT, DgT['z'][6])

    sh = (4, 4, 4) + G.shape
    dg = np.zeros(sh, dtype=g.dtype)
    dg[T_], dg[X_], dg[Y_], dg[Z_] = gT, dxg, dYg, dzg
    ddg = np.zeros((4,) + sh, dtype=g.dtype)
    if ddT is not None:
        ddg[T_, T_] = ddT

    def put(c, d, v):
        ddg[c, d] = v
        ddg[d, c] = v

    put(T_, X_, dxgT)
    put(T_, Z_, dzgT)
    put(T_, Y_, dYgT)
    put(X_, X_, dxxg)
    put(Z_, Z_, dzzg)
    put(X_, Z_, dxzg)
    put(Y_, Y_, dYYg)
    put(X_, Y_, dXYg)
    put(Z_, Y_, dZYg)
    return g, dg, ddg


# ----------------------------------------------------------------------------
# the generalized-harmonic operator
# ----------------------------------------------------------------------------
CS = np.array([1.0, 2.0, 2.0, 2.0])     # underlying-harmonic gauge


def _flat(A):
    return A.reshape(A.shape[:A.ndim - 2] + (-1,))


def inv_metric(G, u):
    """g^ab (4, 4, N) of the rescaled metric of state u, pointwise from the
    stored components (what rhs(full=True)['gi'] returns, without any
    derivatives)."""
    g = to4(G, list(fields(G, u)[:NG])).reshape(4, 4, G.N)
    return np.moveaxis(np.linalg.inv(np.moveaxis(g, -1, 0)), 0, -1)


def _spectral_derivs(G, h, hT):
    """X, Z, XX, ZZ, XZ derivatives of the 7 components h (NG, nx, nz) and X,
    Z derivatives of hT, as a few large matrix products (grouped by parity:
    many small ones keep a threaded BLAS idle)."""
    nx, nz = G.shape
    D = {k: np.empty((NG, nx, nz)) for k in ('x', 'z', 'xx', 'zz', 'xz', 'Tx', 'Tz')}
    for pz in (1, -1):
        idx = [i for i, p in enumerate(PAR) if p[1] == pz]
        k = len(idx)
        R = (np.concatenate([h[idx], hT[idx]]).reshape(2 * k * nx, nz)
             @ np.hstack([G.az.D[pz].T, G.az.D2[pz].T])).reshape(2 * k, nx, 2 * nz)
        D['z'][idx] = R[:k, :, :nz]
        D['zz'][idx] = R[:k, :, nz:]
        D['Tz'][idx] = R[k:, :, :nz]
    for px in (1, -1):
        idx = [i for i, p in enumerate(PAR) if p[0] == px]
        k = len(idx)
        B = np.concatenate([h[idx], hT[idx], D['z'][idx]]).transpose(1, 0, 2).reshape(nx, 3 * k * nz)
        R1 = (G.ax.D[px] @ B).reshape(nx, 3 * k, nz).transpose(1, 0, 2)
        R2 = (G.ax.D2[px] @ B[:, :k * nz]).reshape(nx, k, nz).transpose(1, 0, 2)
        D['x'][idx] = R1[:k]
        D['Tx'][idx] = R1[k:2 * k]
        D['xz'][idx] = R1[2 * k:]
        D['xx'][idx] = R2
    return D

class GH:
    """Right-hand side of the rescaled GH system (vacuum or massless scalar).

    gamma0, g0p: constraint damping at the rate gamma0 alpha^g0p.  muL, muS,
    pL, qL: damped-wave lapse and shift terms (`hfunc`).  maximal = (kappa,
    cK), vacuum only: the lapse target is the evolved field lA (phi slot),
    d_T lA = -kappa (lA - lamM), with lamM = log alpha_M refreshed from
    `maximal_lapse` (cK: see there); mgrid > 0 solves for alpha_M on an
    mgrid-point grid, mfixed > 0 with exactly mfixed GMRES steps
    preconditioned by a frozen LU (so the solve is a smooth function of the
    state), afloor the floor of alpha_M, lbnd its value on the outer faces:
    1 ('one') or the lapse of the state there ('state', for a box cut inside
    the strong field).  fast: numba kernels (gh_kernels); otherwise the
    einsum reference."""

    def __init__(self, G: Grid, gamma0=1.0, scalar=True, cs=CS, filt_every=True, muL=0.0, muS=0.0,
                 pL=1.0, qL=1.0, g0p=0.0, maximal=None, mgrid=0, mfixed=0, afloor=1e-3, lbnd='one', fast=True):
        assert not (maximal and scalar)
        self.G = G
        self.gamma0, self.g0p = gamma0, g0p
        self.muL, self.muS, self.pL, self.qL = muL, muS, pL, qL
        self.maximal = maximal
        self.mgrid, self.mfixed, self.afloor = mgrid, mfixed, afloor
        assert lbnd in ('one', 'state')
        self.lbnd = lbnd
        self.lamM = np.zeros(G.shape)
        self.lu_fixed = None            # the frozen LU of the maximal-lapse solve (mfixed)
        self.scalar = scalar
        self.cs = np.asarray(cs, float)
        self.filt_every = filt_every
        self.fast = fast
        self._fcache = None

    def geometry(self, g7, P7, ell, ddT=None):
        G = self.G
        g, dg, ddg = metric_derivs(G, g7, P7, ell, ddT)
        g, dg, ddg = _flat(g), _flat(dg), _flat(ddg)
        gi = np.linalg.inv(np.moveaxis(g, -1, 0))
        gi = np.moveaxis(gi, 0, -1)
        dgi = -np.einsum('aeN,bfN,cefN->cabN', gi, gi, dg)
        Gl = 0.5 * (np.einsum('acbN->cabN', dg) + np.einsum('bcaN->cabN', dg) - dg)
        Gu = np.einsum('cdN,dabN->cabN', gi, Gl)
        return g, dg, ddg, gi, dgi, Gl, Gu

    def hfunc(self, g, Xv, ell, lref=None):
        """Gauge source H_a(g, X, ell), analytic so that complex steps work.

        Underlying-harmonic part ell g_ab c_b g^{bT}, plus the damped-wave
        terms of Lindblom & Szilagyi (2009) written in the rescaled frame:
            mu_L alpha^(qL-1) log(gamma^{p/2} e^lref / alpha) n_a - mu_S gamma_ai b^i / alpha,
        (p = pL; p = 0 drives the rescaled lapse to e^lref, i.e. keeps
        coordinate time close to central proper time; lref is the
        maximal-lapse target lA, else zero), with b^i = beta^i + ell X^i the
        shift relative to the underlying coordinates (zero in the flat
        similarity metric, as is the log)."""
        gi = np.moveaxis(np.linalg.inv(np.moveaxis(g, -1, 0)), 0, -1)
        F = ell * self.cs[:, None] * gi[:, T_]
        H = np.einsum('abN,bN->aN', g, F)
        if self.muL or self.muS:
            H = H.copy()
            alpha = 1.0 / np.sqrt(-gi[T_, T_])
            beta_u = -gi[T_, 1:] / gi[T_, T_]              # beta^i
            if self.muL:
                gam = np.moveaxis(g[1:, 1:], -1, 0)
                lsq = 0.5 * self.pL * np.log(np.linalg.det(gam))
                if lref is not None:
                    lsq = lsq + lref
                H[T_] = H[T_] + self.muL * (lsq - np.log(alpha)) * (-alpha ** self.qL)
            if self.muS:
                b = beta_u + ell * Xv                        # (3, N)
                H[T_] = H[T_] - self.muS * np.einsum('iN,iN->N', g[T_, 1:], b) / alpha
                H[1:] = H[1:] - self.muS * np.einsum('ijN,jN->iN', g[1:, 1:], b) / alpha
        return H

    def gauge(self, g, gi, dg, dgi, ell, lref=None, dlref=None):
        """H_a and d_c H_a.  The pure underlying-harmonic source has closed
        form derivatives; otherwise d_c H follows by complex steps along
        (d_c g, d_c X, d_c lref)."""
        G = self.G
        cs = self.cs
        if not (self.muL or self.muS):
            F = ell * cs[:, None] * gi[:, T_]                     # F^b
            H = np.einsum('abN,bN->aN', g, F)
            dF = ell * cs[None, :, None] * dgi[:, :, T_]           # d_c F^b
            dH = np.einsum('cabN,bN->caN', dg, F) + np.einsum('abN,cbN->caN', g, dF)
            return H, dH
        # H is a function of ghat; the damping terms have weight 1 under
        # g -> lambda^2 g (mu is a rate), so d_T H must follow d_T ghat = P,
        # not the physical d_T g = P - 2 ell ghat (for the weight-0 harmonic
        # part the two agree).
        Xv = np.stack([G.X.ravel(), np.zeros(G.N), G.Z.ravel()])
        H = self.hfunc(g, Xv, ell, lref)
        h = 1e-30
        dH = np.zeros((4,) + H.shape)
        for c in range(4):
            dX = np.zeros_like(Xv)
            if c > 0:
                dX[c - 1] = 1.0
            dgc = dg[c] + 2 * ell * g if c == T_ else dg[c]
            Hc = self.hfunc(g + 1j * h * dgc, Xv + 1j * h * dX, ell,
                            None if lref is None else lref + 1j * h * dlref[c])
            dH[c] = Hc.imag / h
        return H, dH

    def operator(self, g, dg, ddg, gi, dgi, Gl, Gu, H, dH):
        """E_ab = R_ab - nabla_(a C_b) + damping, with ddg as given."""
        # d_c Gamma^c_ab
        t1 = (np.einsum('cceN,eabN->abN', dgi, Gl)
              + 0.5 * np.einsum('ceN,caebN->abN', gi, ddg)
              + 0.5 * np.einsum('ceN,cbeaN->abN', gi, ddg)
              - 0.5 * np.einsum('ceN,ceabN->abN', gi, ddg))
        # d_b Gamma^c_ac
        t2 = (0.5 * np.einsum('bceN,aceN->abN', dgi, dg)
              + 0.5 * np.einsum('ceN,abceN->abN', gi, ddg))
        trG = np.einsum('ccdN->dN', Gu)                        # Gamma^c_cd
        t3 = np.einsum('dN,dabN->abN', trG, Gu)
        t4 = np.einsum('cbdN,dacN->abN', Gu, Gu)
        R = t1 - t2 + t3 - t4
        Gam = np.einsum('deN,bdeN->bN', gi, Gl)                # Gamma_b
        dGam = (np.einsum('adeN,bdeN->abN', dgi, Gl)
                + 0.5 * np.einsum('deN,adbeN->abN', gi, ddg)
                + 0.5 * np.einsum('deN,aebdN->abN', gi, ddg)
                - 0.5 * np.einsum('deN,abdeN->abN', gi, ddg))
        C = H + Gam
        E = (R - 0.5 * (dGam + np.einsum('abN->baN', dGam))
             + np.einsum('cabN,cN->abN', Gu, Gam)
             - 0.5 * (dH + np.einsum('abN->baN', dH))
             + np.einsum('cabN,cN->abN', Gu, H))
        if self.gamma0:
            alpha = 1.0 / np.sqrt(-gi[T_, T_])
            n_l = np.zeros_like(C)
            n_l[T_] = -alpha
            n_u = np.einsum('abN,bN->aN', gi, n_l)
            nC = np.einsum('aN,aN->N', n_u, C)
            E = E + self.gamma0 * alpha ** self.g0p * (0.5 * (n_l[:, None] * C[None] + C[:, None] * n_l[None])
                                                       - 0.5 * g * nC)
        return E, R, C

    def _lapse_target(self, phi):
        """lref = lA (N,) and d_c lref (4, N) for the maximal-lapse driver
        (d_T lA from its relaxation, on the plane d_Y lA = 0), else None."""
        if not (self.maximal and self.muL):
            return None, None
        G = self.G
        dl = np.zeros((4, G.N))
        dl[T_] = (-self.maximal[0] * (phi - self.lamM)).ravel()
        dl[X_] = G.dx(phi, 1).ravel()
        dl[Z_] = G.dz(phi, 1).ravel()
        return phi.ravel(), dl

    def _fused_ok(self, u):
        return self.fast and not self.scalar and u.dtype == np.float64

    def _rhs_fused(self, u, Cout=None):
        """rhs(u) through gh_kernels.gh_fused (vacuum): the spectral
        derivatives are taken per parity group, and the pointwise algebra,
        cartoon terms included, is one kernel."""
        from vacuum.gh_kernels import gh_fused
        G = self.G
        nx, nz = G.shape
        Fl = fields(G, u)
        g7, P7, phi = Fl[:NG], Fl[NG:2 * NG], Fl[2 * NG]
        ell = float(u[-1])
        fc = self._fcache
        if fc is None or fc[0] != ell:
            fl, dfl = _flat_parts(G, ell)
            fc = (ell, np.array(fl), {k: np.array(v) for k, v in dfl.items()})
            self._fcache = fc
            self._Xv = np.ascontiguousarray(np.stack([G.X.ravel(), np.zeros(G.N), G.Z.ravel()], axis=1))
        _, fl, dfl = fc
        h = g7 - fl
        hT = P7 - 2 * ell * h
        D = _spectral_derivs(G, h, hT)
        for k in ('x', 'z', 'xx', 'zz', 'xz'):
            D[k] += dfl[k]
        D['Tx'] -= 2 * ell * dfl['x']
        D['Tz'] -= 2 * ell * dfl['z']
        lref, dlref = self._lapse_target(phi)
        lref = np.zeros(G.N) if lref is None else np.ascontiguousarray(lref)
        dlref = np.zeros((G.N, 4)) if dlref is None else np.ascontiguousarray(dlref.T)
        r = lambda a: np.ascontiguousarray(a).reshape(NG, G.N)
        A7 = gh_fused(r(g7), r(D['x']), r(D['z']), r(D['xx']), r(D['zz']), r(D['xz']),
                      r(P7 - 2 * ell * g7), r(D['Tx']), r(D['Tz']), self._Xv, ell, self.cs,
                      float(self.muL), float(self.muS), float(self.gamma0), float(self.pL), lref, dlref,
                      float(self.qL), float(self.g0p),
                      np.zeros((4, 1)) if Cout is None else Cout).reshape(NG, nx, nz)
        dP7 = A7 + 4 * ell * P7 - 4 * ell ** 2 * g7
        dphi_T = -self.maximal[0] * (phi - self.lamM) if self.maximal else np.zeros(G.shape)
        return pack(P7, dP7, dphi_T, np.zeros(G.shape), 0.0)

    def constraint(self, u):
        """C_a = H_a + Gamma_a, (4, N) (through the fused kernel when it applies)."""
        if self._fused_ok(u):
            C = np.empty((4, self.G.N))
            self._rhs_fused(u, C)
            return C
        return self.rhs(u, full=True)[1]['C']

    def rhs(self, u, full=False):
        if not full and self._fused_ok(u):
            return self._rhs_fused(u)
        G = self.G
        g7, P7, phi, Pi, ell = unpack(G, u)
        lref, dlref = self._lapse_target(phi)
        if self.fast and u.dtype == np.float64:
            from vacuum.gh_kernels import gh_operator
            g, dg, ddg = metric_derivs(G, g7, P7, ell)
            g, dg, ddg = _flat(g), _flat(dg), _flat(ddg)
            Xv = np.ascontiguousarray(np.stack([G.X.ravel(), np.zeros(G.N), G.Z.ravel()], axis=1))
            En, Cn, gin, Gun = gh_operator(
                np.ascontiguousarray(np.moveaxis(g, -1, 0)),
                np.ascontiguousarray(np.moveaxis(dg, -1, 0)),
                np.ascontiguousarray(np.moveaxis(ddg, -1, 0)),
                Xv, float(ell), self.cs, float(self.muL), float(self.muS), float(self.gamma0), float(self.pL),
                np.zeros(G.N) if lref is None else np.ascontiguousarray(lref),
                np.zeros((G.N, 4)) if dlref is None else np.ascontiguousarray(dlref.T),
                float(self.qL), float(self.g0p))
            E0 = np.moveaxis(En, 0, -1)
            C = np.moveaxis(Cn, 0, -1)
            gi = np.moveaxis(gin, 0, -1)
            Gu = np.moveaxis(Gun, 0, -1)
            H = None
        else:
            g, dg, ddg, gi, dgi, Gl, Gu = self.geometry(g7, P7, ell)
            H, dH = self.gauge(g, gi, dg, dgi, ell, lref, dlref)
            E0, _, C = self.operator(g, dg, ddg, gi, dgi, Gl, Gu, H, dH)
        src = np.zeros_like(E0)
        if self.scalar:
            dphi = np.zeros((4, G.N), dtype=u.dtype)
            dphi[T_] = Pi.ravel()
            dphi[X_] = G.dx(phi, 1).ravel()
            dphi[Z_] = G.dz(phi, 1).ravel()
            src = 8 * np.pi * dphi[:, None] * dphi[None]
        A = 2.0 * (E0 - src) / gi[T_, T_]                       # physical d_T^2 g
        A7 = [A[a, b].reshape(G.shape) for (a, b) in COMPS[:6]]
        AW = (A[X_, X_] - A[Y_, Y_]).reshape(G.shape) / G.X ** 2
        A7.append(AW)
        dP7 = [a + 4 * ell * p - 4 * ell ** 2 * f for a, p, f in zip(A7, P7, g7)]
        if self.scalar:
            ddphi = np.zeros((4, 4, G.N), dtype=u.dtype)
            dPx, dPz = G.dx(Pi, 1).ravel(), G.dz(Pi, 1).ravel()
            ddphi[T_, X_] = ddphi[X_, T_] = dPx
            ddphi[T_, Z_] = ddphi[Z_, T_] = dPz
            ddphi[X_, X_] = G.dxx(phi, 1).ravel()
            ddphi[Z_, Z_] = G.dzz(phi, 1).ravel()
            ddphi[X_, Z_] = ddphi[Z_, X_] = G.dxz(phi, 1, 1).ravel()
            ddphi[Y_, Y_] = dphi[X_] / G.X.ravel()
            Gup = np.einsum('abN,cabN->cN', gi, Gu)             # Gamma^c
            box_rest = (np.einsum('abN,abN->N', gi, ddphi)
                        - np.einsum('cN,cN->N', Gup, dphi))
            dPi = (-box_rest / gi[T_, T_]).reshape(G.shape)
            dphi_T = Pi
        elif self.maximal:
            dPi = np.zeros(G.shape)
            dphi_T = -self.maximal[0] * (phi - self.lamM)
        else:
            dPi = np.zeros(G.shape)
            dphi_T = np.zeros(G.shape)
        du = pack(P7, dP7, dphi_T, dPi, 0.0)
        if full:
            return du, dict(A=A, C=C, g=g, gi=gi, dg=dg, ddg=ddg, H=H)
        return du

    # ------------------------------------------------------------------
    def filter(self, u):
        """Exponential filter of the deviation from the flat similarity
        metric (the flat part is not band-limited on the mapped grid)."""
        G = self.G
        F = fields(G, u).copy()
        fl, _ = _flat_parts(G, u[-1])
        for i in range(NG):
            F[i] -= fl[i]
        for i, (px, pz) in enumerate(FPAR):
            F[i] = G.filt(F[i], px, pz)
        for i in range(NG):
            F[i] += fl[i]
        return np.concatenate([F.ravel(), u[-1:]])

    def step(self, u, dt):
        k1 = self.rhs(u)
        k2 = self.rhs(u + 0.5 * dt * k1)
        k3 = self.rhs(u + 0.5 * dt * k2)
        k4 = self.rhs(u + dt * k3)
        un = u + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        return self.filter(un) if self.filt_every else un


def evolve(G, gh, u, tau, nstep, refresh):
    """u over tau in nstep RK4 steps, the maximal-lapse target re-solved every
    `refresh` steps (the CPU reference of gh_cuda.CudaGH.evolve)."""
    dt = tau / nstep
    for k in range(nstep):
        if gh.maximal and k % refresh == 0:
            gh.lamM = maximal_lapse(G, gh, u)
        u = gh.step(u, dt)
    return u


# ----------------------------------------------------------------------------
# changing the zoom rate between steps: an exact change of chart
# ----------------------------------------------------------------------------
def _solve_PT(G, gh, g7, P7, phi, Pi, ell):
    """P_TT, P_TX, P_TZ from C_a = 0 pointwise (C is affine in them)."""
    zero = np.zeros(G.shape)

    def state(pT):
        return pack(g7, [pT[0], pT[1], pT[2]] + list(P7[3:]), phi, Pi, ell)

    C0 = gh.constraint(state([zero, zero, zero]))
    M = np.zeros((G.N, 3, 3))
    for k in range(3):
        e = [zero, zero, zero]
        e[k] = np.ones(G.shape)
        Ck = gh.constraint(state(e))
        for r, a in enumerate((T_, X_, Z_)):
            M[:, r, k] = Ck[a] - C0[a]
    b = -np.stack([C0[a] for a in (T_, X_, Z_)], axis=-1)
    sol = np.linalg.solve(M, b[..., None])[..., 0]
    return state([sol[:, k].reshape(G.shape) for k in range(3)])


def reframe(G, gh, u, ell_new):
    """The same physical state in the chart with zoom rate ell_new.

    The physical coordinates (t, x^i) are harmonic whatever ell is, and at
    the current instant (L = 1) the new chart differs from the old only in
    d_T' = d_T + (ell - ell_new) X^k d_k.  Hence ghat_Ta and every d_T
    change algebraically; d_T of lapse and shift follow from C'_a = 0, which
    in the new chart states the same harmonic condition.  The controller
    that picks ell_new therefore never touches the physical solution."""
    g7, P7, phi, Pi, ell = unpack(G, u)
    d = ell - ell_new
    X, Z = G.X, G.Z
    gXX, gTX, gTZ, gXZ, gZZ = g7[3], g7[1], g7[2], g7[4], g7[5]
    gTT = g7[0] + 2 * d * (X * gTX + Z * gTZ) + d * d * (X * X * gXX + 2 * X * Z * gXZ + Z * Z * gZZ)
    nTX = gTX + d * (X * gXX + Z * gXZ)
    nTZ = gTZ + d * (X * gXZ + Z * gZZ)

    def adv(f, par):
        return X * G.dx(f, par[0]) + Z * G.dz(f, par[1])

    P_new = [None, None, None]
    for k in (3, 4, 5):
        P_new.append(P7[k] + d * adv(g7[k], PAR[k]))
    P_new.append(P7[6] + d * (2 * g7[6] + adv(g7[6], PAR[6])))
    Pi_new = Pi + d * adv(phi, (1, 1))
    g_new = [gTT, nTX, nTZ, gXX, gXZ, gZZ, g7[6]]
    return _solve_PT(G, gh, g_new, P_new, phi, Pi_new, ell_new)


# ----------------------------------------------------------------------------
# the maximal-lapse target
# ----------------------------------------------------------------------------
# coefficient fields of the lapse operator and their parities (X, Z)
_LAP_PAR = {'gXX': (1, 1), 'gXZ': (-1, -1), 'gZZ': (1, 1), 'gYY': (1, 1),
            'G3X': (-1, 1), 'G3Z': (1, -1), 'K': (1, 1), 'KK': (1, 1), 'alpha': (1, 1)}


def lapse_coeffs(G, gh, u):
    """Pointwise coefficients of D^2 alpha on the plane (gamma^ij, the
    contracted 3-Christoffels G3^k = gamma^ij 3Gamma^k_ij) and K, K_ij K^ij
    at the current instant (L = 1, physical), and the lapse, each (nx, nz)."""
    g7, P7, phi, Pi, ell = unpack(G, u)
    g, dg, _ = metric_derivs(G, g7, P7, ell)        # (only g and dg are needed)
    g, dg = _flat(g), _flat(dg)
    gi = inv_metric(G, u)
    Gl = 0.5 * (np.einsum('acbN->cabN', dg) + np.einsum('bcaN->cabN', dg) - dg)
    Gu = np.einsum('cdN,dabN->cabN', gi, Gl)                    # Gamma^c_ab
    alpha = 1.0 / np.sqrt(-gi[T_, T_])
    beta = -gi[T_, 1:] / gi[T_, T_]                             # beta^k
    gam_u = gi[1:, 1:] - gi[T_, 1:, None] * gi[T_, None, 1:] / gi[T_, T_]
    Kij = -alpha * Gu[T_, 1:, 1:]
    Kmix = np.einsum('ikN,kjN->ijN', gam_u, Kij)                 # K^i_j
    K = np.einsum('iiN->N', Kmix)
    KK = np.einsum('ijN,jiN->N', Kmix, Kmix)
    G3 = (np.einsum('ijN,kijN->kN', gam_u, Gu[1:, 1:, 1:])
          + beta * np.einsum('ijN,ijN->N', gam_u, Gu[T_, 1:, 1:]))  # gamma^ij 3Gamma^k_ij
    # indices 0, 1, 2 of gam_u are X, Y, Z
    co = dict(gXX=gam_u[0, 0], gXZ=gam_u[0, 2], gZZ=gam_u[2, 2], gYY=gam_u[1, 1],
              G3X=G3[0], G3Z=G3[2], K=K, KK=KK, alpha=alpha)
    return {k: v.reshape(G.shape) for k, v in co.items()}


def lapse_matrix(G, co):
    """Collocation matrix of D^2 on even-even scalars from lapse_coeffs (on
    the plane d_Y alpha = 0 and d_YY alpha = d_X alpha / X).  M[(i,k),(j,l)]
    (i, j along X, k, l along Z) = a D2x_ij d_kl + b Dx_ij Dz_kl
    + c d_ij D2z_kl + e Dx_ij d_kl - f d_ij Dz_kl, coefficients at (i, k)."""
    nx, nz = G.shape
    Dx, D2x, Dz, D2z = G.ax.D[1], G.ax.D2[1], G.az.D[1], G.az.D2[1]
    a, b, c = co['gXX'], 2 * co['gXZ'], co['gZZ']
    e = co['gYY'] / G.X - co['G3X']
    f = co['G3Z']
    M4 = np.einsum('ik,ij,kl->ikjl', b, Dx, Dz)
    kk = np.arange(nz)
    M4[:, kk, :, kk] += np.einsum('ik,ij->kij', a, D2x) + np.einsum('ik,ij->kij', e, Dx)
    ii = np.arange(nx)
    M4[ii, :, ii, :] += c[:, :, None] * D2z[None] - f[:, :, None] * Dz[None]
    return M4.reshape(nx * nz, nx * nz)


def _lapse_solve(G, gh, co):
    """log of the solution of D^2 alpha = alpha (K_ij K^ij + cK K) on G from
    the coefficients co, alpha = 1 (or co['alpha'], gh.lbnd) on the outer faces."""
    import scipy.linalg as sla
    from scipy.sparse.linalg import LinearOperator, gmres
    nx, nz = G.shape
    M = lapse_matrix(G, co)
    M[np.diag_indices_from(M)] -= co['KK'].ravel() + gh.maximal[1] * co['K'].ravel()
    rhs = np.zeros(G.N)
    bnd = ((np.arange(nx)[:, None] == nx - 1) | (np.arange(nz)[None, :] == nz - 1)
           | ((np.arange(nz)[None, :] == 0) & (not G.zsym))).ravel()
    M[bnd] = 0.0
    M[bnd, bnd] = 1.0
    rhs[bnd] = co['alpha'].ravel()[bnd] if gh.lbnd == 'state' else 1.0
    if gh.mfixed:
        # a smooth function of the state: the LU of the first operator is
        # kept as a fixed preconditioner and every solve is exactly mfixed
        # GMRES iterations from a zero guess (no tolerance-dependent stopping)
        if gh.lu_fixed is None or gh.lu_fixed[1] != G.N:
            gh.lu_fixed = (sla.lu_factor(M), G.N)
        lu = gh.lu_fixed[0]
        P = LinearOperator(M.shape, matvec=lambda v: sla.lu_solve(lu, v))
        a, _ = gmres(M, rhs, M=P, rtol=0.0, atol=0.0, restart=gh.mfixed, maxiter=1)
    else:
        a = sla.lu_solve(sla.lu_factor(M, overwrite_a=True), rhs)
    return np.log(np.maximum(a, gh.afloor)).reshape(G.shape)


def maximal_lapse(G, gh, u):
    """log alpha_M, the lapse that keeps the slices maximal,

        D^2 alpha = alpha (K_ij K^ij + cK K)

    (the cK term makes d_t K = -cK alpha K, so K relaxes back to zero when the
    lapse only follows alpha_M approximately), with alpha = 1 on the outer
    faces (or the current lapse, gh.lbnd) and even parity about the axis and the equator.  Dense collocation
    solve.

    With gh.mgrid = n_c the solve is done on an n_c-point grid of the same
    extent and map and log alpha_M is interpolated back: the target only has
    to be a smooth, nearly maximal function of the state.  The coefficients
    are computed on the fine grid, where the state is resolved, and sampled
    at the coarse nodes (differentiating an interpolated state on the coarse
    grid aliases under-resolved features, e.g. an outgoing pulse, into a
    spurious near-zero target)."""
    co = lapse_coeffs(G, gh, u)
    nc = gh.mgrid
    if not (nc and nc < G.shape[0]):
        return _lapse_solve(G, gh, co)
    Gc = getattr(gh, '_mgrid_G', None)
    if Gc is None or Gc.shape[0] != nc:
        Gc = gh._mgrid_G = Grid(nc, nc if G.zsym else 2 * nc, G.L, G.ax.a, G.az.a, zsym=G.zsym, b=G.ax.b)
    coc = {k: G.ax.interp(Gc.ax.x, _LAP_PAR[k][0]) @ v @ G.az.interp(Gc.az.x, _LAP_PAR[k][1]).T
           for k, v in co.items()}
    return Gc.ax.interp(G.ax.x, 1) @ _lapse_solve(Gc, gh, coc) @ Gc.az.interp(G.az.x, 1).T


# ----------------------------------------------------------------------------
# invariants and monitors
# ----------------------------------------------------------------------------
def riemann_invariant(G, gh, u):
    """Kretschmann scalar R_abcd R^abcd at L = 1 (the rescaled value)."""
    du, aux = gh.rhs(u, full=True)
    g, gi, dg, ddg = aux['g'], aux['gi'], aux['dg'], aux['ddg'].copy()
    ddg[T_, T_] = aux['A']
    Gl = 0.5 * (np.einsum('acbN->cabN', dg) + np.einsum('bcaN->cabN', dg) - dg)
    Gu = np.einsum('cdN,dabN->cabN', gi, Gl)
    R = 0.5 * (np.einsum('bcadN->abcdN', ddg) + np.einsum('adbcN->abcdN', ddg)
               - np.einsum('acbdN->abcdN', ddg) - np.einsum('bdacN->abcdN', ddg))
    R += (np.einsum('ebcN,fadN,efN->abcdN', Gu, Gu, g)
          - np.einsum('ebdN,facN,efN->abcdN', Gu, Gu, g))
    Ru = np.einsum('abcdN,aeN->ebcdN', R, gi)
    Ru = np.einsum('ebcdN,bfN->efcdN', Ru, gi)
    Ru = np.einsum('efcdN,cgN->efgdN', Ru, gi)
    Ru = np.einsum('efgdN,dhN->efghN', Ru, gi)
    return np.einsum('abcdN,abcdN->N', R, Ru).reshape(G.shape)


def _local_h(x):
    """Node spacing seen by each node of a half axis (the first node's
    neighbour across the origin is its mirror image) or of a full one."""
    left = np.concatenate([[2 * x[0] if x[0] > 0 else x[1] - x[0]], np.diff(x)])
    right = np.concatenate([np.diff(x), [x[-1] - x[-2]]])
    return np.minimum(left, right)


def cfl_dt(G, u, cfl=1.2):
    """Time step from the local characteristic speeds |beta^i| + alpha
    sqrt(gamma^ii) over the local node spacing (collapsed-lapse regions and
    strongly mapped centres then do not set the step for the whole grid)."""
    gi = inv_metric(G, u)
    alpha = 1.0 / np.sqrt(-gi[T_, T_])
    hx = _local_h(G.ax.x)[:, None]
    hz = _local_h(G.az.x)[None, :]
    v = []
    for i in (X_, Z_):
        beta = -gi[T_, i] / gi[T_, T_]
        gii = gi[i, i] - gi[T_, i] ** 2 / gi[T_, T_]
        v.append((np.abs(beta) + alpha * np.sqrt(np.maximum(gii, 0.0))).reshape(G.shape))
    return cfl / np.max(v[0] / hx + v[1] / hz)


def outflow_margin(G, u):
    """min over the outer faces of -g^{nn} (positive: spacelike, outflow)."""
    gi = inv_metric(G, u)
    giXX, giZZ = gi[X_, X_].reshape(G.shape), gi[Z_, Z_].reshape(G.shape)
    return -max(giXX[-1].max(), giZZ[:, -1].max(), -np.inf if G.zsym else giZZ[:, 0].max())


def outflow_rate(G, u, m):
    """The smallest zoom rate at which every outer face of state u has
    -g^{nn} >= m with both characteristics leaving.  A reframe keeps alpha
    and gamma^ij and shifts beta^n by -(ell' - ell) X^n, so on a face at
    X^n = s L (s = +-1) the condition is s beta^n + alpha sqrt(gamma^nn + m)
    <= (ell' - ell) L."""
    gi = inv_metric(G, u)
    alpha = 1.0 / np.sqrt(-gi[T_, T_])
    need = []
    for i, faces in ((X_, ((-1, 1),)), (Z_, ((-1, 1),) if G.zsym else ((-1, 1), (0, -1)))):
        beta = (-gi[T_, i] / gi[T_, T_]).reshape(G.shape)
        gnn = (gi[i, i] - gi[T_, i] ** 2 / gi[T_, T_]).reshape(G.shape)
        al = alpha.reshape(G.shape)
        for k, sgn in faces:
            sl = (k, slice(None)) if i == X_ else (slice(None), k)
            need.append(np.max(sgn * beta[sl] + al[sl] * np.sqrt(np.maximum(gnn[sl], 0.0) + m)))
    return float(u[-1]) + max(need) / G.L


# ----------------------------------------------------------------------------
# initial data
# ----------------------------------------------------------------------------
def from_3p1(G, gam, K, alpha, ell=1.0, phi=None, dtphi=None, gh=None):
    """Rescaled GH state at L = 1 from physical 3+1 data on the plane.

    gam, K: dicts with keys 'XX', 'XZ', 'ZZ', 'W' (W = (t_XX - t_YY)/X^2) for
    the 3-metric and extrinsic curvature; alpha the lapse.  The physical
    shift vanishes on the slice, so the rescaled shift is -ell X^i.  Spatial
    derivatives are taken on the grid (parity-exact); the time derivatives
    of lapse and shift are fixed pointwise by C_a = 0, which with the
    constraints makes d_T C_a vanish too.  phi, dtphi: scalar field and its
    physical time derivative."""
    X, Z = G.X, G.Z
    zero = np.zeros_like(X)
    gXX, gXZ, gZZ, gW = gam['XX'], gam.get('XZ', zero), gam['ZZ'], gam['W']
    # rescaled shift beta^i = -ell X^i, lowered with gamma
    bl_X = -ell * (gXX * X + gXZ * Z)
    bl_Z = -ell * (gXZ * X + gZZ * Z)
    b2 = ell ** 2 * (gXX * X ** 2 + 2 * gXZ * X * Z + gZZ * Z ** 2)
    g7 = [-alpha ** 2 + b2, bl_X, bl_Z, gXX, gXZ, gZZ, gW]

    def adv(f, par):            # X d_X f + Z d_Z f
        return X * G.dx(f, par[0]) + Z * G.dz(f, par[1])

    kc = lambda k: K.get(k, zero)
    PXX = -2 * alpha * kc('XX') - ell * adv(gXX, PAR[3])
    PXZ = -2 * alpha * kc('XZ') - ell * adv(gXZ, PAR[4])
    PZZ = -2 * alpha * kc('ZZ') - ell * adv(gZZ, PAR[5])
    # (t_XX - t_YY) = X^2 W  =>  (X d_X + Z d_Z)(X^2 W) = X^2 (2W + adv W)
    PW = -2 * alpha * kc('W') - ell * (2 * gW + adv(gW, PAR[6]))
    if phi is None:
        phi, Pi = zero.copy(), zero.copy()
    else:
        Pi = dtphi - ell * adv(phi, (1, 1))
    gh = gh or GH(G)
    zero3 = [zero, zero, zero]
    return _solve_PT(G, gh, g7, zero3 + [PXX, PXZ, PZZ, PW], phi, Pi, ell)


def scalar_data(G, amp, profile='gauss', sigma=0.5, r0=0.0, ell=1.0, gh=None):
    """Time-symmetric, conformally flat data for a massless scalar field.

    grad^2 psi = -pi psi phi'^2 is integrated outward from psi(0) = 1; the
    physical lapse is 1 and the shift zero on the initial slice."""
    from scipy.integrate import solve_ivp

    def prof(r):
        if profile == 'gauss':
            return amp * np.exp(-(r ** 2 - r0 ** 2) ** 2 / sigma ** 4) if r0 else \
                amp * np.exp(-r ** 2 / sigma ** 2)
        raise ValueError(profile)

    def dprof(r):
        if r0:
            return prof(r) * (-4 * r * (r ** 2 - r0 ** 2) / sigma ** 4)
        return prof(r) * (-2 * r / sigma ** 2)

    def ode(r, y):
        psi, dpsi = y
        if r < 1e-12:
            return [dpsi, -np.pi * psi * dprof(r) ** 2 / 3.0]
        return [dpsi, -2 * dpsi / r - np.pi * psi * dprof(r) ** 2]

    rmax = G.R.max() * 1.01
    sol = solve_ivp(ode, [0, rmax], [1.0, 0.0], rtol=1e-13, atol=1e-15,
                    dense_output=True, method='DOP853')
    psi = sol.sol(G.R.ravel())[0].reshape(G.shape)
    p4 = psi ** 4
    gam = {'XX': p4, 'ZZ': p4, 'W': np.zeros_like(p4)}
    return from_3p1(G, gam, {}, np.ones_like(p4), ell, prof(G.R),
                    np.zeros_like(p4), gh)


# ----------------------------------------------------------------------------
# checks
# ----------------------------------------------------------------------------
def check_ricci():
    """Ricci of an analytic axisymmetric metric: cartoon + tensor algebra
    against an independent evaluation in 3-D Cartesian coordinates (N2).

    sympy supplies g, dg, ddg of the full 4-D metric (no symmetry used);
    Ricci is then formed with explicit loops.  The metric is not a solution,
    so every term of R_ab is exercised."""
    import sympy as sp
    T, x, y, z = sp.symbols('T x y z', real=True)
    xs = [T, x, y, z]
    Xv = [0, x, y, z]
    r2 = x ** 2 + y ** 2 + z ** 2
    e = sp.Rational(1, 10)
    A = e * sp.exp(-r2) * (1 + T / 3)
    B = e * sp.exp(-r2 / 2) * (1 - T / 5) * (1 + z ** 2)
    Cc = e * sp.exp(-r2) * (2 + sp.sin(T))
    Dd = e * sp.exp(-r2) * sp.cos(T) * (1 + z ** 2)
    Ee = e * sp.exp(-r2) * (1 + T * z ** 2)
    g = sp.zeros(4, 4)
    g[0, 0] = -(1 - r2) * (1 + A)
    for i in (1, 2, 3):
        g[0, i] = g[i, 0] = -Xv[i] * (1 + Dd)
        g[i, i] = 1 + Cc
    for i in (1, 2):                     # smooth rho-rho anisotropy x_i x_j
        for j in (1, 2):
            g[i, j] += B * Xv[i] * Xv[j]
    for i in (1, 2):                     # rho-z cross terms x_i z
        g[i, 3] += Ee * Xv[i] * z
        g[3, i] = g[i, 3]
    g[0, 3] += Ee * z                    # extra T-z piece, odd in z
    g[3, 0] = g[0, 3]
    fg = sp.lambdify(xs, g, 'numpy')
    fdg = sp.lambdify(xs, [[sp.diff(g[a, b], xs[c]) for a in range(4) for b in range(4)]
                           for c in range(4)], 'numpy')
    fddg = sp.lambdify(xs, [[[sp.diff(g[a, b], xs[c], xs[d]) for a in range(4) for b in range(4)]
                             for d in range(4)] for c in range(4)], 'numpy')

    def ev(f, pt, shape):
        return np.array([np.broadcast_to(v, ()) for v in np.ravel(np.array(f(*pt), dtype=object))],
                        float).reshape(shape)

    def ricci_at(pt):
        gg = ev(fg, pt, (4, 4))
        d1 = ev(fdg, pt, (4, 4, 4))
        d2 = ev(fddg, pt, (4, 4, 4, 4))
        gi = np.linalg.inv(gg)
        dgi = np.array([-gi @ d1[k] @ gi for k in range(4)])
        Gam = np.zeros((4, 4, 4))
        dGam = np.zeros((4, 4, 4, 4))          # d_e Gamma^a_bc
        for a in range(4):
            for b in range(4):
                for c in range(4):
                    Gam[a, b, c] = sum(0.5 * gi[a, d] * (d1[b, d, c] + d1[c, d, b] - d1[d, b, c])
                                       for d in range(4))
                    for k in range(4):
                        dGam[k, a, b, c] = sum(
                            0.5 * dgi[k, a, d] * (d1[b, d, c] + d1[c, d, b] - d1[d, b, c])
                            + 0.5 * gi[a, d] * (d2[k, b, d, c] + d2[k, c, d, b] - d2[k, d, b, c])
                            for d in range(4))
        R = np.zeros((4, 4))
        for a in range(4):
            for b in range(4):
                R[a, b] = sum(dGam[c, c, a, b] - dGam[b, c, a, c]
                              + sum(Gam[c, c, d] * Gam[d, a, b] - Gam[c, b, d] * Gam[d, a, c]
                                    for d in range(4)) for c in range(4))
        return R

    G = Grid(14, 14, L=2.0, ax=1.0, az=1.0)
    T0 = 0.3
    vals = np.zeros((4, 4) + G.shape)
    valsT = np.zeros((4, 4) + G.shape)
    ddT = np.zeros((4, 4) + G.shape)
    for i in range(G.shape[0]):
        for k in range(G.shape[1]):
            pt = (T0, G.X[i, k], 0.0, G.Z[i, k])
            vals[:, :, i, k] = ev(fg, pt, (4, 4))
            d1 = ev(fdg, pt, (4, 4, 4))
            d2 = ev(fddg, pt, (4, 4, 4, 4))
            valsT[:, :, i, k] = d1[0]
            ddT[:, :, i, k] = d2[0, 0]
    # the analytic metric is PHYSICAL; the state stores ghat = g at L = 1
    # and P with d_T g = P - 2 ell ghat
    ell = 1.0
    g7 = [vals[a, b] for (a, b) in COMPS[:6]] + [(vals[1, 1] - vals[2, 2]) / G.X ** 2]
    gT7 = [valsT[a, b] for (a, b) in COMPS[:6]] + [(valsT[1, 1] - valsT[2, 2]) / G.X ** 2]
    P7 = [gt + 2 * ell * f for gt, f in zip(gT7, g7)]
    gh = GH(G, gamma0=0.0, scalar=False)
    gg, dg, ddg, gi, dgi, Gl, Gu = gh.geometry(g7, P7, ell, ddT=ddT)
    Hz = np.zeros((4, G.N))
    dHz = np.zeros((4, 4, G.N))
    _, R, _ = gh.operator(gg, dg, ddg, gi, dgi, Gl, Gu, Hz, dHz)
    worst, scale = 0.0, 0.0
    for (i, k) in [(0, 0), (3, 5), (7, 2), (10, 11)]:
        Rex = ricci_at((T0, G.X[i, k], 0.0, G.Z[i, k]))
        n = i * G.shape[1] + k
        worst = max(worst, np.max(np.abs(R[:, :, n] - Rex)))
        scale = max(scale, np.max(np.abs(Rex)))
    return worst, scale


def check_kernels(n=16, seed=1):
    """numba operator vs the einsum reference on a perturbed, non-solution
    state, with every gauge term switched on: scalar field (gh_operator),
    and vacuum with the maximal-lapse target (gh_fused)."""
    import warnings
    warnings.simplefilter('ignore')
    G = Grid(n, n, 3.0, 1.5, 1.5, b=0.9)
    rng = np.random.default_rng(seed)
    u = scalar_data(G, 0.25, sigma=0.5, ell=0.8, gh=GH(G))
    F = fields(G, u)
    k = np.exp(-G.R ** 2)
    for i in range(2 * NG):
        px, pz = (PAR + PAR)[i]
        F[i] += 0.05 * k * (G.X ** ((1 - px) // 2)) * (G.Z ** ((1 - pz) // 2)) * rng.standard_normal()
    worst = 0.0
    for muL, muS, pL, qL in ((0, 0, 1.0, 1.0), (0.3, 15, 1.0, 1.0), (0.7, 3, 0.0, 1.0), (1.5, 20, 0.0, -1.0)):
        g0p = -1.0 if qL < 0 else 0.0
        a = GH(G, gamma0=2.0, muL=muL, muS=muS, pL=pL, qL=qL, g0p=g0p, fast=True)
        b = GH(G, gamma0=2.0, muL=muL, muS=muS, pL=pL, qL=qL, g0p=g0p, fast=False)
        da, db = a.rhs(u), b.rhs(u)
        worst = max(worst, np.max(np.abs(da - db)) / np.max(np.abs(db)))
    # vacuum, maximal-lapse driver: the phi slot is the lapse target lA
    F[2 * NG] = -0.2 * k * (1 + 0.3 * G.Z ** 2)
    F[2 * NG + 1] = 0.0
    lamM = -0.15 * k
    res = []
    for fast in (True, False):
        c = GH(G, gamma0=2.0, scalar=False, muL=3.0, muS=15, pL=0.0, qL=-1.0, g0p=-1.0, maximal=(3.0, 1.0),
               fast=fast)
        c.lamM = lamM
        res.append(c.rhs(u))
    return max(worst, np.max(np.abs(res[0] - res[1])) / np.max(np.abs(res[1])))


def check_flat(nx=16, nz=16, L=3.0, ell=1.0):
    G = Grid(nx, nz, L)
    gh = GH(G)
    u = flat_state(G, ell)
    du = gh.rhs(u)
    return np.max(np.abs(du[:-1]))


def main():
    t0 = time.time()
    e = check_flat()
    print(f"flat similarity metric, max|RHS| = {e:.2e}")
    print(f"numba kernels vs einsum reference, rel. max diff = {check_kernels():.1e}")
    w, sc = check_ricci()
    print(f"Ricci vs independent 3-D evaluation, max err = {w:.2e} (|R| up to {sc:.2f})")
    print(f"[{time.time() - t0:.1f} s]")


if __name__ == '__main__':
    main()
