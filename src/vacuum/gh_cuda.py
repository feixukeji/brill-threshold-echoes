#!/usr/bin/env python3
"""The rescaled GH evolution (vacuum, maximal-lapse driver) on a GPU, for a
batch of states at once: gh_sim.GH.step and gh_sim.evolve with the same
fixed steps, filter and maximal-target refreshes.

States are torch tensors (B, NA, nx, nz) of the active fields (the 7 metric
components, their P, and the lapse target lA) -- an active vector v
(gh_sim.active) is v.reshape(NA, nx, nz).  The pointwise algebra is
gh_kernels._core compiled for the device (numba.cuda, one thread per
point); the spectral derivatives, filter and the maximal-lapse solve (coarse
grid, fixed-count left-preconditioned GMRES with a frozen LU, as
gh_sim.maximal_lapse with gh.mfixed) are torch operations.

    E = CudaGH(G, gh);  E.set_ell(ell);  S = E.from_active(V);  E.freeze_lu(S)
    S1 = E.evolve(S, tau, nstep, refresh)

numba.cuda needs libNVVM: jobs/gh_gpu.sh points CUDA_HOME at symlinks into
the pip CUDA wheels.

    python src/vacuum/gh_cuda.py         # GPU vs CPU check (needs a GPU)"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import torch
from numba import cuda, float64

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/
from vacuum import gh_sim as g

NA = g.NA                             # active fields per state
TPB = 128                             # threads per block


def _patch_ptx_version():
    """libNVVM of the pip CUDA 13.3 wheels (CUDA_HOME, see jobs/gh_gpu.sh)
    writes PTX ISA 9.3, and a CUDA 13.0 driver JIT-links at most 9.0 (13.x ->
    9.x); these kernels use nothing newer, so the header is relabelled to
    the driver's version."""
    import re
    from numba.cuda import codegen as cg
    if getattr(cg.CUDACodeLibrary, '_gh_patched', False):
        return
    orig = cg.CUDACodeLibrary.get_asm_str

    def get_asm_str(self, cc=None):
        from numba.cuda.cudadrv.driver import driver
        major, minor = driver.get_version()
        ver = f'{major - 4}.{minor}' if major >= 13 else None
        s = orig(self, cc)
        return re.sub(r'^\.version \d+\.\d+', f'.version {ver}', s, count=1, flags=re.M) if ver else s

    cg.CUDACodeLibrary.get_asm_str = get_asm_str
    cg.CUDACodeLibrary._gh_patched = True


_patch_ptx_version()


@cuda.jit(device=True, inline=True)
def _inv4(m, out):
    a = cuda.local.array(16, float64)
    inv = cuda.local.array(16, float64)
    for i in range(4):
        for j in range(4):
            a[4 * i + j] = m[i, j]
    inv[0] = a[5]*a[10]*a[15] - a[5]*a[11]*a[14] - a[9]*a[6]*a[15] + a[9]*a[7]*a[14] + a[13]*a[6]*a[11] - a[13]*a[7]*a[10]
    inv[4] = -a[4]*a[10]*a[15] + a[4]*a[11]*a[14] + a[8]*a[6]*a[15] - a[8]*a[7]*a[14] - a[12]*a[6]*a[11] + a[12]*a[7]*a[10]
    inv[8] = a[4]*a[9]*a[15] - a[4]*a[11]*a[13] - a[8]*a[5]*a[15] + a[8]*a[7]*a[13] + a[12]*a[5]*a[11] - a[12]*a[7]*a[9]
    inv[12] = -a[4]*a[9]*a[14] + a[4]*a[10]*a[13] + a[8]*a[5]*a[14] - a[8]*a[6]*a[13] - a[12]*a[5]*a[10] + a[12]*a[6]*a[9]
    inv[1] = -a[1]*a[10]*a[15] + a[1]*a[11]*a[14] + a[9]*a[2]*a[15] - a[9]*a[3]*a[14] - a[13]*a[2]*a[11] + a[13]*a[3]*a[10]
    inv[5] = a[0]*a[10]*a[15] - a[0]*a[11]*a[14] - a[8]*a[2]*a[15] + a[8]*a[3]*a[14] + a[12]*a[2]*a[11] - a[12]*a[3]*a[10]
    inv[9] = -a[0]*a[9]*a[15] + a[0]*a[11]*a[13] + a[8]*a[1]*a[15] - a[8]*a[3]*a[13] - a[12]*a[1]*a[11] + a[12]*a[3]*a[9]
    inv[13] = a[0]*a[9]*a[14] - a[0]*a[10]*a[13] - a[8]*a[1]*a[14] + a[8]*a[2]*a[13] + a[12]*a[1]*a[10] - a[12]*a[2]*a[9]
    inv[2] = a[1]*a[6]*a[15] - a[1]*a[7]*a[14] - a[5]*a[2]*a[15] + a[5]*a[3]*a[14] + a[13]*a[2]*a[7] - a[13]*a[3]*a[6]
    inv[6] = -a[0]*a[6]*a[15] + a[0]*a[7]*a[14] + a[4]*a[2]*a[15] - a[4]*a[3]*a[14] - a[12]*a[2]*a[7] + a[12]*a[3]*a[6]
    inv[10] = a[0]*a[5]*a[15] - a[0]*a[7]*a[13] - a[4]*a[1]*a[15] + a[4]*a[3]*a[13] + a[12]*a[1]*a[7] - a[12]*a[3]*a[5]
    inv[14] = -a[0]*a[5]*a[14] + a[0]*a[6]*a[13] + a[4]*a[1]*a[14] - a[4]*a[2]*a[13] - a[12]*a[1]*a[6] + a[12]*a[2]*a[5]
    inv[3] = -a[1]*a[6]*a[11] + a[1]*a[7]*a[10] + a[5]*a[2]*a[11] - a[5]*a[3]*a[10] - a[9]*a[2]*a[7] + a[9]*a[3]*a[6]
    inv[7] = a[0]*a[6]*a[11] - a[0]*a[7]*a[10] - a[4]*a[2]*a[11] + a[4]*a[3]*a[10] + a[8]*a[2]*a[7] - a[8]*a[3]*a[6]
    inv[11] = -a[0]*a[5]*a[11] + a[0]*a[7]*a[9] + a[4]*a[1]*a[11] - a[4]*a[3]*a[9] - a[8]*a[1]*a[7] + a[8]*a[3]*a[5]
    inv[15] = a[0]*a[5]*a[10] - a[0]*a[6]*a[9] - a[4]*a[1]*a[10] + a[4]*a[2]*a[9] + a[8]*a[1]*a[6] - a[8]*a[2]*a[5]
    det = a[0]*inv[0] + a[1]*inv[4] + a[2]*inv[8] + a[3]*inv[12]
    for i in range(4):
        for j in range(4):
            out[i, j] = inv[4 * i + j] / det


@cuda.jit(device=True, inline=True)
def _core(gg, d1, gi, sbox, xl, ell, cs, muL, muS, g0n, pL, lrefn, dlrefn, qL, g0p,
          E, C, Gu, Gl, dgi, H, dH, hmode, Hx, dHx):
    """gh_kernels._core on the device.  The gauge source H, dH (caller's
    arrays) is replaced by Hx, dHx if hmode is 2 (a perturbation evolved
    with the background's gauge source, delta H = 0).  Gu, Gl, dgi are left
    filled for the caller."""
    tmp = cuda.local.array((4, 4), float64)
    Gam = cuda.local.array(4, float64)
    trG = cuda.local.array(4, float64)
    F = cuda.local.array(4, float64)
    dF = cuda.local.array((4, 4), float64)
    dh = cuda.local.array((4, 4, 4), float64)
    dgih = cuda.local.array((4, 4, 4), float64)
    dalpha = cuda.local.array(4, float64)
    bvec = cuda.local.array(4, float64)
    db = cuda.local.array((4, 4), float64)
    for c in range(4):
        for a in range(4):
            for f in range(4):
                s = 0.0
                for e in range(4):
                    s += gi[a, e] * d1[c, e, f]
                tmp[a, f] = s
        for a in range(4):
            for b in range(4):
                s = 0.0
                for f in range(4):
                    s += tmp[a, f] * gi[b, f]
                dgi[c, a, b] = -s
    for c in range(4):
        for a in range(4):
            for b in range(4):
                Gl[c, a, b] = 0.5 * (d1[a, c, b] + d1[b, c, a] - d1[c, a, b])
    for c in range(4):
        for a in range(4):
            for b in range(a, 4):
                s = 0.0
                for d in range(4):
                    s += gi[c, d] * Gl[d, a, b]
                Gu[c, a, b] = s
                Gu[c, b, a] = s
    for c in range(4):
        s = 0.0
        for d in range(4):
            for e in range(4):
                s += gi[d, e] * Gl[c, d, e]
        Gam[c] = s
    for d in range(4):
        s = 0.0
        for c in range(4):
            s += Gu[c, c, d]
        trG[d] = s
    # ---- gauge source H_a and d_c H_a ----
    for a in range(4):
        H[a] = 0.0
        for c in range(4):
            dH[c, a] = 0.0
    for b in range(4):
        F[b] = ell * cs[b] * gi[b, 0]
        for c in range(4):
            dF[c, b] = ell * cs[b] * dgi[c, b, 0]
    for a in range(4):
        s = 0.0
        for b in range(4):
            s += gg[a, b] * F[b]
        H[a] = s
        for c in range(4):
            s = 0.0
            for b in range(4):
                s += d1[c, a, b] * F[b] + gg[a, b] * dF[c, b]
            dH[c, a] = s
    if muL != 0.0 or muS != 0.0:
        for c in range(4):
            for a in range(4):
                for b in range(4):
                    dh[c, a, b] = d1[c, a, b]
                    dgih[c, a, b] = dgi[c, a, b]
        for a in range(4):
            for b in range(4):
                dh[0, a, b] += 2.0 * ell * gg[a, b]
                dgih[0, a, b] -= 2.0 * ell * gi[a, b]
        gTT = gi[0, 0]
        alpha = 1.0 / math.sqrt(-gTT)
        for c in range(4):
            dalpha[c] = 0.5 * alpha ** 3 * dgih[c, 0, 0]
        if muL != 0.0:
            m00 = gg[1, 1]
            m01 = gg[1, 2]
            m02 = gg[1, 3]
            m11 = gg[2, 2]
            m12 = gg[2, 3]
            m22 = gg[3, 3]
            det3 = (m00 * (m11 * m22 - m12 * m12) - m01 * (m01 * m22 - m12 * m02)
                    + m02 * (m01 * m12 - m11 * m02))
            lsq = 0.5 * pL * math.log(det3) + lrefn
            lg = lsq - math.log(alpha)
            aq = alpha ** qL
            H[0] += -muL * aq * lg
            for c in range(4):
                dlsq = 0.0
                for i in range(1, 4):
                    for j in range(1, 4):
                        gij = gi[i, j] - gi[0, i] * gi[0, j] / gTT
                        dlsq += 0.5 * pL * gij * dh[c, i, j]
                dlsq += dlrefn[c]
                dlg = dlsq - dalpha[c] / alpha
                dH[c, 0] += -muL * (qL * aq / alpha * dalpha[c] * lg + aq * dlg)
        if muS != 0.0:
            for i in range(4):
                bvec[i] = 0.0
                for c in range(4):
                    db[c, i] = 0.0
            for i in range(1, 4):
                bvec[i] = -gi[0, i] / gTT + ell * xl[i - 1]
                for c in range(4):
                    db[c, i] = -dgih[c, 0, i] / gTT + gi[0, i] * dgih[c, 0, 0] / gTT ** 2
                db[i, i] += ell
            for a in range(4):
                s = 0.0
                for i in range(1, 4):
                    s += gg[a, i] * bvec[i]
                H[a] += -muS * s / alpha
                for c in range(4):
                    ds = 0.0
                    for i in range(1, 4):
                        ds += dh[c, a, i] * bvec[i] + gg[a, i] * db[c, i]
                    dH[c, a] += -muS * (ds / alpha - s * dalpha[c] / alpha ** 2)
    if hmode == 2:
        for a in range(4):
            H[a] = Hx[a]
            for c in range(4):
                dH[c, a] = dHx[c, a]
    for a in range(4):
        C[a] = H[a] + Gam[a]
    # ---- operator ----
    for a in range(4):
        for b in range(a, 4):
            val = -0.5 * sbox[a, b]
            q = 0.0
            for c in range(4):
                for e in range(4):
                    q += dgi[c, c, e] * Gl[e, a, b]
                    q -= 0.5 * dgi[b, c, e] * d1[a, c, e]
                    q -= 0.5 * (dgi[a, c, e] * Gl[b, c, e] + dgi[b, c, e] * Gl[a, c, e])
            for d in range(4):
                q += trG[d] * Gu[d, a, b]
                for c in range(4):
                    q -= Gu[c, b, d] * Gu[d, a, c]
            val += q
            for c in range(4):
                val += Gu[c, a, b] * (Gam[c] + H[c])
            val -= 0.5 * (dH[a, b] + dH[b, a])
            E[a, b] = val
            E[b, a] = val
    if g0n != 0.0:
        alpha0 = 1.0 / math.sqrt(-gi[0, 0])
        nl0 = -alpha0
        g0 = g0n * alpha0 ** g0p
        nC = 0.0
        for a in range(4):
            nC += gi[a, 0] * nl0 * C[a]
        for a in range(4):
            for b in range(4):
                na = nl0 if a == 0 else 0.0
                nb = nl0 if b == 0 else 0.0
                E[a, b] += g0 * (0.5 * (na * C[b] + C[a] * nb) - 0.5 * gg[a, b] * nC)


@cuda.jit(device=True, inline=True)
def _fill4(out, c, b, n, yy):
    for i in range(4):
        for j in range(4):
            out[i, j] = 0.0
    out[0, 0] = c[b, 0, n]
    out[0, 1] = c[b, 1, n]
    out[1, 0] = c[b, 1, n]
    out[0, 3] = c[b, 2, n]
    out[3, 0] = c[b, 2, n]
    out[1, 1] = c[b, 3, n]
    out[1, 3] = c[b, 4, n]
    out[3, 1] = c[b, 4, n]
    out[3, 3] = c[b, 5, n]
    out[2, 2] = yy


@cuda.jit
def _k_fused(F, Fx, Fz, Fxx, Fzz, Fxz, HT, HTx, HTz, Xv, ell, cs, muL, muS, gamma0, pL,
             lref, dlref, qL, g0p, A7, Cout, Hio, dHio, hmode, b0, nb, LC):
    """gh_kernels.gh_fused for the states b0 .. b0 + nb - 1 of a batch:
    inputs (B, 7, N), lref (B, N), dlref (B, 4, N); A7 (B, 7, N).  Cout
    (B, 4, N) if its last dimension is N (else skipped).  hmode 1: store
    H_a, d_c H_a in Hio (4, N), dHio (4, 4, N); 2: use them.  LC (B, 9, N)
    if its last dimension is N: the maximal-lapse coefficients of
    gh_sim.lapse_coeffs (gXX, gXZ, gZZ, gYY, G3X, G3Z, K, KK) and the lapse."""
    p = cuda.grid(1)
    N = F.shape[2]
    if p >= nb * N:
        return
    bl = p // N
    n = p - bl * N
    b = b0 + bl
    gg = cuda.local.array((4, 4), float64)
    d1 = cuda.local.array((4, 4, 4), float64)
    t = cuda.local.array((4, 4), float64)
    dxg = cuda.local.array((4, 4), float64)
    gi = cuda.local.array((4, 4), float64)
    sbox = cuda.local.array((4, 4), float64)
    E = cuda.local.array((4, 4), float64)
    C = cuda.local.array(4, float64)
    Gu = cuda.local.array((4, 4, 4), float64)
    Gl = cuda.local.array((4, 4, 4), float64)
    dgi = cuda.local.array((4, 4, 4), float64)
    xl = cuda.local.array(3, float64)
    dl = cuda.local.array(4, float64)
    Hl = cuda.local.array(4, float64)
    dHl = cuda.local.array((4, 4), float64)
    Hx = cuda.local.array(4, float64)
    dHx = cuda.local.array((4, 4), float64)
    X = Xv[n, 0]
    iX = 1.0 / X
    X2 = X * X
    W = F[b, 6, n]
    Wx = Fx[b, 6, n]
    Wz = Fz[b, 6, n]
    WT = HT[b, 6, n]
    _fill4(gg, F, b, n, F[b, 3, n] - X2 * W)
    _fill4(t, HT, b, n, HT[b, 3, n] - X2 * WT)
    for i in range(4):
        for j in range(4):
            d1[0, i, j] = t[i, j]
            d1[2, i, j] = 0.0
    _fill4(dxg, Fx, b, n, Fx[b, 3, n] - 2.0 * X * W - X2 * Wx)
    _fill4(t, Fz, b, n, Fz[b, 3, n] - X2 * Wz)
    for i in range(4):
        for j in range(4):
            d1[1, i, j] = dxg[i, j]
            d1[3, i, j] = t[i, j]
    d1[2, 0, 2] = gg[0, 1] * iX
    d1[2, 2, 0] = gg[0, 1] * iX
    d1[2, 1, 2] = X * W
    d1[2, 2, 1] = X * W
    d1[2, 3, 2] = gg[3, 1] * iX
    d1[2, 2, 3] = gg[3, 1] * iX
    _inv4(gg, gi)
    for i in range(4):
        for j in range(4):
            sbox[i, j] = 0.0
    _fill4(t, HTx, b, n, HTx[b, 3, n] - 2.0 * X * WT - X2 * HTx[b, 6, n])
    for i in range(4):
        for j in range(4):
            sbox[i, j] += 2.0 * gi[0, 1] * t[i, j]
    _fill4(t, HTz, b, n, HTz[b, 3, n] - X2 * HTz[b, 6, n])
    for i in range(4):
        for j in range(4):
            sbox[i, j] += 2.0 * gi[0, 3] * t[i, j]
    _fill4(t, Fxx, b, n, Fxx[b, 3, n] - 2.0 * W - 4.0 * X * Wx - X2 * Fxx[b, 6, n])
    for i in range(4):
        for j in range(4):
            sbox[i, j] += gi[1, 1] * t[i, j]
    _fill4(t, Fzz, b, n, Fzz[b, 3, n] - X2 * Fzz[b, 6, n])
    for i in range(4):
        for j in range(4):
            sbox[i, j] += gi[3, 3] * t[i, j]
    _fill4(t, Fxz, b, n, Fxz[b, 3, n] - 2.0 * X * Wz - X2 * Fxz[b, 6, n])
    for i in range(4):
        for j in range(4):
            sbox[i, j] += 2.0 * gi[1, 3] * t[i, j]
    for i in range(4):
        for j in range(4):
            t[i, j] = 0.0
    t[0, 0] = dxg[0, 0] * iX
    t[0, 1] = (dxg[0, 1] - gg[0, 1] * iX) * iX
    t[1, 0] = t[0, 1]
    t[0, 3] = dxg[0, 3] * iX
    t[3, 0] = t[0, 3]
    t[1, 1] = dxg[1, 1] * iX - 2.0 * W
    t[1, 3] = (dxg[1, 3] - gg[1, 3] * iX) * iX
    t[3, 1] = t[1, 3]
    t[3, 3] = dxg[3, 3] * iX
    t[2, 2] = dxg[2, 2] * iX + 2.0 * W
    for i in range(4):
        for j in range(4):
            sbox[i, j] += gi[2, 2] * t[i, j]
    for k in range(3):
        xl[k] = Xv[n, k]
    for c in range(4):
        dl[c] = dlref[b, c, n]
    if hmode == 2:
        for a in range(4):
            Hx[a] = Hio[a, n]
            for c in range(4):
                dHx[c, a] = dHio[c, a, n]
    _core(gg, d1, gi, sbox, xl, ell, cs, muL, muS, gamma0, pL, lref[b, n], dl, qL, g0p,
          E, C, Gu, Gl, dgi, Hl, dHl, hmode, Hx, dHx)
    if hmode == 1:
        for a in range(4):
            Hio[a, n] = Hl[a]
            for c in range(4):
                dHio[c, a, n] = dHl[c, a]
    f = 2.0 / gi[0, 0]
    A7[b, 0, n] = f * E[0, 0]
    A7[b, 1, n] = f * E[0, 1]
    A7[b, 2, n] = f * E[0, 3]
    A7[b, 3, n] = f * E[1, 1]
    A7[b, 4, n] = f * E[1, 3]
    A7[b, 5, n] = f * E[3, 3]
    A7[b, 6, n] = f * (E[1, 1] - E[2, 2]) / X2
    if Cout.shape[2] == N:
        for c in range(4):
            Cout[b, c, n] = C[c]
    if LC.shape[2] == N:
        # gh_sim.lapse_coeffs: gamma^ij, K_ij = -alpha Gamma^T_ij, K, K_ij K^ij,
        # G3^k = gamma^ij 3Gamma^k_ij (3Gamma^k_ij = Gamma^k_ij + beta^k Gamma^T_ij)
        gTT = gi[0, 0]
        alpha = 1.0 / math.sqrt(-gTT)
        gam = cuda.local.array((3, 3), float64)
        Kij = cuda.local.array((3, 3), float64)
        Km = cuda.local.array((3, 3), float64)
        for i in range(3):
            for j in range(3):
                gam[i, j] = gi[i + 1, j + 1] - gi[0, i + 1] * gi[0, j + 1] / gTT
                Kij[i, j] = -alpha * Gu[0, i + 1, j + 1]
        K = 0.0
        for i in range(3):
            for j in range(3):
                s = 0.0
                for k in range(3):
                    s += gam[i, k] * Kij[k, j]
                Km[i, j] = s
            K += Km[i, i]
        KK = 0.0
        for i in range(3):
            for j in range(3):
                KK += Km[i, j] * Km[j, i]
        s0 = 0.0
        for i in range(3):
            for j in range(3):
                s0 += gam[i, j] * Gu[0, i + 1, j + 1]
        LC[b, 0, n] = gam[0, 0]
        LC[b, 1, n] = gam[0, 2]
        LC[b, 2, n] = gam[2, 2]
        LC[b, 3, n] = gam[1, 1]
        for kk in range(2):
            k = 2 * kk          # X (0) and Z (2) of the spatial indices
            s = 0.0
            for i in range(3):
                for j in range(3):
                    s += gam[i, j] * Gu[k + 1, i + 1, j + 1]
            LC[b, 4 + kk, n] = s + (-gi[0, k + 1] / gTT) * s0
        LC[b, 6, n] = K
        LC[b, 7, n] = KK
        LC[b, 8, n] = alpha


def _ca(x):
    return cuda.as_cuda_array(x.detach())


class _Pt:
    """Pointwise stand-in for the grid in gh_sim.to4 / _cartoon / _yy_from."""

    def __init__(self, X, n):
        self.X = np.full(n, float(X))
        self.shape = (n,)


def weyl_EZ(X0, g7, gT7, Dg, DgT, A7):
    """E^Z_Z in the normal frame at one node (X = X0) for B states, from the
    7 components (7, B) of g, d_T g, their X/Z(/XX/ZZ/XZ) derivatives (dicts
    of (7, B)) and A7 = d_T^2 g -- the assembly of gh_sim.metric_derivs and
    gh_sim.riemann_invariant."""
    T_, X_, Y_, Z_ = g.T_, g.X_, g.Y_, g.Z_
    n = g7.shape[1]
    G = _Pt(X0, n)
    X = G.X
    zero = np.zeros(n)
    gg = g.to4(G, list(g7))
    gT = g.to4(G, list(gT7))

    def full(c7d, W, dW, kind):
        t = g.to4(G, list(c7d[:6]) + [zero])
        t[Y_, Y_] = g._yy_from(G, None, W, c7d[3], dW, kind)
        return t

    W, WT = g7[6], gT7[6]
    dxg, dzg = full(Dg['x'], W, Dg['x'][6], 'x'), full(Dg['z'], W, Dg['z'][6], 'z')
    dxgT, dzgT = full(DgT['x'], WT, DgT['x'][6], 'x'), full(DgT['z'], WT, DgT['z'][6], 'z')
    dxxg = g.to4(G, list(Dg['xx'][:6]) + [zero])
    dxxg[Y_, Y_] = Dg['xx'][3] - 2 * W - 4 * X * Dg['x'][6] - X ** 2 * Dg['xx'][6]
    dzzg = g.to4(G, list(Dg['zz'][:6]) + [zero])
    dzzg[Y_, Y_] = Dg['zz'][3] - X ** 2 * Dg['zz'][6]
    dxzg = g.to4(G, list(Dg['xz'][:6]) + [zero])
    dxzg[Y_, Y_] = Dg['xz'][3] - 2 * X * Dg['z'][6] - X ** 2 * Dg['xz'][6]
    dYg, dYYg, dXYg, dZYg = g._cartoon(G, gg, W, dxg, Dg['x'][6], dzg, Dg['z'][6])
    dYgT = g._cartoon(G, gT, WT, dxgT, DgT['x'][6], dzgT, DgT['z'][6])[0]
    dg = np.zeros((4, 4, 4, n))
    dg[T_], dg[X_], dg[Y_], dg[Z_] = gT, dxg, dYg, dzg
    ddg = np.zeros((4, 4, 4, 4, n))
    for (c, d), v in (((T_, T_), g.to4(G, list(A7))), ((T_, X_), dxgT), ((T_, Z_), dzgT), ((T_, Y_), dYgT),
                      ((X_, X_), dxxg), ((Z_, Z_), dzzg), ((X_, Z_), dxzg), ((Y_, Y_), dYYg),
                      ((X_, Y_), dXYg), ((Z_, Y_), dZYg)):
        ddg[c, d] = v
        ddg[d, c] = v
    with np.errstate(all='ignore'):
        gm = np.moveaxis(gg, -1, 0)
        gm = np.where(np.isfinite(gm), gm, 0.0) + np.where(np.all(np.isfinite(gm), axis=(1, 2))[:, None, None], 0, np.eye(4))
        try:
            gi = np.moveaxis(np.linalg.inv(gm), 0, -1)
        except np.linalg.LinAlgError:
            return np.full(n, np.nan)
    Gl = 0.5 * (np.einsum('acbN->cabN', dg) + np.einsum('bcaN->cabN', dg) - dg)
    Gu = np.einsum('cdN,dabN->cabN', gi, Gl)
    R = 0.5 * (np.einsum('bcadN->abcdN', ddg) + np.einsum('adbcN->abcdN', ddg)
               - np.einsum('acbdN->abcdN', ddg) - np.einsum('bdacN->abcdN', ddg))
    R += np.einsum('ebcN,fadN,efN->abcdN', Gu, Gu, gg) - np.einsum('ebdN,facN,efN->abcdN', Gu, Gu, gg)
    out = np.full(n, np.nan)
    for b in range(n):
        if not (np.all(np.isfinite(R[..., b])) and np.all(np.isfinite(gi[:, :, b])) and gi[T_, T_, b] < 0):
            continue                    # a collapsing / failed member: no value
        nl = np.zeros(4)
        nl[T_] = -1.0 / np.sqrt(-gi[T_, T_, b])
        nu = gi[:, :, b] @ nl
        Eb = np.einsum('acbd,c,d->ab', R[..., b], nu, nu)
        # E_ab n^b = 0, so g^{Zc} E_cZ = h^{Zc} E_cZ (the Euclidean pseudo-inverse of h_ab is not h^ab
        # when the shift is nonzero: it gave E^Z_Z / (1 + (beta^Z)^2) on the axis)
        out[b] = (gi[:, :, b] @ Eb)[Z_, Z_]
    return out


def _kret_nodes(X, c):
    """Kretschmann scalar R_abcd R^abcd at n nodes (torch, X (n,) their X):
    c maps 'g', 'gT', 'A' (the 7 components of g, d_T g, d_T^2 g of the
    physical metric at L = 1) and 'x', 'z', 'xx', 'zz', 'xz', 'Tx', 'Tz'
    (their spatial derivatives) to (7, n) -- gh_sim.metric_derivs,
    _cartoon and riemann_invariant on the GPU."""
    T_, X_, Y_, Z_ = g.T_, g.X_, g.Y_, g.Z_
    n = X.shape[0]
    X2, iX = X ** 2, 1.0 / X

    def to4(c7, yy=None):
        t = torch.zeros((4, 4, n), dtype=X.dtype, device=X.device)
        for (a, b), f in zip(g.COMPS[:6], c7[:6]):
            t[a, b] = f
            t[b, a] = f
        t[Y_, Y_] = c7[3] - X2 * c7[6] if yy is None else yy
        return t

    def cartoon(t, W, dxt, dzt, dxW, dzW):
        dY, dYY, dXY, dZY = (torch.zeros_like(t) for _ in range(4))

        def sym(A, a, b, v):
            A[a, b] = v
            A[b, a] = v
        sym(dY, T_, Y_, t[T_, X_] * iX)
        sym(dY, X_, Y_, X * W)
        sym(dY, Z_, Y_, t[Z_, X_] * iX)
        sym(dYY, T_, T_, dxt[T_, T_] * iX)
        sym(dYY, T_, X_, (dxt[T_, X_] - t[T_, X_] * iX) * iX)
        sym(dYY, T_, Z_, dxt[T_, Z_] * iX)
        sym(dYY, X_, X_, dxt[X_, X_] * iX - 2 * W)
        sym(dYY, X_, Z_, (dxt[X_, Z_] - t[X_, Z_] * iX) * iX)
        sym(dYY, Z_, Z_, dxt[Z_, Z_] * iX)
        sym(dYY, Y_, Y_, dxt[Y_, Y_] * iX + 2 * W)
        sym(dXY, T_, Y_, (dxt[T_, X_] - t[T_, X_] * iX) * iX)
        sym(dXY, X_, Y_, W + X * dxW)
        sym(dXY, Z_, Y_, (dxt[Z_, X_] - t[Z_, X_] * iX) * iX)
        sym(dZY, T_, Y_, dzt[T_, X_] * iX)
        sym(dZY, X_, Y_, X * dzW)
        sym(dZY, Z_, Y_, dzt[Z_, X_] * iX)
        return dY, dYY, dXY, dZY

    W, WT = c['g'][6], c['gT'][6]
    gg, gT = to4(c['g']), to4(c['gT'])
    dxg = to4(c['x'], c['x'][3] - 2 * X * W - X2 * c['x'][6])
    dzg = to4(c['z'], c['z'][3] - X2 * c['z'][6])
    dxgT = to4(c['Tx'], c['Tx'][3] - 2 * X * WT - X2 * c['Tx'][6])
    dzgT = to4(c['Tz'], c['Tz'][3] - X2 * c['Tz'][6])
    dxxg = to4(c['xx'], c['xx'][3] - 2 * W - 4 * X * c['x'][6] - X2 * c['xx'][6])
    dzzg = to4(c['zz'], c['zz'][3] - X2 * c['zz'][6])
    dxzg = to4(c['xz'], c['xz'][3] - 2 * X * c['z'][6] - X2 * c['xz'][6])
    dYg, dYYg, dXYg, dZYg = cartoon(gg, W, dxg, dzg, c['x'][6], c['z'][6])
    dYgT = cartoon(gT, WT, dxgT, dzgT, c['Tx'][6], c['Tz'][6])[0]
    dg = torch.stack([gT, dxg, dYg, dzg])          # T_, X_, Y_, Z_ = 0, 1, 2, 3
    ddg = torch.zeros((4, 4, 4, 4, n), dtype=X.dtype, device=X.device)
    for (a, b), v in (((T_, T_), to4(c['A'])), ((T_, X_), dxgT), ((T_, Z_), dzgT), ((T_, Y_), dYgT),
                      ((X_, X_), dxxg), ((Z_, Z_), dzzg), ((X_, Z_), dxzg), ((Y_, Y_), dYYg),
                      ((X_, Y_), dXYg), ((Z_, Y_), dZYg)):
        ddg[a, b] = v
        ddg[b, a] = v
    gi, info = torch.linalg.inv_ex(gg.permute(2, 0, 1))
    gi = gi.permute(1, 2, 0)
    Gl = 0.5 * (torch.einsum('acbN->cabN', dg) + torch.einsum('bcaN->cabN', dg) - dg)
    Gu = torch.einsum('cdN,dabN->cabN', gi, Gl)
    R = 0.5 * (torch.einsum('bcadN->abcdN', ddg) + torch.einsum('adbcN->abcdN', ddg)
               - torch.einsum('acbdN->abcdN', ddg) - torch.einsum('bdacN->abcdN', ddg))
    R = R + torch.einsum('fbcN,fadN->abcdN', Gl, Gu) - torch.einsum('fbdN,facN->abcdN', Gl, Gu)
    Ru = torch.einsum('abcdN,aeN->ebcdN', R, gi)
    Ru = torch.einsum('ebcdN,bfN->efcdN', Ru, gi)
    Ru = torch.einsum('efcdN,cgN->efgdN', Ru, gi)
    Ru = torch.einsum('efgdN,dhN->efghN', Ru, gi)
    I = torch.einsum('abcdN,abcdN->N', R, Ru)
    return torch.where(info == 0, I, torch.full_like(I, float('nan')))


class CudaGH:
    """Batched GPU rhs / step / evolve of gh_sim.GH (see the module docstring)."""

    def __init__(self, G, gh, device='cuda'):
        assert not gh.scalar and gh.maximal
        self.G, self.gh, self.dev = G, gh, device
        t = self.t = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float64, device=device)
        nx, nz = G.shape
        self.shape, self.N = (nx, nz), G.N
        self.ic = int(np.argmin(np.abs(G.az.x)))       # Z node nearest the centre (0 on a half grid)
        self.Dx = {p: t(G.ax.D[p]) for p in (1, -1)}
        self.D2x = {p: t(G.ax.D2[p]) for p in (1, -1)}
        self.DzT = {p: t(G.az.D[p].T) for p in (1, -1)}
        self.D2zT = {p: t(G.az.D2[p].T) for p in (1, -1)}
        self.Fx = {p: t(G.ax.F[p]) for p in (1, -1)}
        self.FzT = {p: t(G.az.F[p].T) for p in (1, -1)}
        self.Xv = t(np.stack([G.X.ravel(), np.zeros(G.N), G.Z.ravel()], axis=1))
        self.cs = t(gh.cs)
        par = g.PAR
        self.zg = {p: torch.tensor([i for i in range(g.NG) if par[i][1] == p], device=device) for p in (1, -1)}
        self.xg = {p: torch.tensor([i for i in range(g.NG) if par[i][0] == p], device=device) for p in (1, -1)}
        fp = g.PAR + g.PAR + [(1, 1)]
        self.fgroups = {}
        for i, pp in enumerate(fp):
            self.fgroups.setdefault(pp, []).append(i)
        self.fgroups = {k: torch.tensor(v, device=device) for k, v in self.fgroups.items()}
        self.kap, self.cK = float(gh.maximal[0]), float(gh.maximal[1])
        self.ell = None
        self._Hio = self._dHio = None
        self.dummy3 = torch.zeros((1, 1, 1), dtype=torch.float64, device=device)
        self.dummyH = torch.zeros((1, 1), dtype=torch.float64, device=device)
        self.dummydH = torch.zeros((1, 1, 1), dtype=torch.float64, device=device)
        self._max_init()

    # -------------------------------------------------------------- state
    def from_active(self, v):
        v = np.asarray(v)
        return self.t(v.reshape(-1, NA, *self.shape))

    def to_active(self, S):
        return S.reshape(S.shape[0], -1).cpu().numpy()

    def set_ell(self, ell):
        if self.ell == ell:
            return
        self.ell = float(ell)
        fl, dfl = g._flat_parts(self.G, self.ell)
        self.fl = self.t(np.array(fl))                                  # (7, nx, nz)
        self.dfl = {k: self.t(np.array(v)) for k, v in dfl.items()}

    # -------------------------------------------------------------- rhs
    def _derivs(self, h, hT):
        """X, Z, XX, ZZ, XZ of h and X, Z of hT, (B, 7, nx, nz) each."""
        D = {k: torch.empty_like(h) for k in ('x', 'z', 'xx', 'zz', 'xz', 'Tx', 'Tz')}
        for pz in (1, -1):
            idx = self.zg[pz]
            D['z'][:, idx] = h[:, idx] @ self.DzT[pz]
            D['zz'][:, idx] = h[:, idx] @ self.D2zT[pz]
            D['Tz'][:, idx] = hT[:, idx] @ self.DzT[pz]
        for px in (1, -1):
            idx = self.xg[px]
            D['x'][:, idx] = self.Dx[px] @ h[:, idx]
            D['xx'][:, idx] = self.D2x[px] @ h[:, idx]
            D['Tx'][:, idx] = self.Dx[px] @ hT[:, idx]
            D['xz'][:, idx] = self.Dx[px] @ D['z'][:, idx]
        return D

    def _pointwise(self, S, lamM, hmode=0, C=False, lc=False):
        """A7 (B, 7, N) and optionally the constraints (B, 4, N) and the lapse
        coefficients (B, 8, N).  hmode: 0 every state its own gauge source;
        'frozen': state 0 stores its H_a, d_c H_a and the others use them."""
        B = S.shape[0]
        N = self.N
        ell = self.ell
        g7, P7, phi = S[:, :g.NG], S[:, g.NG:2 * g.NG], S[:, 2 * g.NG]
        h = g7 - self.fl
        hT = P7 - 2 * ell * h
        D = self._derivs(h, hT)
        for k in ('x', 'z', 'xx', 'zz', 'xz'):
            D[k] += self.dfl[k]
        D['Tx'] -= 2 * ell * self.dfl['x']
        D['Tz'] -= 2 * ell * self.dfl['z']
        dl = torch.zeros((B, 4) + self.shape, dtype=torch.float64, device=self.dev)
        dl[:, 0] = -self.kap * (phi - lamM)
        dl[:, 1] = self.Dx[1] @ phi
        dl[:, 3] = phi @ self.DzT[1]
        r = lambda a: a.reshape(B, -1, N).contiguous()
        A7 = torch.empty((B, g.NG, N), dtype=torch.float64, device=self.dev)
        Co = torch.empty((B, 4, N), dtype=torch.float64, device=self.dev) if C else self.dummy3
        LC = torch.empty((B, 9, N), dtype=torch.float64, device=self.dev) if lc else self.dummy3
        args = [r(g7), r(D['x']), r(D['z']), r(D['xx']), r(D['zz']), r(D['xz']), r(P7 - 2 * ell * g7),
                r(D['Tx']), r(D['Tz'])]
        args = [_ca(a) for a in args]
        gh = self.gh
        common = (_ca(self.Xv), ell, _ca(self.cs), float(gh.muL), float(gh.muS), float(gh.gamma0), float(gh.pL),
                  _ca(r(phi[:, None])[:, 0]), _ca(r(dl)), float(gh.qL), float(gh.g0p), _ca(A7), _ca(Co))
        if hmode == 'frozen':
            if self._Hio is None or self._Hio.shape[1] != N:
                self._Hio = torch.zeros((4, N), dtype=torch.float64, device=self.dev)
                self._dHio = torch.zeros((4, 4, N), dtype=torch.float64, device=self.dev)
            launches = [(1, 0, 1)] + ([(2, 1, B - 1)] if B > 1 else [])
            Hio, dHio = _ca(self._Hio), _ca(self._dHio)
        else:
            launches = [(0, 0, B)]
            Hio, dHio = _ca(self.dummyH), _ca(self.dummydH)
        for hm, b0, nb in launches:
            nt = nb * N
            _k_fused[(nt + TPB - 1) // TPB, TPB](*args, *common, Hio, dHio, hm, b0, nb, _ca(LC))
        return A7, (Co if C else None), (LC if lc else None)

    def rhs(self, S, lamM, hmode=0):
        """d_T of the states S (B, NA, nx, nz), lamM (B or 1, nx, nz) the
        maximal-lapse targets."""
        A7, _, _ = self._pointwise(S, lamM, hmode)
        ell = self.ell
        g7, P7, phi = S[:, :g.NG], S[:, g.NG:2 * g.NG], S[:, 2 * g.NG]
        dP7 = A7.reshape(P7.shape) + 4 * ell * P7 - 4 * ell ** 2 * g7
        dphi = -self.kap * (phi - lamM)
        return torch.cat([P7, dP7, dphi[:, None]], dim=1)

    def constraint(self, S, lamM):
        return self._pointwise(S, lamM, C=True)[1]

    def axis_E(self, S, lamM, iz, alpha=False):
        """E^Z_Z (rescaled, L = 1; physical value / L^2 outside) of every
        state at the axis nodes (X node 0, Z nodes iz), (B, len(iz)).  On the
        axis the Weyl tensor is invariant under the rotations about it, so
        only Psi_2 survives (real: no twist) and E^Z_Z is the same for every
        observer in the axis plane, the slice normal included: a
        gauge-invariant field on the axis surface.  At the centre (iz = 0)
        E = diag(-e/2, -e/2, e) and B = 0.  alpha: also the lapse at the
        centre node (the refresh diagnostic a0)."""
        B, iz = S.shape[0], np.atleast_1d(iz)
        it = torch.as_tensor(iz, device=self.dev)
        A7, _, LC = self._pointwise(S, lamM, lc=alpha)
        A7 = A7[:, :, it]                                                # (B, 7, nz') physical d_T^2 g
        ell = self.ell
        g7, P7 = S[:, :g.NG], S[:, g.NG:2 * g.NG]
        h = g7 - self.fl
        hT = P7 - 2 * ell * h
        D = self._derivs(h, hT)
        ax = lambda a: a[:, :, 0][:, :, it].transpose(0, 1).reshape(g.NG, -1).cpu().numpy()   # (7, B nz')
        Dg = {k: ax(D[k] + self.dfl[k]) for k in ('x', 'z', 'xx', 'zz', 'xz')}
        DgT = {'x': ax(D['Tx'] - 2 * ell * self.dfl['x']), 'z': ax(D['Tz'] - 2 * ell * self.dfl['z'])}
        g70 = ax(g7)
        e = weyl_EZ(self.G.X.ravel()[0], g70, ax(P7) - 2 * ell * g70, Dg, DgT,
                    A7.transpose(0, 1).reshape(g.NG, -1).cpu().numpy())
        if alpha:
            return e.reshape(B, len(iz)), LC[:, 8, self.ic].cpu().numpy()
        return e.reshape(B, len(iz))

    def kretschmann(self, S, lamM, chunk=1 << 16):
        """Kretschmann scalar (rescaled, L = 1; physical value / L^4
        outside) of every state on the whole grid, (B, N) torch."""
        B, N, ell = S.shape[0], self.N, self.ell
        A7 = self._pointwise(S, lamM)[0]
        g7, P7 = S[:, :g.NG], S[:, g.NG:2 * g.NG]
        h = g7 - self.fl
        D = self._derivs(h, P7 - 2 * ell * h)
        r = lambda a: a.reshape(B, g.NG, N)
        c = {'g': r(g7), 'gT': r(P7 - 2 * ell * g7), 'A': A7.reshape(B, g.NG, N)}
        for k in ('x', 'z', 'xx', 'zz', 'xz'):
            c[k] = r(D[k] + self.dfl[k])
        c['Tx'] = r(D['Tx'] - 2 * ell * self.dfl['x'])
        c['Tz'] = r(D['Tz'] - 2 * ell * self.dfl['z'])
        X = self.Xv[:, 0]
        out = torch.empty((B, N), dtype=torch.float64, device=self.dev)
        for b in range(B):
            for i0 in range(0, N, chunk):
                i1 = min(N, i0 + chunk)
                out[b, i0:i1] = _kret_nodes(X[i0:i1], {k: v[b, :, i0:i1] for k, v in c.items()})
        return out

    def central_E(self, S, lamM, alpha=False):
        """E^Z_Z at the node nearest the centre (axis_E at Z node ic): a
        gauge-invariant scalar along the central worldline (alpha: and the
        lapse there)."""
        if alpha:
            e, a0 = self.axis_E(S, lamM, [self.ic], alpha=True)
            return e[:, 0], a0
        return self.axis_E(S, lamM, [self.ic])[:, 0]

    def filter(self, S):
        S = S.clone()
        S[:, :g.NG] -= self.fl
        out = torch.empty_like(S)
        for (px, pz), idx in self.fgroups.items():
            out[:, idx] = self.Fx[px] @ S[:, idx] @ self.FzT[pz]
        out[:, :g.NG] += self.fl
        return out

    def step(self, S, lamM, dt, hmode=0):
        k1 = self.rhs(S, lamM, hmode)
        k2 = self.rhs(S + 0.5 * dt * k1, lamM, hmode)
        k3 = self.rhs(S + 0.5 * dt * k2, lamM, hmode)
        k4 = self.rhs(S + dt * k3, lamM, hmode)
        Sn = S + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        return self.filter(Sn) if self.gh.filt_every else Sn

    def reframe(self, S, lamM, ell_new):
        """gh_sim.reframe on the batch: the same physical states in the chart
        with zoom rate ell_new (d_T' = d_T + (ell - ell_new) X^k d_k).  The
        T-components of ghat and the d_T of the spatial metric change
        algebraically; P_T. follow from C_a = 0 pointwise in the new chart
        (lapse and b = beta + ell X are the same in both charts, so lamM
        stays valid).  Sets self.ell = ell_new and returns the new batch."""
        d = self.ell - float(ell_new)
        X = self.Xv[:, 0].reshape(self.shape)
        Z = self.Xv[:, 2].reshape(self.shape)

        def adv(f, k):
            px, pz = g.PAR[k]
            return X * (self.Dx[px] @ f) + Z * (f @ self.DzT[pz])
        Sn = S.clone()
        gTX, gTZ, gXX, gXZ, gZZ, W = (S[:, k] for k in (1, 2, 3, 4, 5, 6))
        Sn[:, 0] = S[:, 0] + 2 * d * (X * gTX + Z * gTZ) + d * d * (X * X * gXX + 2 * X * Z * gXZ + Z * Z * gZZ)
        Sn[:, 1] = gTX + d * (X * gXX + Z * gXZ)
        Sn[:, 2] = gTZ + d * (X * gXZ + Z * gZZ)
        for k in (3, 4, 5):
            Sn[:, g.NG + k] = S[:, g.NG + k] + d * adv(S[:, k], k)
        Sn[:, g.NG + 6] = S[:, g.NG + 6] + d * (2 * W + adv(W, 6))
        self.set_ell(ell_new)
        return self.solve_PT(Sn, lamM)

    def solve_PT(self, Sn, lamM):
        """Sn with P_TT, P_TX, P_TZ from C_a = 0 pointwise (C is affine in
        them; the other fields as given)."""
        B = Sn.shape[0]
        St = Sn.clone()
        St[:, g.NG:g.NG + 3] = 0.0
        C0 = self.constraint(St, lamM)[:, [0, 1, 3]]
        M = torch.empty((B, 3, 3, self.N), dtype=torch.float64, device=self.dev)
        for k in range(3):
            St[:, g.NG:g.NG + 3] = 0.0
            St[:, g.NG + k] = 1.0
            M[:, :, k] = self.constraint(St, lamM)[:, [0, 1, 3]] - C0
        sol = torch.linalg.solve(M.permute(0, 3, 1, 2), -C0.permute(0, 2, 1)[..., None])[..., 0]   # (B, N, 3)
        Sn[:, g.NG:g.NG + 3] = sol.permute(0, 2, 1).reshape(B, 3, *self.shape)
        return Sn

    # ------------------------------------------------------- maximal lapse
    def _max_init(self):
        G, gh = self.G, self.gh
        nc = gh.mgrid
        assert nc and nc < G.shape[0] and gh.mfixed, 'coarse, fixed-count maximal solve only'
        Gc = g.Grid(nc, nc if G.zsym else 2 * nc, G.L, G.ax.a, G.az.a, zsym=G.zsym, b=G.ax.b)
        self.Gc, self.nc, self.mfixed = Gc, nc, int(gh.mfixed)
        self.ncz = Gc.shape[1]
        t = self.t
        self.Bx = {p: t(G.ax.interp(Gc.ax.x, p)) for p in (1, -1)}
        self.BzT = {p: t(G.az.interp(Gc.az.x, p).T) for p in (1, -1)}
        self.Cx = t(Gc.ax.interp(G.ax.x, 1))
        self.CzT = t(Gc.az.interp(G.az.x, 1).T)
        self.cDx, self.cD2x = t(Gc.ax.D[1]), t(Gc.ax.D2[1])
        self.cDz, self.cD2z = t(Gc.az.D[1]), t(Gc.az.D2[1])
        self.cX = t(Gc.X)
        ncz = self.ncz
        bnd = ((np.arange(nc)[:, None] == nc - 1) | (np.arange(ncz)[None, :] == ncz - 1)
               | ((np.arange(ncz)[None, :] == 0) & (not G.zsym))).ravel()
        self.bnd = torch.tensor(np.flatnonzero(bnd), device=self.dev)
        self.mrhs = torch.zeros(nc * ncz, dtype=torch.float64, device=self.dev)
        self.mrhs[self.bnd] = 1.0
        self.lbnd = getattr(gh, 'lbnd', 'one')      # 'state': alpha_M on the faces = the state's lapse there
        self.afloor = float(gh.afloor)
        self.lu = None
        self.lpar = [g._LAP_PAR[k] for k in ('gXX', 'gXZ', 'gZZ', 'gYY', 'G3X', 'G3Z', 'K', 'KK')]

    def _max_coeffs(self, S):
        """The coefficients of gh_sim.lapse_matrix on the coarse grid, (B, nc, nc)
        each: (a, b, c, e, f, K K + cK K)."""
        B = S.shape[0]
        _, C, LC = self._pointwise(S, torch.zeros_like(S[:, 0]), C=True, lc=True)
        # diagnostics of the states at this refresh (per state): max |C_a|,
        # max K_ij K^ij, min lapse and the lapse at the node nearest the origin
        self.diag = dict(C=C.abs().amax(dim=(1, 2)), KK=LC[:, 7].amax(dim=1), amin=LC[:, 8].amin(dim=1),
                         a0=LC[:, 8, self.ic].clone(), kki=LC[:, 7].argmax(dim=1), ami=LC[:, 8].argmin(dim=1))
        LC = LC.reshape(B, 9, *self.shape)
        co = [self.Bx[px] @ LC[:, k] @ self.BzT[pz] for k, (px, pz) in enumerate(self.lpar)]
        gXX, gXZ, gZZ, gYY, G3X, G3Z, K, KK = co
        if self.lbnd == 'state':
            self._abnd = (self.Bx[1] @ LC[:, 8] @ self.BzT[1]).reshape(B, -1)[:, self.bnd]
        return gXX, 2 * gXZ, gZZ, gYY / self.cX - G3X, G3Z, KK + self.cK * K

    def _max_matrix(self, S):
        """The coarse collocation matrices of gh_sim.maximal_lapse, (B, Nc, Nc)."""
        B = S.shape[0]
        nc = self.nc
        a, bb, c, e, f, dg = self._max_coeffs(S)
        es = torch.einsum
        M4 = es('bik,ij,kl->bikjl', bb, self.cDx, self.cDz)
        # a D2x_ij d_kl + e Dx_ij d_kl on the (k, l) diagonal; c d_ij D2z_kl - f d_ij Dz_kl on (i, j)
        M4.diagonal(dim1=2, dim2=4).add_(es('bik,ij->bijk', a, self.cD2x) + es('bik,ij->bijk', e, self.cDx))
        M4.diagonal(dim1=1, dim2=3).add_(es('bik,kl->bkli', c, self.cD2z) - es('bik,kl->bkli', f, self.cDz))
        n = nc * self.ncz
        M = M4.reshape(B, n, n)
        M.diagonal(dim1=1, dim2=2).sub_(dg.reshape(B, -1))
        M[:, self.bnd, :] = 0.0
        M[:, self.bnd, self.bnd] = 1.0
        return M

    def _max_op(self, S):
        """v -> M v for the matrices of _max_matrix, by the tensor-product
        derivatives (nc^3 per state instead of nc^4)."""
        B = S.shape[0]
        nc = self.nc
        a, bb, c, e, f, dg = self._max_coeffs(S)
        DzT, D2zT = self.cDz.T.contiguous(), self.cD2z.T.contiguous()

        def mv(V):
            v = V.reshape(B, nc, self.ncz)
            vx = self.cDx @ v
            r = (bb * (vx @ DzT) + a * (self.cD2x @ v) + e * vx + c * (v @ D2zT) - f * (v @ DzT)
                 - dg * v).reshape(B, -1)
            r[:, self.bnd] = V[:, self.bnd]
            return r
        return mv

    def freeze_lu(self, S0):
        """The fixed preconditioner: LU of the maximal-lapse operator at the
        states S0 (L, NA, nx, nz) (as gh_sim.maximal_lapse freezes it at its
        first call).  With L > 1 a batch of B states uses LU b % L for state b
        (the batch arranged as B / L groups of L)."""
        # one state at a time: MAGMA's batched LU is meant for small matrices,
        # and one dense 9216^2 operator is 0.7 GB
        LUs, pivs = [], []
        for b in range(S0.shape[0]):
            LU, piv = torch.linalg.lu_factor(self._max_matrix(S0[b:b + 1])[0])
            LUs.append(LU)
            pivs.append(piv)
        self.lu = (torch.stack(LUs), torch.stack(pivs))

    def maximal(self, S):
        """log alpha_M (B, nx, nz): exactly mfixed left-preconditioned GMRES
        iterations from zero (scipy's gmres with restart = mfixed, maxiter = 1)."""
        assert self.lu is not None, 'freeze_lu first'
        mv = self._max_op(S)
        B, n = S.shape[0], self.nc * self.ncz
        LU, piv = self.lu
        if LU.dim() == 2:
            LU, piv = LU[None], piv[None]
        L = LU.shape[0]
        assert B % L == 0

        def ps(V):                      # (B, n) -> (B, n), state b with LU b % L
            W = V.reshape(B // L, L, n).permute(1, 2, 0)
            return torch.linalg.lu_solve(LU, piv, W).permute(2, 0, 1).reshape(B, n)
        m = self.mfixed
        V = torch.zeros((m + 1, B, n), dtype=torch.float64, device=self.dev)
        Hm = torch.zeros((B, m + 1, m), dtype=torch.float64, device=self.dev)
        if self.lbnd == 'state':
            r0 = torch.zeros((B, n), dtype=torch.float64, device=self.dev)
            r0[:, self.bnd] = self._abnd
            v0 = ps(r0)
        else:
            v0 = ps(self.mrhs.expand(B, n))
        beta = torch.linalg.norm(v0, dim=1)
        V[0] = v0 / beta[:, None]
        for j in range(m):
            w = ps(mv(V[j]))
            for k in range(j + 1):
                h = (V[k] * w).sum(1)
                Hm[:, k, j] = h
                w = w - h[:, None] * V[k]
            h1 = torch.linalg.norm(w, dim=1)
            Hm[:, j + 1, j] = h1
            V[j + 1] = w / h1[:, None]
        rhs = torch.zeros((B, m + 1, 1), dtype=torch.float64, device=self.dev)
        rhs[:, 0, 0] = beta
        y = torch.linalg.lstsq(Hm, rhs).solution[:, :, 0]
        a = torch.einsum('kbn,bk->bn', V[:m], y)
        lc = torch.log(torch.clamp(a, min=self.afloor)).reshape(B, self.nc, self.ncz)
        return self.Cx @ lc @ self.CzT

    def cfl_dt(self, S, cfl=1.2):
        """gh_sim.cfl_dt for every state of the batch, (B,): cfl over the
        largest (|beta^i| + alpha sqrt(gamma^ii)) / h_i summed over X, Z."""
        if not hasattr(self, '_hx'):
            self._hx = self.t(g._local_h(self.G.ax.x))[:, None]
            self._hz = self.t(g._local_h(self.G.az.x))[None, :]
        g7 = S[:, :g.NG]
        B = S.shape[0]
        m3 = torch.empty((B,) + self.shape + (3, 3), dtype=torch.float64, device=self.dev)
        # the (T, X, Z) block; g^YY decouples (g_TY = g_XY = g_ZY = 0 on the plane)
        for (i, j), k in (((0, 0), 0), ((0, 1), 1), ((0, 2), 2), ((1, 1), 3), ((1, 2), 4), ((2, 2), 5)):
            m3[..., i, j] = g7[:, k]
            m3[..., j, i] = g7[:, k]
        gi = torch.linalg.inv(m3)
        gTT = gi[..., 0, 0]
        alpha = 1.0 / torch.sqrt(-gTT)
        rate = 0.0
        for i, h in ((1, self._hx), (2, self._hz)):
            beta = -gi[..., 0, i] / gTT
            gii = gi[..., i, i] - gi[..., 0, i] ** 2 / gTT
            rate = rate + (beta.abs() + alpha * torch.sqrt(torch.clamp(gii, min=0.0))) / h
        return cfl / rate.amax(dim=(1, 2))

    # -------------------------------------------------------------- maps
    def evolve(self, S, tau, nstep, refresh):
        """gh_sim.evolve for every state of the batch (each its own maximal
        target, re-solved every `refresh` steps)."""
        dt = tau / nstep
        lam = None
        for k in range(nstep):
            if k % refresh == 0:
                lam = self.maximal(S)
            S = self.step(S, lam, dt)
        return S

    def evolve_frozen(self, S, tau, nstep, refresh):
        """As evolve, but state 0 is the background and every stage of the
        others uses its gauge source H_a, d_c H_a and maximal target (delta H
        = 0: perturbations S[b] - S[0] in the gauge of prescribed source
        functions, whose pure-gauge part obeys the wave equation instead of
        the state-dependent drivers)."""
        dt = tau / nstep
        lam = None
        for k in range(nstep):
            if k % refresh == 0:
                lam = self.maximal(S[:1])
            S = self.step(S, lam, dt, hmode='frozen')
        return S


def check(n=64, tau=0.2, nstep=20, refresh=5):
    """The GPU map against the CPU reference (gh_sim) on near-critical Brill
    data in the production gauge: maximal target, rhs, constraints, reframe
    and a short evolution; and the frozen-gauge evolution of copies of the
    background, which must reproduce the plain one."""
    import warnings
    warnings.simplefilter('ignore')
    from vacuum import gh_data as gd
    cfg = dict(n=n, nz=n, L=24.0, ax=6.33, az=5.37, b=0.995, gamma0=20.0, muL=3.0, muS=15.0, pL=0.0,
               qL=-1.0, g0p=-1.0, maximal=3.0, cK=1.0, mgrid=n // 2, mfixed=40)
    G, gh = g.make(cfg)
    u, _ = gd.brill_data(G, -3.5, 0.4, ell=0.0965, gh=gh, nr=48, nmu=20)
    rel = lambda a, b: float(np.max(np.abs(a - b)) / np.max(np.abs(b)))
    gh.lamM = g.maximal_lapse(G, gh, u)                         # (freezes the LU at u)
    E = CudaGH(G, gh)
    E.set_ell(float(u[-1]))
    S = E.from_active(g.active(G, u)[None])
    E.freeze_lu(S)
    lam = E.maximal(S)
    out = dict(maximal=rel(lam[0].cpu().numpy(), gh.lamM))
    out['rhs'] = rel(E.to_active(E.rhs(S, lam))[0], g.fields(G, gh.rhs(u))[:NA].ravel())
    out['constraint'] = float((E.constraint(S, lam)[0].cpu().numpy() - gh.constraint(u)).__abs__().max())
    out['reframe'] = rel(E.to_active(E.reframe(S, lam, 0.15))[0], g.active(G, g.reframe(G, gh, u, 0.15)))
    E.set_ell(float(u[-1]))
    S1 = E.evolve(S, tau, nstep, refresh)
    out['evolve'] = rel(E.to_active(S1)[0], g.active(G, g.evolve(G, gh, u, tau, nstep, refresh)))
    S2 = E.evolve_frozen(torch.cat([S, S, S]), tau, nstep, refresh)
    out['frozen'] = float((S2 - S1).abs().max() / S1.abs().max())
    lam1 = E.maximal(S1)
    u1 = g.full_state(G, E.to_active(S1)[0], E.ell)
    gh.lamM = lam1[0].cpu().numpy()
    I = E.kretschmann(S1, lam1)[0].cpu().numpy().reshape(G.shape)
    out['Kretschmann'] = rel(I, g.riemann_invariant(G, gh, u1))
    out['I vs 12e^2'] = abs(I[0, E.ic] / (12 * float(E.central_E(S1, lam1)[0]) ** 2) - 1)
    return out


if __name__ == '__main__':
    for k, v in check().items():
        print(f'GPU vs CPU {k:10s} rel. max diff {v:.1e}' if k != 'constraint' else
              f'GPU vs CPU {k:10s} max diff {v:.1e}')
