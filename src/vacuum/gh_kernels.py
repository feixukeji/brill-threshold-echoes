"""numba kernels for gh_sim: the pointwise tensor algebra of the GH operator
(CPU; gh_cuda carries the same algebra for the GPU).

Layout: point index first, g[n, a, b], dg[n, c, a, b], ddg[n, c, d, a, b].
The operator uses the reduced form

  R_ab - nabla_(a Gamma_b) = -1/2 g^cd d_c d_d g_ab + Q_ab(g, dg)

(the other second derivatives cancel identically), so only one contraction
with ddg is needed.  gh_sim.GH.operator (einsum, all terms of R_ab) is the
reference it is checked against (`check_kernels`)."""
from __future__ import annotations

import numpy as np
from numba import njit, prange

T_ = 0
NSCR = 408          # scratch doubles per point of _core
CHUNK = 64          # points per parallel chunk (scratch allocated per chunk)


@njit(cache=True, fastmath=False, inline='always')
def _inv4(m, out):
    a = m.ravel()
    inv = out.ravel()          # (out must not alias m)
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
    for i in range(16):
        inv[i] = inv[i] / det


@njit(cache=True, inline='always')
def _core(gg, d1, gi, sbox, Xn, ell, cs, muL, muS, g0n, pL, lrefn, dlrefn, qL, g0p, E, C, Guo, S):
    """The GH operator at one point from g, dg (d1[c, a, b]), g^-1 and
    sbox[a, b] = g^cd d_c d_d g_ab (d_T^2 g taken as zero): fills E[a, b],
    C[a], Guo[c, a, b].  S: scratch of NSCR doubles (views below, so the
    per-point work allocates nothing)."""
    dgi = S[0:64].reshape((4, 4, 4))
    tmp = S[64:80].reshape((4, 4))
    Gl = S[80:144].reshape((4, 4, 4))
    Gu = S[144:208].reshape((4, 4, 4))
    Gam = S[208:212]
    trG = S[212:216]
    H = S[216:220]
    dH = S[220:236].reshape((4, 4))
    F = S[236:240]
    dF = S[240:256].reshape((4, 4))
    dh = S[256:320].reshape((4, 4, 4))
    dgih = S[320:384].reshape((4, 4, 4))
    dalpha = S[384:388]
    bvec = S[388:392]
    db = S[392:408].reshape((4, 4))
    # dgi[c,a,b] = -gi[a,e] gi[b,f] d1[c,e,f]
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
    # Gamma_cab and Gamma^c_ab
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
        for a in range(4):
            for b in range(4):
                Guo[c, a, b] = Gu[c, a, b]
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
    H[:] = 0.0
    dH[:, :] = 0.0                  # dH[c, a] = d_c H_a
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
        # rescaled derivative directions: T uses d1[0] + 2 ell g
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
        alpha = 1.0 / np.sqrt(-gTT)
        for c in range(4):
            dalpha[c] = 0.5 * alpha ** 3 * dgih[c, 0, 0]
        if muL != 0.0:
            # spatial inverse gamma^ij = g^ij - g^Ti g^Tj / g^TT
            ldet = 0.0
            # log sqrt(det gamma) via 3x3 determinant
            m00 = gg[1, 1]; m01 = gg[1, 2]; m02 = gg[1, 3]
            m11 = gg[2, 2]; m12 = gg[2, 3]; m22 = gg[3, 3]
            det3 = (m00 * (m11 * m22 - m12 * m12) - m01 * (m01 * m22 - m12 * m02)
                    + m02 * (m01 * m12 - m11 * m02))
            # target log lapse: pL log sqrt(gamma) + lrefn, d_c lref =
            # dlrefn[c] (lref: the evolved target lA of the maximal-lapse
            # driver, else zero)
            lsq = 0.5 * pL * np.log(det3) + lrefn
            lg = lsq - np.log(alpha)
            # damping rate muL alpha^(qL - 1) (qL = -1: bamps' 0.4/alpha^2)
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
            bvec[:] = 0.0
            db[:, :] = 0.0                   # db[c, i]
            for i in range(1, 4):
                bvec[i] = -gi[0, i] / gTT + ell * Xn[i - 1]
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
    for a in range(4):
        C[a] = H[a] + Gam[a]
    # ---- operator ----
    for a in range(4):
        for b in range(a, 4):
            # -1/2 g^cd d_c d_d g_ab
            val = -0.5 * sbox[a, b]
            # Q_ab
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
        alpha0 = 1.0 / np.sqrt(-gi[0, 0])
        nl0 = -alpha0
        # damping rate gamma0 alpha^g0p per unit proper time (g0p = -1:
        # a constant rate per unit T, as bamps' alpha gamma0 = 1)
        g0 = g0n * alpha0 ** g0p
        nC = 0.0
        for a in range(4):
            nC += gi[a, 0] * nl0 * C[a]
        for a in range(4):
            for b in range(4):
                na = nl0 if a == 0 else 0.0
                nb = nl0 if b == 0 else 0.0
                E[a, b] += g0 * (0.5 * (na * C[b] + C[a] * nb) - 0.5 * gg[a, b] * nC)


@njit(cache=True, parallel=True)
def gh_operator(g, dg, ddg, Xv, ell, cs, muL, muS, gamma0, pL, lref, dlref, qL, g0p):
    """Returns E[n,a,b] (with d_T^2 g taken as zero), C[n,a], gi[n,a,b],
    Gu[n,c,a,b].  Gauge source: underlying harmonic ell g_ab c_b g^{bT}
    plus the rescaled damped-wave terms (see GH.hfunc); d_T of the damped
    terms follows d_T ghat = d_T g + 2 ell g (they have weight 1); gamma0
    the constraint-damping rate."""
    N = g.shape[0]
    E = np.zeros((N, 4, 4))
    Cout = np.zeros((N, 4))
    giout = np.zeros((N, 4, 4))
    Guout = np.zeros((N, 4, 4, 4))
    for n in prange(N):
        gi = np.empty((4, 4))
        _inv4(g[n], gi)
        giout[n] = gi
        dd = ddg[n]
        sbox = np.zeros((4, 4))
        for a in range(4):
            for b in range(4):
                s = 0.0
                for c in range(4):
                    for d in range(4):
                        s += gi[c, d] * dd[c, d, a, b]
                sbox[a, b] = s
        _core(g[n], dg[n], gi, sbox, Xv[n], ell, cs, muL, muS, gamma0, pL, lref[n], dlref[n],
              qL, g0p, E[n], Cout[n], Guout[n], np.empty(NSCR))
    return E, Cout, giout, Guout


@njit(cache=True, inline='always')
def _fill4(out, c, n, yy):
    """Symmetric 4x4 on the plane from the 6 stored components c[0:6, n]
    (TT, TX, TZ, XX, XZ, ZZ) and the YY entry; Y-offdiagonal entries zero."""
    for a in range(4):
        for b in range(4):
            out[a, b] = 0.0
    out[0, 0] = c[0, n]
    out[0, 1] = out[1, 0] = c[1, n]
    out[0, 3] = out[3, 0] = c[2, n]
    out[1, 1] = c[3, n]
    out[1, 3] = out[3, 1] = c[4, n]
    out[3, 3] = c[5, n]
    out[2, 2] = yy


@njit(cache=True, parallel=True)
def gh_fused(F, Fx, Fz, Fxx, Fzz, Fxz, HT, HTx, HTz, Xv, ell, cs, muL, muS, gamma0, pL,
             lref, dlref, qL, g0p, Cout):
    """Physical d_T^2 of the 7 stored components, A7[k, n] (the W entry is
    (A_XX - A_YY) / X^2), from the 7 components on the plane and their
    spectral derivatives in (7, N) layout: F (with the flat part), Fx, Fz,
    Fxx, Fzz, Fxz its X, Z, XX, ZZ, XZ derivatives; HT = d_T g (physical)
    and HTx, HTz its X, Z derivatives.  The 4x4 metric, dg and the Y
    derivatives (cartoon) are assembled pointwise exactly as
    gh_sim.metric_derivs / _cartoon do, then the operator is gh_operator's
    (_core).  Same result as gh_operator without the (N, 4, 4, 4, 4)
    second-derivative array and its transposes.  Cout (4, N) receives the
    constraint C_a = H_a + Gamma_a (pass a (4, 1) array to skip it)."""
    N = F.shape[1]
    A7 = np.zeros((7, N))
    nch = (N + CHUNK - 1) // CHUNK
    for ch in prange(nch):
        _fused_chunk(ch * CHUNK, min(N, (ch + 1) * CHUNK), F, Fx, Fz, Fxx, Fzz, Fxz, HT, HTx, HTz, Xv,
                     ell, cs, muL, muS, gamma0, pL, lref, dlref, qL, g0p, A7, Cout)
    return A7


@njit(cache=True)
def _fused_chunk(lo, hi, F, Fx, Fz, Fxx, Fzz, Fxz, HT, HTx, HTz, Xv, ell, cs, muL, muS, gamma0, pL,
                 lref, dlref, qL, g0p, A7, Cout):
    """Points lo..hi-1 of gh_fused.  One call per chunk, and _core sees only
    chunk-local arrays: numba reference-counts every array passed to a
    function (atomically), and per-point calls on the shared inputs made the
    threads contend."""
    gg = np.empty((4, 4))
    d1 = np.zeros((4, 4, 4))
    t = np.empty((4, 4))
    dxg = np.empty((4, 4))
    gi = np.empty((4, 4))
    sbox = np.empty((4, 4))
    E = np.empty((4, 4))
    C = np.empty(4)
    Gu = np.empty((4, 4, 4))
    S = np.empty(NSCR)
    xl = np.empty(3)
    dl = np.empty(4)
    # chunk-local contiguous copies of the (7, N) inputs: per point the
    # kernel reads 63 rows, which as strided streams defeat the prefetcher
    F = F[:, lo:hi].copy()
    Fx = Fx[:, lo:hi].copy()
    Fz = Fz[:, lo:hi].copy()
    Fxx = Fxx[:, lo:hi].copy()
    Fzz = Fzz[:, lo:hi].copy()
    Fxz = Fxz[:, lo:hi].copy()
    HT = HT[:, lo:hi].copy()
    HTx = HTx[:, lo:hi].copy()
    HTz = HTz[:, lo:hi].copy()
    for n in range(hi - lo):
        ng = lo + n
        X = Xv[ng, 0]
        iX = 1.0 / X
        X2 = X * X
        W, Wx, Wz = F[6, n], Fx[6, n], Fz[6, n]
        WT = HT[6, n]
        _fill4(gg, F, n, F[3, n] - X2 * W)
        d1[2, :, :] = 0.0
        _fill4(t, HT, n, HT[3, n] - X2 * WT)
        d1[0] = t
        _fill4(dxg, Fx, n, Fx[3, n] - 2.0 * X * W - X2 * Wx)
        d1[1] = dxg
        _fill4(t, Fz, n, Fz[3, n] - X2 * Wz)
        d1[3] = t
        d1[2, 0, 2] = d1[2, 2, 0] = gg[0, 1] * iX
        d1[2, 1, 2] = d1[2, 2, 1] = X * W
        d1[2, 3, 2] = d1[2, 2, 3] = gg[3, 1] * iX
        _inv4(gg, gi)
        # sbox = g^cd d_c d_d g_ab over the plane's nonzero g^cd (g^TY =
        # g^XY = g^ZY = 0), d_T^2 g taken as zero
        sbox[:, :] = 0.0
        _fill4(t, HTx, n, HTx[3, n] - 2.0 * X * WT - X2 * HTx[6, n])
        for a in range(4):
            for b in range(4):
                sbox[a, b] += 2.0 * gi[0, 1] * t[a, b]
        _fill4(t, HTz, n, HTz[3, n] - X2 * HTz[6, n])
        for a in range(4):
            for b in range(4):
                sbox[a, b] += 2.0 * gi[0, 3] * t[a, b]
        _fill4(t, Fxx, n, Fxx[3, n] - 2.0 * W - 4.0 * X * Wx - X2 * Fxx[6, n])
        for a in range(4):
            for b in range(4):
                sbox[a, b] += gi[1, 1] * t[a, b]
        _fill4(t, Fzz, n, Fzz[3, n] - X2 * Fzz[6, n])
        for a in range(4):
            for b in range(4):
                sbox[a, b] += gi[3, 3] * t[a, b]
        _fill4(t, Fxz, n, Fxz[3, n] - 2.0 * X * Wz - X2 * Fxz[6, n])
        for a in range(4):
            for b in range(4):
                sbox[a, b] += 2.0 * gi[1, 3] * t[a, b]
        # d_Y d_Y g_ab (cartoon)
        for a in range(4):
            for b in range(4):
                t[a, b] = 0.0
        t[0, 0] = dxg[0, 0] * iX
        t[0, 1] = t[1, 0] = (dxg[0, 1] - gg[0, 1] * iX) * iX
        t[0, 3] = t[3, 0] = dxg[0, 3] * iX
        t[1, 1] = dxg[1, 1] * iX - 2.0 * W
        t[1, 3] = t[3, 1] = (dxg[1, 3] - gg[1, 3] * iX) * iX
        t[3, 3] = dxg[3, 3] * iX
        t[2, 2] = dxg[2, 2] * iX + 2.0 * W
        for a in range(4):
            for b in range(4):
                sbox[a, b] += gi[2, 2] * t[a, b]
        for k in range(3):
            xl[k] = Xv[ng, k]
        for c in range(4):
            dl[c] = dlref[ng, c]
        _core(gg, d1, gi, sbox, xl, ell, cs, muL, muS, gamma0, pL, lref[ng], dl,
              qL, g0p, E, C, Gu, S)
        f = 2.0 / gi[0, 0]
        A7[0, ng] = f * E[0, 0]
        A7[1, ng] = f * E[0, 1]
        A7[2, ng] = f * E[0, 3]
        A7[3, ng] = f * E[1, 1]
        A7[4, ng] = f * E[1, 3]
        A7[5, ng] = f * E[3, 3]
        A7[6, ng] = f * (E[1, 1] - E[2, 2]) / X2
        if Cout.shape[1] > 1:
            for c in range(4):
                Cout[c, ng] = C[c]
