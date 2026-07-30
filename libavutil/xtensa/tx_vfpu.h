/*
 * HiFi4 VFPU packed-complex primitives for FFmpeg's split-radix FFT / MDCT.
 *
 * This file is part of FFmpeg (SOF Xtensa VFPU acceleration).
 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 */

/*
 * A TXComplex {re, im} is exactly one HiFi4 xtfloatx2 pair (memory word0 = re,
 * word1 = im). The split-radix butterflies become packed 2-wide add/sub plus a
 * lane-swap complex multiply, using the value-returning XT_*_SX2 builtins added
 * to our LLVM backend (mul/add/sub/madd/msub) + AE_SEL32_LH. LLVM clang has no
 * native xtfloatx2 memory op, so pairs move as ae_int32x2 raw bits and the
 * .sx2 builtins reinterpret them as v2f32 in registers (same idiom as
 * float_dsp_init.c). The lane convention here is validated bit-exact against
 * the scalar CMUL/BUTTERFLIES reference and on-silicon by
 * vector_fmul_window_xtensa.
 *
 * MUST be included AFTER tx_priv.h (needs TXComplex / TXSample) and only takes
 * effect for the float instantiation (TX_FLOAT) on the HiFi4-VFPU LLVM build.
 */

#ifndef AVUTIL_XTENSA_TX_VFPU_H
#define AVUTIL_XTENSA_TX_VFPU_H

#if defined(__has_include)
#  if __has_include(<xtensa/config/core-isa.h>)
#    include <xtensa/config/core-isa.h>
#  endif
#endif

#if !defined(__XCC__) && defined(__clang__) && defined(__has_builtin) &&        \
    __has_builtin(__builtin_xtensa_mul_sx2) &&                                  \
    defined(XCHAL_HAVE_HIFI4_VFPU) && XCHAL_HAVE_HIFI4_VFPU &&                  \
    !(defined(XCHAL_HAVE_HIFI5_VFPU) && XCHAL_HAVE_HIFI5_VFPU) &&               \
    defined(__has_include) && __has_include(<xtensahifiintrin.h>) &&            \
    defined(TX_FLOAT)

#include <xtensahifiintrin.h>
#include "libavutil/attributes.h"

#define FF_TX_XTENSA_VFPU 1

typedef ae_int32x2 ffcx;                 /* packed complex (re,im), raw bits */

/* Build a 2-lane pair {lo, hi} (word0=lo, word1=hi). Fixed stack address =>
 * store-to-load forwarding keeps it in L1. */
static av_always_inline ffcx ff_cx_set(float lo, float hi)
{
    float __attribute__((aligned(8))) mm[2] = { lo, hi };
    return AE_L32X2_I((const ae_int32x2 *)mm, 0);
}

static av_always_inline ffcx ff_cx_load(const TXComplex *z)
{
    return AE_L32X2_I((const ae_int32x2 *)z, 0);
}

static av_always_inline void ff_cx_store(TXComplex *z, ffcx v)
{
    AE_S32X2_I(v, (ae_int32x2 *)z, 0);
}

#define FF_CADD(a, b)  XT_ADD_SX2((a), (b))
#define FF_CSUB(a, b)  XT_SUB_SX2((a), (b))
#define FF_CSWAP(a)    AE_SEL32_LH((a), (a))   /* (re,im) -> (im,re) */
#define FF_NEG_LO      ff_cx_set(1.0f, -1.0f)  /* (+1,-1) : y = (x.im, -x.re) */

/* a (x) w,  w = (wre, wim):  (re*wre - im*wim,  im*wre + re*wim) */
static av_always_inline ffcx ff_cmul(ffcx a, float wre, float wim)
{
    ffcx WRE  = ff_cx_set(wre, wre);
    ffcx WMIX = ff_cx_set(-wim, wim);
    return XT_MADD_SX2(XT_MUL_SX2(a, WRE), FF_CSWAP(a), WMIX);
}

/* a (x) conj(w),  conj(w) = (wre, -wim):  (re*wre + im*wim,  im*wre - re*wim) */
static av_always_inline ffcx ff_cmul_conj(ffcx a, float wre, float wim)
{
    ffcx WRE  = ff_cx_set(wre, wre);
    ffcx WMIX = ff_cx_set(wim, -wim);
    return XT_MADD_SX2(XT_MUL_SX2(a, WRE), FF_CSWAP(a), WMIX);
}

/* radix-4 BUTTERFLIES core: given a0,a1 and the two CMUL results t,u, write
 *   z[a0] = a0 + (t+u)      z[a2] = a0 - (t+u)
 *   z[a1] = a1 + r          z[a3] = a1 - r     where r = (d.im, -d.re), d = t-u
 */
static av_always_inline void ff_tx_vfpu_bfly(TXComplex *z,
                                             int a0, int a1, int a2, int a3,
                                             ffcx A0, ffcx A1, ffcx t, ffcx u)
{
    ffcx sum = FF_CADD(t, u);
    ffcx sd  = FF_CSWAP(FF_CSUB(t, u));      /* (d.im, d.re) */
    ff_cx_store(z + a0, FF_CADD(A0, sum));
    ff_cx_store(z + a2, FF_CSUB(A0, sum));
    ff_cx_store(z + a1, XT_MADD_SX2(A1, sd, FF_NEG_LO));   /* A1 + (d.im,-d.re) */
    ff_cx_store(z + a3, XT_MSUB_SX2(A1, sd, FF_NEG_LO));   /* A1 - (d.im,-d.re) */
}

/* TRANSFORM + BUTTERFLIES on z[a0..a3] with twiddle w=(wre,wim). */
static av_always_inline void ff_tx_vfpu_group4(TXComplex *z,
                                               int a0, int a1, int a2, int a3,
                                               float wre, float wim)
{
    ffcx t = ff_cmul_conj(ff_cx_load(z + a2), wre, wim);   /* a2 (x) conj(w) */
    ffcx u = ff_cmul     (ff_cx_load(z + a3), wre, wim);   /* a3 (x) w       */
    ff_tx_vfpu_bfly(z, a0, a1, a2, a3,
                    ff_cx_load(z + a0), ff_cx_load(z + a1), t, u);
}

static av_always_inline void ff_tx_vfpu_fft2(TXComplex *dst, TXComplex *src)
{
    ffcx s0 = ff_cx_load(src + 0), s1 = ff_cx_load(src + 1);
    ff_cx_store(dst + 0, FF_CADD(s0, s1));
    ff_cx_store(dst + 1, FF_CSUB(s0, s1));
}

static av_always_inline void ff_tx_vfpu_fft4(TXComplex *dst, TXComplex *src)
{
    ffcx s0 = ff_cx_load(src + 0), s1 = ff_cx_load(src + 1);
    ffcx s2 = ff_cx_load(src + 2), s3 = ff_cx_load(src + 3);
    ffcx a = FF_CADD(s0, s1), b = FF_CSUB(s0, s1);
    ffcx c = FF_CADD(s2, s3), d = FF_CSUB(s2, s3);
    ffcx dr = XT_MUL_SX2(FF_CSWAP(d), FF_NEG_LO);          /* (d.im,-d.re)=-i*d */
    ff_cx_store(dst + 0, FF_CADD(a, c));
    ff_cx_store(dst + 2, FF_CSUB(a, c));
    ff_cx_store(dst + 1, FF_CADD(b, dr));
    ff_cx_store(dst + 3, FF_CSUB(b, dr));
}

static av_always_inline void ff_tx_vfpu_fft8(TXComplex *dst, TXComplex *src,
                                             TXSample cos)
{
    ff_tx_vfpu_fft4(dst, src);
    ffcx S4 = ff_cx_load(src + 4), S5 = ff_cx_load(src + 5);
    ffcx S6 = ff_cx_load(src + 6), S7 = ff_cx_load(src + 7);
    ffcx t = FF_CADD(S4, S5); ff_cx_store(dst + 5, FF_CSUB(S4, S5));
    ffcx u = FF_CADD(S6, S7); ff_cx_store(dst + 7, FF_CSUB(S6, S7));
    ff_tx_vfpu_bfly(dst, 0, 2, 4, 6,
                    ff_cx_load(dst + 0), ff_cx_load(dst + 2), t, u);
    ff_tx_vfpu_group4(dst, 1, 3, 5, 7, cos, cos);
}

static av_always_inline void ff_tx_vfpu_fft16(TXComplex *dst, TXComplex *src,
                                              TXSample cos8, const TXSample *cos)
{
    ff_tx_vfpu_fft8(dst +  0, src +  0, cos8);
    ff_tx_vfpu_fft4(dst +  8, src +  8);
    ff_tx_vfpu_fft4(dst + 12, src + 12);
    ffcx t = ff_cx_load(dst + 8), u = ff_cx_load(dst + 12);
    ff_tx_vfpu_bfly(dst, 0, 4, 8, 12,
                    ff_cx_load(dst + 0), ff_cx_load(dst + 4), t, u);
    ff_tx_vfpu_group4(dst, 2, 6, 10, 14, cos[2], cos[2]);
    ff_tx_vfpu_group4(dst, 1, 5,  9, 13, cos[1], cos[3]);
    ff_tx_vfpu_group4(dst, 3, 7, 11, 15, cos[3], cos[1]);
}

/* z[0..8*len-1], cos[0..2*len-1]; wim = cos + 2*len - 7 (descending). */
static av_always_inline void ff_tx_vfpu_sr_combine(TXComplex *z,
                                                   const TXSample *cos, int len)
{
    int o1 = 2 * len, o2 = 4 * len, o3 = 6 * len;
    const TXSample *wim = cos + o1 - 7;

    for (int i = 0; i < len; i += 4) {
        ff_tx_vfpu_group4(z, 0, o1 + 0, o2 + 0, o3 + 0, cos[0], wim[7]);
        ff_tx_vfpu_group4(z, 2, o1 + 2, o2 + 2, o3 + 2, cos[2], wim[5]);
        ff_tx_vfpu_group4(z, 4, o1 + 4, o2 + 4, o3 + 4, cos[4], wim[3]);
        ff_tx_vfpu_group4(z, 6, o1 + 6, o2 + 6, o3 + 6, cos[6], wim[1]);
        ff_tx_vfpu_group4(z, 1, o1 + 1, o2 + 1, o3 + 1, cos[1], wim[6]);
        ff_tx_vfpu_group4(z, 3, o1 + 3, o2 + 3, o3 + 3, cos[3], wim[4]);
        ff_tx_vfpu_group4(z, 5, o1 + 5, o2 + 5, o3 + 5, cos[5], wim[2]);
        ff_tx_vfpu_group4(z, 7, o1 + 7, o2 + 7, o3 + 7, cos[7], wim[0]);
        z   += 8;
        cos += 8;
        wim -= 8;
    }
}

#endif /* HiFi4 VFPU + TX_FLOAT */
#endif /* AVUTIL_XTENSA_TX_VFPU_H */
