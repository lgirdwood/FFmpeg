/*
 * Xtensa HiFi optimised float DSP functions
 * Copyright (c) 2026 Intel Corporation
 *
 * This file is part of FFmpeg.
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
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with FFmpeg; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

/*
 * These kernels back the AVFloatDSPContext used for AAC/MP3/Opus windowing and
 * overlap-add (see the SOF ffmpeg_dec HIFI.md analysis). They come in two SIMD
 * flavours, selected by the toolchain:
 *
 *  - Cadence XCC / xt-clang: the native single-precision float SIMD
 *    (xtfloatx2, 2 lanes) via XT_MUL_SX2 / XT_MADD_SX2, on any HiFi core with
 *    the float VFPU option (ace30/ptl HiFi4, ace40 HiFi5).
 *
 *  - Upstream LLVM Xtensa clang: the HiFi5 4-wide packed-float ops
 *    (XT_MUL_SX2X2 / XT_MADD_SX2X2, a v2f32 register pair, dual output) added
 *    to our backend. LLVM clang has no xtfloatx2 memory ops, so packed values
 *    are moved as ae_int32x2 (AE_L32X2_IP / AE_S32X2_IP) and reinterpreted in
 *    registers by the .sx2x2 builtins. Needs -mcpu=intel_ace40 (HiFi5 VFPU).
 *
 * Under the Zephyr-SDK GCC cross-build neither the intrinsics header nor the
 * core config is on the include path, so both FF_XTENSA_HIFI_FLOAT* are 0, this
 * file adds nothing, and float_dsp.c keeps its portable scalar C kernels.
 * Correctness is identical in every case.
 */

#include <stdint.h>

#include "libavutil/attributes.h"
#include "libavutil/float_dsp.h"

#if defined(__has_include)
#  if __has_include(<xtensa/config/core-isa.h>)
#    include <xtensa/config/core-isa.h>
#  endif
#endif

/*
 * Path selection, most specific first:
 *  1. XCC HiFi5/HiFi4 VFPU -> native 2-lane xtfloatx2 (FF_XTENSA_HIFI_FLOAT).
 *  2. LLVM Xtensa clang with the HiFi5 .sx2x2 builtins AND an ace40 core config
 *     -> 4-wide packed float (FF_XTENSA_HIFI_FLOAT_LLVM).
 *  3. anything else -> scalar C fallback.
 *
 * The __XCC__ guard keeps the LLVM clang out of the Cadence branch (its
 * xt_hifiN.h is a fixed-point ae_* wrapper with no xtfloatx2), and the
 * __has_builtin guard keeps GCC out of the LLVM branch.
 */
#if defined(__XCC__) && defined(__has_include) && \
    defined(XCHAL_HAVE_HIFI5) && XCHAL_HAVE_HIFI5 && \
    defined(XCHAL_HAVE_HIFI5_VFPU) && XCHAL_HAVE_HIFI5_VFPU && \
    __has_include(<xtensa/tie/xt_hifi5.h>)
#  include <xtensa/tie/xt_hifi5.h>
#  define FF_XTENSA_HIFI_FLOAT 1
#elif defined(__XCC__) && defined(__has_include) && \
    defined(XCHAL_HAVE_HIFI4) && XCHAL_HAVE_HIFI4 && \
    defined(XCHAL_HAVE_HIFI4_VFPU) && XCHAL_HAVE_HIFI4_VFPU && \
    __has_include(<xtensa/tie/xt_hifi4.h>)
#  include <xtensa/tie/xt_hifi4.h>
#  define FF_XTENSA_HIFI_FLOAT 1
#elif !defined(__XCC__) && defined(__has_builtin) && \
    __has_builtin(__builtin_xtensa_mul_sx2x2) && \
    defined(XCHAL_HAVE_HIFI5_VFPU) && XCHAL_HAVE_HIFI5_VFPU && \
    defined(__has_include) && __has_include(<xtensahifiintrin.h>)
#  include <xtensahifiintrin.h>
#  define FF_XTENSA_HIFI_FLOAT_LLVM 1
#elif !defined(__XCC__) && defined(__has_builtin) && \
    __has_builtin(__builtin_xtensa_mul_sx2) && \
    defined(XCHAL_HAVE_HIFI4_VFPU) && XCHAL_HAVE_HIFI4_VFPU && \
    defined(__has_include) && __has_include(<xtensahifiintrin.h>)
#  include <xtensahifiintrin.h>
#  define FF_XTENSA_HIFI_FLOAT_LLVM4 1
#endif

/* ------------------------------------------------------------------------- */
#if defined(FF_XTENSA_HIFI_FLOAT)   /* Cadence XCC: native 2-lane xtfloatx2   */
/* ------------------------------------------------------------------------- */

#define FF_XT_V (int)sizeof(xtfloatx2)   /* 8: two floats per SIMD vector */

/* Broadcast a scalar to both lanes via an 8-byte-aligned pair load. */
static av_always_inline xtfloatx2 ff_xt_splat(float mul)
{
    float __attribute__((aligned(8))) mm[2] = { mul, mul };
    return XT_LSX2I((const xtfloatx2 *)mm, 0);
}

/* dst[i] = src0[i] * src1[i] */
static void vector_fmul_xtensa(float *dst, const float *src0,
                               const float *src1, int len)
{
    const xtfloatx2 *p0 = (const xtfloatx2 *)src0;
    const xtfloatx2 *p1 = (const xtfloatx2 *)src1;
    xtfloatx2 *pd = (xtfloatx2 *)dst;
    xtfloatx2 a, b;
    int i;

    for (i = 0; i < (len >> 1); i++) {
        XT_LSX2IP(a, p0, FF_XT_V);
        XT_LSX2IP(b, p1, FF_XT_V);
        XT_SSX2IP(XT_MUL_SX2(a, b), pd, FF_XT_V);
    }
    if (len & 1)
        dst[len - 1] = src0[len - 1] * src1[len - 1];
}

/* dst[i] = src[i] * mul */
static void vector_fmul_scalar_xtensa(float *dst, const float *src,
                                      float mul, int len)
{
    const xtfloatx2 *ps = (const xtfloatx2 *)src;
    xtfloatx2 *pd = (xtfloatx2 *)dst;
    xtfloatx2 vmul = ff_xt_splat(mul);
    xtfloatx2 a;
    int i;

    for (i = 0; i < (len >> 1); i++) {
        XT_LSX2IP(a, ps, FF_XT_V);
        XT_SSX2IP(XT_MUL_SX2(a, vmul), pd, FF_XT_V);
    }
    if (len & 1)
        dst[len - 1] = src[len - 1] * mul;
}

/* dst[i] += src[i] * mul */
static void vector_fmac_scalar_xtensa(float *dst, const float *src,
                                      float mul, int len)
{
    const xtfloatx2 *ps = (const xtfloatx2 *)src;
    const xtfloatx2 *pdl = (const xtfloatx2 *)dst;
    xtfloatx2 *pds = (xtfloatx2 *)dst;
    xtfloatx2 vmul = ff_xt_splat(mul);
    xtfloatx2 a, d;
    int i;

    for (i = 0; i < (len >> 1); i++) {
        XT_LSX2IP(a, ps, FF_XT_V);
        XT_LSX2IP(d, pdl, FF_XT_V);
        XT_MADD_SX2(d, a, vmul);          /* d += a * vmul */
        XT_SSX2IP(d, pds, FF_XT_V);
    }
    if (len & 1)
        dst[len - 1] += src[len - 1] * mul;
}

/* dst[i] = src0[i] * src1[i] + src2[i] */
static void vector_fmul_add_xtensa(float *dst, const float *src0,
                                   const float *src1, const float *src2,
                                   int len)
{
    const xtfloatx2 *p0 = (const xtfloatx2 *)src0;
    const xtfloatx2 *p1 = (const xtfloatx2 *)src1;
    const xtfloatx2 *p2 = (const xtfloatx2 *)src2;
    xtfloatx2 *pd = (xtfloatx2 *)dst;
    xtfloatx2 a, b, c;
    int i;

    for (i = 0; i < (len >> 1); i++) {
        XT_LSX2IP(a, p0, FF_XT_V);
        XT_LSX2IP(b, p1, FF_XT_V);
        XT_LSX2IP(c, p2, FF_XT_V);
        XT_MADD_SX2(c, a, b);             /* c += a * b */
        XT_SSX2IP(c, pd, FF_XT_V);
    }
    if (len & 1)
        dst[len - 1] = src0[len - 1] * src1[len - 1] + src2[len - 1];
}

/* ------------------------------------------------------------------------- */
#elif defined(FF_XTENSA_HIFI_FLOAT_LLVM)  /* LLVM clang: HiFi5 4-wide .sx2x2  */
/* ------------------------------------------------------------------------- */

/*
 * .sx2x2 processes a v2f32 register pair (4 floats) per op, dual-output:
 *   MUL_SX2X2 (a,b, c,d,e,f):  a = c*e, b = d*f     (lanewise, verified on
 *   MADD_SX2X2(a,b, c,d,e,f):  a += c*e, b += d*f    the HiFi5 xt-run sim)
 * The LLVM backend has no xtfloatx2 memory op, so 8-byte pairs move as
 * ae_int32x2 (raw bits) and the .sx2x2 builtins reinterpret them as v2f32 in
 * registers -- no scalar float touches memory. Loops step 4 floats (2 pairs).
 */

#define FF_XT_V 8                        /* bytes per ae_int32x2 pair (2 floats) */

/* Broadcast a scalar to both lanes of a packed pair (raw-bits ae_int32x2). */
static av_always_inline ae_int32x2 ff_xt_splat(float mul)
{
    float __attribute__((aligned(8))) mm[2] = { mul, mul };
    return AE_L32X2_I((const ae_int32x2 *)mm, 0);
}

/* dst[i] = src0[i] * src1[i] */
static void vector_fmul_xtensa(float *dst, const float *src0,
                               const float *src1, int len)
{
    ae_int32x2 *p0 = (ae_int32x2 *)src0;
    ae_int32x2 *p1 = (ae_int32x2 *)src1;
    ae_int32x2 *pd = (ae_int32x2 *)dst;
    ae_int32x2 c, d, e, f;
    ae_xtfloatx2 a, b;
    int i;

    for (i = 0; i + 4 <= len; i += 4) {
        AE_L32X2_IP(c, p0, FF_XT_V);
        AE_L32X2_IP(d, p0, FF_XT_V);
        AE_L32X2_IP(e, p1, FF_XT_V);
        AE_L32X2_IP(f, p1, FF_XT_V);
        XT_MUL_SX2X2(a, b, c, d, e, f);   /* a=c*e, b=d*f */
        AE_S32X2_IP(a, pd, FF_XT_V);
        AE_S32X2_IP(b, pd, FF_XT_V);
    }
    for (; i < len; i++)
        dst[i] = src0[i] * src1[i];
}

/* dst[i] = src[i] * mul */
static void vector_fmul_scalar_xtensa(float *dst, const float *src,
                                      float mul, int len)
{
    ae_int32x2 *ps = (ae_int32x2 *)src;
    ae_int32x2 *pd = (ae_int32x2 *)dst;
    ae_int32x2 vmul = ff_xt_splat(mul);
    ae_int32x2 c, d;
    ae_xtfloatx2 a, b;
    int i;

    for (i = 0; i + 4 <= len; i += 4) {
        AE_L32X2_IP(c, ps, FF_XT_V);
        AE_L32X2_IP(d, ps, FF_XT_V);
        XT_MUL_SX2X2(a, b, c, d, vmul, vmul);
        AE_S32X2_IP(a, pd, FF_XT_V);
        AE_S32X2_IP(b, pd, FF_XT_V);
    }
    for (; i < len; i++)
        dst[i] = src[i] * mul;
}

/* dst[i] += src[i] * mul */
static void vector_fmac_scalar_xtensa(float *dst, const float *src,
                                      float mul, int len)
{
    ae_int32x2 *ps  = (ae_int32x2 *)src;
    ae_int32x2 *pdl = (ae_int32x2 *)dst;
    ae_int32x2 *pds = (ae_int32x2 *)dst;
    ae_int32x2 vmul = ff_xt_splat(mul);
    ae_int32x2 c, d;
    ae_xtfloatx2 a, b;                    /* accumulators = current dst pairs */
    int i;

    for (i = 0; i + 4 <= len; i += 4) {
        AE_L32X2_IP(a, pdl, FF_XT_V);
        AE_L32X2_IP(b, pdl, FF_XT_V);
        AE_L32X2_IP(c, ps, FF_XT_V);
        AE_L32X2_IP(d, ps, FF_XT_V);
        XT_MADD_SX2X2(a, b, c, d, vmul, vmul);   /* a += c*mul, b += d*mul */
        AE_S32X2_IP(a, pds, FF_XT_V);
        AE_S32X2_IP(b, pds, FF_XT_V);
    }
    for (; i < len; i++)
        dst[i] += src[i] * mul;
}

/* dst[i] = src0[i] * src1[i] + src2[i] */
static void vector_fmul_add_xtensa(float *dst, const float *src0,
                                   const float *src1, const float *src2,
                                   int len)
{
    ae_int32x2 *p0 = (ae_int32x2 *)src0;
    ae_int32x2 *p1 = (ae_int32x2 *)src1;
    ae_int32x2 *p2 = (ae_int32x2 *)src2;
    ae_int32x2 *pd = (ae_int32x2 *)dst;
    ae_int32x2 c, d, e, f;
    ae_xtfloatx2 a, b;                    /* accumulators = src2 pairs */
    int i;

    for (i = 0; i + 4 <= len; i += 4) {
        AE_L32X2_IP(a, p2, FF_XT_V);
        AE_L32X2_IP(b, p2, FF_XT_V);
        AE_L32X2_IP(c, p0, FF_XT_V);
        AE_L32X2_IP(d, p0, FF_XT_V);
        AE_L32X2_IP(e, p1, FF_XT_V);
        AE_L32X2_IP(f, p1, FF_XT_V);
        XT_MADD_SX2X2(a, b, c, d, e, f);  /* a += c*e, b += d*f */
        AE_S32X2_IP(a, pd, FF_XT_V);
        AE_S32X2_IP(b, pd, FF_XT_V);
    }
    for (; i < len; i++)
        dst[i] = src0[i] * src1[i] + src2[i];
}

/* ------------------------------------------------------------------------- */
#elif defined(FF_XTENSA_HIFI_FLOAT_LLVM4)  /* LLVM clang: HiFi4 2-wide .sx2   */
/* ------------------------------------------------------------------------- */

/*
 * HiFi4 VFPU 2-wide packed float (xtfloatx2, 2 lanes) via the value-returning
 * XT_MUL_SX2 / XT_MADD_SX2 builtins added to our LLVM backend. LLVM clang has
 * no xtfloatx2 memory op, so 8-byte pairs move as ae_int32x2 (AE_L32X2_IP /
 * AE_S32X2_IP, raw bits) and the .sx2 builtins reinterpret them as v2f32 in
 * registers -- no scalar float touches memory. Loops step 2 floats (1 pair).
 */

#define FF_XT_V 8                        /* bytes per ae_int32x2 pair (2 floats) */

/* Broadcast a scalar to both lanes of a packed pair (raw-bits ae_int32x2). */
static av_always_inline ae_int32x2 ff_xt_splat(float mul)
{
    float __attribute__((aligned(8))) mm[2] = { mul, mul };
    return AE_L32X2_I((const ae_int32x2 *)mm, 0);
}

/* dst[i] = src0[i] * src1[i] */
static void vector_fmul_xtensa(float *dst, const float *src0,
                               const float *src1, int len)
{
    ae_int32x2 *p0 = (ae_int32x2 *)src0;
    ae_int32x2 *p1 = (ae_int32x2 *)src1;
    ae_int32x2 *pd = (ae_int32x2 *)dst;
    ae_int32x2 c, e;
    int i;

    for (i = 0; i + 2 <= len; i += 2) {
        AE_L32X2_IP(c, p0, FF_XT_V);
        AE_L32X2_IP(e, p1, FF_XT_V);
        AE_S32X2_IP(XT_MUL_SX2(c, e), pd, FF_XT_V);
    }
    if (len & 1)
        dst[len - 1] = src0[len - 1] * src1[len - 1];
}

/* dst[i] = src[i] * mul */
static void vector_fmul_scalar_xtensa(float *dst, const float *src,
                                      float mul, int len)
{
    ae_int32x2 *ps = (ae_int32x2 *)src;
    ae_int32x2 *pd = (ae_int32x2 *)dst;
    ae_int32x2 vmul = ff_xt_splat(mul);
    ae_int32x2 c;
    int i;

    for (i = 0; i + 2 <= len; i += 2) {
        AE_L32X2_IP(c, ps, FF_XT_V);
        AE_S32X2_IP(XT_MUL_SX2(c, vmul), pd, FF_XT_V);
    }
    if (len & 1)
        dst[len - 1] = src[len - 1] * mul;
}

/* dst[i] += src[i] * mul */
static void vector_fmac_scalar_xtensa(float *dst, const float *src,
                                      float mul, int len)
{
    ae_int32x2 *ps  = (ae_int32x2 *)src;
    ae_int32x2 *pdl = (ae_int32x2 *)dst;
    ae_int32x2 *pds = (ae_int32x2 *)dst;
    ae_int32x2 vmul = ff_xt_splat(mul);
    ae_int32x2 c, d;
    int i;

    for (i = 0; i + 2 <= len; i += 2) {
        AE_L32X2_IP(d, pdl, FF_XT_V);      /* current dst pair = accumulator */
        AE_L32X2_IP(c, ps, FF_XT_V);
        AE_S32X2_IP(XT_MADD_SX2(d, c, vmul), pds, FF_XT_V);  /* d + c*vmul */
    }
    if (len & 1)
        dst[len - 1] += src[len - 1] * mul;
}

/* dst[i] = src0[i] * src1[i] + src2[i] */
static void vector_fmul_add_xtensa(float *dst, const float *src0,
                                   const float *src1, const float *src2,
                                   int len)
{
    ae_int32x2 *p0 = (ae_int32x2 *)src0;
    ae_int32x2 *p1 = (ae_int32x2 *)src1;
    ae_int32x2 *p2 = (ae_int32x2 *)src2;
    ae_int32x2 *pd = (ae_int32x2 *)dst;
    ae_int32x2 c, e, g;
    int i;

    for (i = 0; i + 2 <= len; i += 2) {
        AE_L32X2_IP(g, p2, FF_XT_V);       /* src2 pair = accumulator */
        AE_L32X2_IP(c, p0, FF_XT_V);
        AE_L32X2_IP(e, p1, FF_XT_V);
        AE_S32X2_IP(XT_MADD_SX2(g, c, e), pd, FF_XT_V);  /* g + c*e */
    }
    if (len & 1)
        dst[len - 1] = src0[len - 1] * src1[len - 1] + src2[len - 1];
}


/* vector_fmul_window: mirrored overlap-add window (see vector_fmul_window_c).
 * For ascending index k (0..len-1):
 *   dst[k]           = src0[k]*win[2len-1-k] - src1[len-1-k]*win[k]
 *   dst[2len-1-k]    = src0[k]*win[k]        + src1[len-1-k]*win[2len-1-k]
 * The reversed operands (src1[len-1-k], win[2len-1-k]) are fetched as a
 * contiguous pair then lane-swapped (AE_SEL32_LH(x,x) exchanges the two 32-bit
 * halves); the mirrored output pair is lane-swapped before the descending store.
 * Load/store are inverses, so the packed math is correct irrespective of which
 * physical lane maps to which memory word (only the pair pairing matters). */
static void vector_fmul_window_xtensa(float *dst, const float *src0,
                                      const float *src1, const float *win,
                                      int len)
{
    int k;
    for (k = 0; k + 2 <= len; k += 2) {
        ae_int32x2 A  = AE_L32X2_I((const ae_int32x2 *)(src0 + k), 0);
        ae_int32x2 Wi = AE_L32X2_I((const ae_int32x2 *)(win  + k), 0);
        /* contiguous reversed-side pairs (ascending in memory) */
        ae_int32x2 Bl = AE_L32X2_I((const ae_int32x2 *)(src1 + (len - 2 - k)), 0);
        ae_int32x2 Wl = AE_L32X2_I((const ae_int32x2 *)(win  + (2 * len - 2 - k)), 0);
        ae_int32x2 B  = AE_SEL32_LH(Bl, Bl);   /* {src1[len-1-k],  src1[len-2-k]}  */
        ae_int32x2 Wj = AE_SEL32_LH(Wl, Wl);   /* {win[2len-1-k],  win[2len-2-k]}  */
        ae_int32x2 L  = XT_MSUB_SX2(XT_MUL_SX2(A, Wj), B, Wi);  /* A*Wj - B*Wi */
        ae_int32x2 R  = XT_MADD_SX2(XT_MUL_SX2(A, Wi), B, Wj);  /* A*Wi + B*Wj */
        AE_S32X2_I(L, (ae_int32x2 *)(dst + k), 0);
        AE_S32X2_I(AE_SEL32_LH(R, R),
                   (ae_int32x2 *)(dst + (2 * len - 2 - k)), 0);
    }
    for (; k < len; k++) {   /* scalar tail (AAC len is always even) */
        float s0 = src0[k], s1 = src1[len - 1 - k];
        float wi = win[k],  wj = win[2 * len - 1 - k];
        dst[k]             = s0 * wj - s1 * wi;
        dst[2 * len - 1 - k] = s0 * wi + s1 * wj;
    }
}

#endif /* SIMD kernel selection */

av_cold void ff_float_dsp_init_xtensa(AVFloatDSPContext *fdsp)
{
#if defined(FF_XTENSA_HIFI_FLOAT) || defined(FF_XTENSA_HIFI_FLOAT_LLVM) || \
    defined(FF_XTENSA_HIFI_FLOAT_LLVM4)
    fdsp->vector_fmul        = vector_fmul_xtensa;
    fdsp->vector_fmul_scalar = vector_fmul_scalar_xtensa;
    fdsp->vector_fmac_scalar = vector_fmac_scalar_xtensa;
    fdsp->vector_fmul_add    = vector_fmul_add_xtensa;
#if defined(FF_XTENSA_HIFI_FLOAT_LLVM4)
    fdsp->vector_fmul_window = vector_fmul_window_xtensa;
#endif
#else
    (void)fdsp;   /* no VFPU / no core config: keep the scalar C kernels */
#endif
}
