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
 * overlap-add (see the SOF ffmpeg_dec HIFI.md analysis). They use the Cadence
 * single-precision float SIMD (xtfloatx2, 2 lanes), which requires a HiFi core
 * with the VFPU option (e.g. Intel ace30/ptl: HiFi4 + XCHAL_HAVE_HIFI4_VFPU).
 *
 * The intrinsic path is compiled only when the toolchain exposes the Xtensa
 * core config (XCHAL_* via <xtensa/config/core-isa.h>) AND that core has a
 * float VFPU -- i.e. an xt-clang/XCC build for a HiFi4/5-VFPU core. Under the
 * generic Zephyr-SDK GCC cross-build the core config is not on the include path,
 * so FF_XTENSA_HIFI_FLOAT is 0, this file adds nothing, and float_dsp.c keeps
 * its portable scalar C kernels. Correctness is identical either way.
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
 * Take the SIMD path only when the toolchain provides BOTH the core config
 * (reporting a float-VFPU HiFi core) AND the matching Cadence intrinsics header.
 * The Zephyr-SDK GCC ships a core-isa.h with XCHAL_HAVE_HIFI4=1 but NOT
 * xt_hifi4.h, so the header check keeps that build on the scalar fallback;
 * an xt-clang/XCC build for the ace30 core has both and takes the SIMD path.
 */
#if defined(__has_include) && \
    defined(XCHAL_HAVE_HIFI5) && XCHAL_HAVE_HIFI5 && \
    defined(XCHAL_HAVE_HIFI5_VFPU) && XCHAL_HAVE_HIFI5_VFPU && \
    __has_include(<xtensa/tie/xt_hifi5.h>)
#  include <xtensa/tie/xt_hifi5.h>
#  define FF_XTENSA_HIFI_FLOAT 1
#elif defined(__has_include) && \
    defined(XCHAL_HAVE_HIFI4) && XCHAL_HAVE_HIFI4 && \
    defined(XCHAL_HAVE_HIFI4_VFPU) && XCHAL_HAVE_HIFI4_VFPU && \
    __has_include(<xtensa/tie/xt_hifi4.h>)
#  include <xtensa/tie/xt_hifi4.h>
#  define FF_XTENSA_HIFI_FLOAT 1
#else
#  define FF_XTENSA_HIFI_FLOAT 0
#endif

#if FF_XTENSA_HIFI_FLOAT

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

#endif /* FF_XTENSA_HIFI_FLOAT */

av_cold void ff_float_dsp_init_xtensa(AVFloatDSPContext *fdsp)
{
#if FF_XTENSA_HIFI_FLOAT
    fdsp->vector_fmul        = vector_fmul_xtensa;
    fdsp->vector_fmul_scalar = vector_fmul_scalar_xtensa;
    fdsp->vector_fmac_scalar = vector_fmac_scalar_xtensa;
    fdsp->vector_fmul_add    = vector_fmul_add_xtensa;
#else
    (void)fdsp;   /* no VFPU / no core config: keep the scalar C kernels */
#endif
}
