/*
 * AAC decoder
 * Copyright (c) 2005-2006 Oded Shimon ( ods15 ods15 dyndns org )
 * Copyright (c) 2006-2007 Maxim Gavrilov ( maxim.gavrilov gmail com )
 * Copyright (c) 2008-2013 Alex Converse <alex.converse@gmail.com>
 *
 * AAC LATM decoder
 * Copyright (c) 2008-2010 Paul Kendall <paul@kcbbs.gen.nz>
 * Copyright (c) 2010      Janne Grunau <janne-libav@jannau.net>
 *
 * AAC decoder fixed-point implementation
 * Copyright (c) 2013
 *      MIPS Technologies, Inc., California.
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

#define USE_FIXED 0

#include "libavutil/thread.h"

#include "libavcodec/aac_defines.h"

#include "libavcodec/avcodec.h"
#include "aacdec.h"
#include "libavcodec/aactab.h"
#include "libavcodec/sinewin.h"
#include "libavcodec/kbdwin.h"
#include <string.h>
#include "aac_kbd_hardcoded.h"
#include "libavcodec/cbrt_data.h"
#include "libavutil/mathematics.h"
#include "libavutil/log.h"
#include "libavcodec/aacsbr.h"

DECLARE_ALIGNED(32, static float, sine_96)[96];
DECLARE_ALIGNED(32, static float, sine_120)[120];
DECLARE_ALIGNED(32, static float, sine_768)[768];
DECLARE_ALIGNED(32, static float, sine_960)[960];
DECLARE_ALIGNED(32, static float, aac_kbd_long_960)[960];
DECLARE_ALIGNED(32, static float, aac_kbd_short_120)[120];
DECLARE_ALIGNED(32, static float, aac_kbd_long_768)[768];
DECLARE_ALIGNED(32, static float, aac_kbd_short_96)[96];

static void init_tables_float_fn(void)
{
    ff_cbrt_tableinit();

    /* Hardcoded at build time: runtime ff_kbd_window_init() spends ~15s in
     * soft-float double av_bessel_i0() at avcodec_open2() on the SOF DSP.
     * Values are IEEE-identical (rational approximation + correctly-rounded
     * sqrt). See aac_kbd_hardcoded.h. */
    memcpy(ff_aac_kbd_long_1024, hc_aac_kbd_long_1024, sizeof(hc_aac_kbd_long_1024));
    memcpy(ff_aac_kbd_short_128, hc_aac_kbd_short_128, sizeof(hc_aac_kbd_short_128));

    memcpy(aac_kbd_long_960, hc_aac_kbd_long_960, sizeof(hc_aac_kbd_long_960));
    memcpy(aac_kbd_short_120, hc_aac_kbd_short_120, sizeof(hc_aac_kbd_short_120));

    ff_sine_window_init(sine_960, 960);
    ff_sine_window_init(sine_120, 120);
    ff_init_ff_sine_windows(9);

    ff_aac_sbr_init();

    ff_aac_float_common_init();
}

static const float cce_scale[] = {
    1.09050773266525765921, //2^(1/8)
    1.18920711500272106672, //2^(1/4)
    M_SQRT2,
    2,
};

/** Dequantization-related **/
#include "aacdec_tab.h"
#include "libavutil/intfloat.h"

#include "config.h"
#if ARCH_ARM
#include "libavcodec/arm/aac.h"
#endif

/*
 * SOF/aphid: this silicon has NO scalar HW-FP; a scalar `float` multiply is
 * soft-float (~thousands of ccount). The plain-codebook dequant below normally
 * multiplies each coefficient inline by the band scalefactor -> dominated loud
 * frames (~8000 ccount/coeff). When the VFPU vector_fmul_scalar override is
 * present, define the VMULn helpers to only GATHER the codebook value (applying
 * sign via an integer xor on the float bit-pattern, no float arithmetic); the
 * uniform per-band scale is then applied once via ac->fdsp->vector_fmul_scalar
 * in decode_spectrum_and_dequant (see aacdec_proc_template.c). Algebraically
 * bit-exact: (+-v)*s == +-(v*s). Mirrors what the escape codebook already does.
 */
#if defined(__has_include)
#  if __has_include(<xtensa/config/core-isa.h>)
#    include <xtensa/config/core-isa.h>
#  endif
#endif
#if !defined(__XCC__) && defined(__has_builtin) && \
    __has_builtin(__builtin_xtensa_mul_sx2) && \
    defined(XCHAL_HAVE_HIFI4_VFPU) && XCHAL_HAVE_HIFI4_VFPU
#  define FF_AAC_VFPU_DEQUANT 1

/*
 * Bit-exact scalar float multiply via the HiFi4 VFPU. This silicon has no
 * scalar HW-FP, so a plain `a*b` becomes a soft-float __mulsf3 call (~8000
 * ccount). Route it through one 2-lane XT_MUL_SX2 and read lane 0: the VFPU
 * lane is IEEE-754 single, round-to-nearest, so the result is bit-identical to
 * the scalar multiply while costing a few cycles instead of thousands. Used by
 * apply_tns (aacdec_dsp_template.c), whose recursive AR filter must keep its
 * exact per-tap summation order (so it can't be a wider vector reduction).
 */
#if defined(__has_include) && __has_include(<xtensahifiintrin.h>)
#  include <xtensahifiintrin.h>
#endif
static inline float ff_vfpu_smul(float a, float b)
{
    float __attribute__((aligned(8))) aa[2] = { a, a };
    float __attribute__((aligned(8))) bb[2] = { b, b };
    float __attribute__((aligned(8))) rr[2];
    ae_int32x2 va = AE_L32X2_I((const ae_int32x2 *)aa, 0);
    ae_int32x2 vb = AE_L32X2_I((const ae_int32x2 *)bb, 0);
    AE_S32X2_I(XT_MUL_SX2(va, vb), (ae_int32x2 *)rr, 0);
    return rr[0];
}

static inline float *VMUL2(float *dst, const float *v, unsigned idx,
                           const float *scale)
{
    (void)scale;
    *dst++ = v[idx    & 15];
    *dst++ = v[idx>>4 & 15];
    return dst;
}
#define VMUL2 VMUL2

static inline float *VMUL4(float *dst, const float *v, unsigned idx,
                           const float *scale)
{
    (void)scale;
    *dst++ = v[idx    & 3];
    *dst++ = v[idx>>2 & 3];
    *dst++ = v[idx>>4 & 3];
    *dst++ = v[idx>>6 & 3];
    return dst;
}
#define VMUL4 VMUL4

static inline float *VMUL2S(float *dst, const float *v, unsigned idx,
                            unsigned sign, const float *scale)
{
    union av_intfloat32 t0, t1;
    (void)scale;
    t0.f = v[idx    & 15];
    t1.f = v[idx>>4 & 15];
    t0.i ^= sign >> 1 << 31;
    t1.i ^= sign      << 31;
    *dst++ = t0.f;
    *dst++ = t1.f;
    return dst;
}
#define VMUL2S VMUL2S

static inline float *VMUL4S(float *dst, const float *v, unsigned idx,
                            unsigned sign, const float *scale)
{
    unsigned nz = idx >> 12;
    union av_intfloat32 t;
    (void)scale;
    t.f = v[idx    & 3]; t.i ^= sign & 1U<<31; *dst++ = t.f;
    sign <<= nz & 1; nz >>= 1;
    t.f = v[idx>>2 & 3]; t.i ^= sign & 1U<<31; *dst++ = t.f;
    sign <<= nz & 1; nz >>= 1;
    t.f = v[idx>>4 & 3]; t.i ^= sign & 1U<<31; *dst++ = t.f;
    sign <<= nz & 1;
    t.f = v[idx>>6 & 3]; t.i ^= sign & 1U<<31; *dst++ = t.f;
    return dst;
}
#define VMUL4S VMUL4S
#endif /* FF_AAC_VFPU_DEQUANT */

#ifndef VMUL2
static inline float *VMUL2(float *dst, const float *v, unsigned idx,
                           const float *scale)
{
    float s = *scale;
    *dst++ = v[idx    & 15] * s;
    *dst++ = v[idx>>4 & 15] * s;
    return dst;
}
#endif

#ifndef VMUL4
static inline float *VMUL4(float *dst, const float *v, unsigned idx,
                           const float *scale)
{
    float s = *scale;
    *dst++ = v[idx    & 3] * s;
    *dst++ = v[idx>>2 & 3] * s;
    *dst++ = v[idx>>4 & 3] * s;
    *dst++ = v[idx>>6 & 3] * s;
    return dst;
}
#endif

#ifndef VMUL2S
static inline float *VMUL2S(float *dst, const float *v, unsigned idx,
                            unsigned sign, const float *scale)
{
    union av_intfloat32 s0, s1;

    s0.f = s1.f = *scale;
    s0.i ^= sign >> 1 << 31;
    s1.i ^= sign      << 31;

    *dst++ = v[idx    & 15] * s0.f;
    *dst++ = v[idx>>4 & 15] * s1.f;

    return dst;
}
#endif

#ifndef VMUL4S
static inline float *VMUL4S(float *dst, const float *v, unsigned idx,
                            unsigned sign, const float *scale)
{
    unsigned nz = idx >> 12;
    union av_intfloat32 s = { .f = *scale };
    union av_intfloat32 t;

    t.i = s.i ^ (sign & 1U<<31);
    *dst++ = v[idx    & 3] * t.f;

    sign <<= nz & 1; nz >>= 1;
    t.i = s.i ^ (sign & 1U<<31);
    *dst++ = v[idx>>2 & 3] * t.f;

    sign <<= nz & 1; nz >>= 1;
    t.i = s.i ^ (sign & 1U<<31);
    *dst++ = v[idx>>4 & 3] * t.f;

    sign <<= nz & 1;
    t.i = s.i ^ (sign & 1U<<31);
    *dst++ = v[idx>>6 & 3] * t.f;

    return dst;
}
#endif

#include "aacdec_float_coupling.h"
#include "aacdec_float_prediction.h"
#include "aacdec_dsp_template.c"
#include "aacdec_proc_template.c"

av_cold int ff_aac_decode_init_float(AVCodecContext *avctx)
{
    static AVOnce init_float_once = AV_ONCE_INIT;
    AACDecContext *ac = avctx->priv_data;

    ac->is_fixed = 0;
    avctx->sample_fmt = AV_SAMPLE_FMT_FLTP;

    aac_dsp_init(&ac->dsp);
    aac_proc_init(&ac->proc);

    ac->fdsp = avpriv_float_dsp_alloc(avctx->flags & AV_CODEC_FLAG_BITEXACT);
    if (!ac->fdsp)
        return AVERROR(ENOMEM);

    ff_thread_once(&init_float_once, init_tables_float_fn);

    return ff_aac_decode_init(avctx);
}
