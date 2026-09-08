/*
 * Xtensa HiFi optimised FLAC DSP functions
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

#include <stdint.h>
#include "config.h"
#include "libavutil/attributes.h"
#include "libavutil/samplefmt.h"
#include "libavcodec/flacdsp.h"

#if defined(__has_include)
#  if __has_include(<xtensa/config/core-isa.h>)
#    include <xtensa/config/core-isa.h>
#  endif
#endif

#if defined(__has_include) && \
    ((defined(XCHAL_HAVE_HIFI3) && XCHAL_HAVE_HIFI3) || \
     (defined(XCHAL_HAVE_HIFI4) && XCHAL_HAVE_HIFI4) || \
     (defined(XCHAL_HAVE_HIFI5) && XCHAL_HAVE_HIFI5))
#  if __has_include(<xtensa/tie/xt_hifi3.h>)
#    include <xtensa/tie/xt_hifi3.h>
#    define FF_XTENSA_HIFI_FLAC 1
#  elif __has_include(<xtensa/tie/xt_hifi4.h>)
#    include <xtensa/tie/xt_hifi4.h>
#    define FF_XTENSA_HIFI_FLAC 1
#  elif __has_include(<xtensahifiintrin.h>)
#    include <xtensahifiintrin.h>
#    define FF_XTENSA_HIFI_FLAC 1
#  endif
#endif

/* =========================================================================
 * 1. LPC16 Residual Prediction Filter
 * ========================================================================= */

#define LPC16_UNROLL_ORDER(ORDER)                                            \
static av_always_inline void flac_lpc_16_order_##ORDER(int32_t *decoded,     \
                                                       const int coeffs[32], \
                                                       int qlevel, int len)  \
{                                                                            \
    int i, j;                                                                \
    for (i = (ORDER); i < len - 1; i += 2, decoded += 2) {                  \
        int c = coeffs[0];                                                   \
        int d = decoded[0];                                                  \
        int s0 = 0, s1 = 0;                                                  \
        _Pragma("unroll")                                                    \
        for (j = 1; j < (ORDER); j++) {                                      \
            s0 += c * d;                                                     \
            d = decoded[j];                                                  \
            s1 += c * d;                                                     \
            c = coeffs[j];                                                   \
        }                                                                    \
        s0 += c * d;                                                         \
        d = decoded[j] += (s0 >> qlevel);                                    \
        s1 += c * d;                                                         \
        decoded[j + 1] += (s1 >> qlevel);                                    \
    }                                                                        \
    if (i < len) {                                                           \
        int sum = 0;                                                         \
        _Pragma("unroll")                                                    \
        for (j = 0; j < (ORDER); j++)                                        \
            sum += coeffs[j] * decoded[j];                                   \
        decoded[j] += (sum >> qlevel);                                       \
    }                                                                        \
}

LPC16_UNROLL_ORDER(1)
LPC16_UNROLL_ORDER(2)
LPC16_UNROLL_ORDER(4)
LPC16_UNROLL_ORDER(6)
LPC16_UNROLL_ORDER(8)
LPC16_UNROLL_ORDER(10)
LPC16_UNROLL_ORDER(12)
LPC16_UNROLL_ORDER(16)

static void flac_lpc_16_xtensa(int32_t *decoded, const int coeffs[32],
                               int pred_order, int qlevel, int len)
{
    int i, j;

    switch (pred_order) {
    case 1:  flac_lpc_16_order_1(decoded, coeffs, qlevel, len);  return;
    case 2:  flac_lpc_16_order_2(decoded, coeffs, qlevel, len);  return;
    case 4:  flac_lpc_16_order_4(decoded, coeffs, qlevel, len);  return;
    case 6:  flac_lpc_16_order_6(decoded, coeffs, qlevel, len);  return;
    case 8:  flac_lpc_16_order_8(decoded, coeffs, qlevel, len);  return;
    case 10: flac_lpc_16_order_10(decoded, coeffs, qlevel, len); return;
    case 12: flac_lpc_16_order_12(decoded, coeffs, qlevel, len); return;
    case 16: flac_lpc_16_order_16(decoded, coeffs, qlevel, len); return;
    default: break;
    }

    /* General order loop unrolled by 2 across sample pairs, unrolled by 4 in MACs */
    for (i = pred_order; i < len - 1; i += 2, decoded += 2) {
        int c = coeffs[0];
        int d = decoded[0];
        int s0 = 0, s1 = 0;

        for (j = 1; j + 3 < pred_order; j += 4) {
            s0 += c * d; d = decoded[j + 0]; s1 += c * d; c = coeffs[j + 0];
            s0 += c * d; d = decoded[j + 1]; s1 += c * d; c = coeffs[j + 1];
            s0 += c * d; d = decoded[j + 2]; s1 += c * d; c = coeffs[j + 2];
            s0 += c * d; d = decoded[j + 3]; s1 += c * d; c = coeffs[j + 3];
        }
        for (; j < pred_order; j++) {
            s0 += c * d;
            d = decoded[j];
            s1 += c * d;
            c = coeffs[j];
        }
        s0 += c * d;
        d = decoded[j] += (s0 >> qlevel);
        s1 += c * d;
        decoded[j + 1] += (s1 >> qlevel);
    }
    if (i < len) {
        int sum = 0;
        for (j = 0; j < pred_order; j++)
            sum += coeffs[j] * decoded[j];
        decoded[j] += (sum >> qlevel);
    }
}

/* =========================================================================
 * 2. LPC32 Residual Prediction Filter (64-bit Accumulation)
 * ========================================================================= */

#define LPC32_UNROLL_ORDER(ORDER)                                            \
static av_always_inline void flac_lpc_32_order_##ORDER(int32_t *decoded,     \
                                                       const int coeffs[32], \
                                                       int qlevel, int len)  \
{                                                                            \
    int i, j;                                                                \
    for (i = (ORDER); i < len; i++, decoded++) {                             \
        int64_t sum = 0;                                                     \
        _Pragma("unroll")                                                    \
        for (j = 0; j < (ORDER); j++)                                        \
            sum += (int64_t)coeffs[j] * decoded[j];                          \
        decoded[(ORDER)] += (int32_t)(sum >> qlevel);                        \
    }                                                                        \
}

LPC32_UNROLL_ORDER(4)
LPC32_UNROLL_ORDER(8)
LPC32_UNROLL_ORDER(12)
LPC32_UNROLL_ORDER(16)

static void flac_lpc_32_xtensa(int32_t *decoded, const int coeffs[32],
                               int pred_order, int qlevel, int len)
{
    int i, j;

    switch (pred_order) {
    case 4:  flac_lpc_32_order_4(decoded, coeffs, qlevel, len);  return;
    case 8:  flac_lpc_32_order_8(decoded, coeffs, qlevel, len);  return;
    case 12: flac_lpc_32_order_12(decoded, coeffs, qlevel, len); return;
    case 16: flac_lpc_32_order_16(decoded, coeffs, qlevel, len); return;
    default: break;
    }

    for (i = pred_order; i < len; i++, decoded++) {
        int64_t sum = 0;
        for (j = 0; j + 3 < pred_order; j += 4) {
            sum += (int64_t)coeffs[j + 0] * decoded[j + 0];
            sum += (int64_t)coeffs[j + 1] * decoded[j + 1];
            sum += (int64_t)coeffs[j + 2] * decoded[j + 2];
            sum += (int64_t)coeffs[j + 3] * decoded[j + 3];
        }
        for (; j < pred_order; j++)
            sum += (int64_t)coeffs[j] * decoded[j];
        decoded[pred_order] += (int32_t)(sum >> qlevel);
    }
}

/* =========================================================================
 * 3. Wasted Bits
 * ========================================================================= */

static void flac_wasted_32_xtensa(int32_t *decoded, int wasted, int len)
{
    int i = 0;
    for (; i + 4 <= len; i += 4) {
        decoded[i + 0] = (uint32_t)decoded[i + 0] << wasted;
        decoded[i + 1] = (uint32_t)decoded[i + 1] << wasted;
        decoded[i + 2] = (uint32_t)decoded[i + 2] << wasted;
        decoded[i + 3] = (uint32_t)decoded[i + 3] << wasted;
    }
    for (; i < len; i++)
        decoded[i] = (uint32_t)decoded[i] << wasted;
}

/* =========================================================================
 * 4. Channel Decorrelators: S16 Interleaved
 * ========================================================================= */

static void flac_decorrelate_ls_16_xtensa(uint8_t **out, int32_t **in,
                                          int channels, int len, int shift)
{
    int16_t *dst = (int16_t *)out[0];
    const int32_t *in0 = in[0];
    const int32_t *in1 = in[1];
    int i = 0;

    for (; i + 4 <= len; i += 4) {
        int32_t a0 = in0[i + 0], b0 = in1[i + 0];
        int32_t a1 = in0[i + 1], b1 = in1[i + 1];
        int32_t a2 = in0[i + 2], b2 = in1[i + 2];
        int32_t a3 = in0[i + 3], b3 = in1[i + 3];

        dst[2 * i + 0] = (int16_t)(a0 << shift);
        dst[2 * i + 1] = (int16_t)((a0 - b0) << shift);
        dst[2 * i + 2] = (int16_t)(a1 << shift);
        dst[2 * i + 3] = (int16_t)((a1 - b1) << shift);
        dst[2 * i + 4] = (int16_t)(a2 << shift);
        dst[2 * i + 5] = (int16_t)((a2 - b2) << shift);
        dst[2 * i + 6] = (int16_t)(a3 << shift);
        dst[2 * i + 7] = (int16_t)((a3 - b3) << shift);
    }
    for (; i < len; i++) {
        int32_t a = in0[i], b = in1[i];
        dst[2 * i + 0] = (int16_t)(a << shift);
        dst[2 * i + 1] = (int16_t)((a - b) << shift);
    }
}

static void flac_decorrelate_rs_16_xtensa(uint8_t **out, int32_t **in,
                                          int channels, int len, int shift)
{
    int16_t *dst = (int16_t *)out[0];
    const int32_t *in0 = in[0];
    const int32_t *in1 = in[1];
    int i = 0;

    for (; i + 4 <= len; i += 4) {
        int32_t a0 = in0[i + 0], b0 = in1[i + 0];
        int32_t a1 = in0[i + 1], b1 = in1[i + 1];
        int32_t a2 = in0[i + 2], b2 = in1[i + 2];
        int32_t a3 = in0[i + 3], b3 = in1[i + 3];

        dst[2 * i + 0] = (int16_t)((a0 + b0) << shift);
        dst[2 * i + 1] = (int16_t)(b0 << shift);
        dst[2 * i + 2] = (int16_t)((a1 + b1) << shift);
        dst[2 * i + 3] = (int16_t)(b1 << shift);
        dst[2 * i + 4] = (int16_t)((a2 + b2) << shift);
        dst[2 * i + 5] = (int16_t)(b2 << shift);
        dst[2 * i + 6] = (int16_t)((a3 + b3) << shift);
        dst[2 * i + 7] = (int16_t)(b3 << shift);
    }
    for (; i < len; i++) {
        int32_t a = in0[i], b = in1[i];
        dst[2 * i + 0] = (int16_t)((a + b) << shift);
        dst[2 * i + 1] = (int16_t)(b << shift);
    }
}

static void flac_decorrelate_ms_16_xtensa(uint8_t **out, int32_t **in,
                                          int channels, int len, int shift)
{
    int16_t *dst = (int16_t *)out[0];
    const int32_t *in0 = in[0];
    const int32_t *in1 = in[1];
    int i = 0;

    for (; i + 4 <= len; i += 4) {
        int32_t a0 = in0[i + 0], b0 = in1[i + 0];
        int32_t a1 = in0[i + 1], b1 = in1[i + 1];
        int32_t a2 = in0[i + 2], b2 = in1[i + 2];
        int32_t a3 = in0[i + 3], b3 = in1[i + 3];

        a0 -= b0 >> 1;
        a1 -= b1 >> 1;
        a2 -= b2 >> 1;
        a3 -= b3 >> 1;

        dst[2 * i + 0] = (int16_t)((a0 + b0) << shift);
        dst[2 * i + 1] = (int16_t)(a0 << shift);
        dst[2 * i + 2] = (int16_t)((a1 + b1) << shift);
        dst[2 * i + 3] = (int16_t)(a1 << shift);
        dst[2 * i + 4] = (int16_t)((a2 + b2) << shift);
        dst[2 * i + 5] = (int16_t)(a2 << shift);
        dst[2 * i + 6] = (int16_t)((a3 + b3) << shift);
        dst[2 * i + 7] = (int16_t)(a3 << shift);
    }
    for (; i < len; i++) {
        int32_t a = in0[i], b = in1[i];
        a -= b >> 1;
        dst[2 * i + 0] = (int16_t)((a + b) << shift);
        dst[2 * i + 1] = (int16_t)(a << shift);
    }
}

static void flac_decorrelate_indep2_16_xtensa(uint8_t **out, int32_t **in,
                                              int channels, int len, int shift)
{
    int16_t *dst = (int16_t *)out[0];
    const int32_t *in0 = in[0];
    const int32_t *in1 = in[1];
    int i = 0;

    for (; i + 4 <= len; i += 4) {
        dst[2 * i + 0] = (int16_t)(in0[i + 0] << shift);
        dst[2 * i + 1] = (int16_t)(in1[i + 0] << shift);
        dst[2 * i + 2] = (int16_t)(in0[i + 1] << shift);
        dst[2 * i + 3] = (int16_t)(in1[i + 1] << shift);
        dst[2 * i + 4] = (int16_t)(in0[i + 2] << shift);
        dst[2 * i + 5] = (int16_t)(in1[i + 2] << shift);
        dst[2 * i + 6] = (int16_t)(in0[i + 3] << shift);
        dst[2 * i + 7] = (int16_t)(in1[i + 3] << shift);
    }
    for (; i < len; i++) {
        dst[2 * i + 0] = (int16_t)(in0[i] << shift);
        dst[2 * i + 1] = (int16_t)(in1[i] << shift);
    }
}

/* =========================================================================
 * 5. Channel Decorrelators: S32 Interleaved
 * ========================================================================= */

static void flac_decorrelate_ls_32_xtensa(uint8_t **out, int32_t **in,
                                          int channels, int len, int shift)
{
    int32_t *dst = (int32_t *)out[0];
    const int32_t *in0 = in[0];
    const int32_t *in1 = in[1];
    int i = 0;

#if defined(FF_XTENSA_HIFI_FLAC)
    if (shift == 0) {
        for (; i + 2 <= len; i += 2) {
            ae_int32x2 a = AE_L32X2_I((const ae_int32x2 *)&in0[i], 0);
            ae_int32x2 b = AE_L32X2_I((const ae_int32x2 *)&in1[i], 0);
            ae_int32x2 diff = AE_SUB32(a, b);
            ae_int32x2 pair0 = AE_SEL32_LL(a, diff);
            ae_int32x2 pair1 = AE_SEL32_HH(a, diff);

            AE_S32X2_I(pair0, (ae_int32x2 *)&dst[2 * i + 0], 0);
            AE_S32X2_I(pair1, (ae_int32x2 *)&dst[2 * i + 2], 0);
        }
    }
#endif
    for (; i < len; i++) {
        int32_t a = in0[i], b = in1[i];
        dst[2 * i + 0] = (uint32_t)a << shift;
        dst[2 * i + 1] = (uint32_t)(a - b) << shift;
    }
}

static void flac_decorrelate_rs_32_xtensa(uint8_t **out, int32_t **in,
                                          int channels, int len, int shift)
{
    int32_t *dst = (int32_t *)out[0];
    const int32_t *in0 = in[0];
    const int32_t *in1 = in[1];
    int i = 0;

#if defined(FF_XTENSA_HIFI_FLAC)
    if (shift == 0) {
        for (; i + 2 <= len; i += 2) {
            ae_int32x2 a = AE_L32X2_I((const ae_int32x2 *)&in0[i], 0);
            ae_int32x2 b = AE_L32X2_I((const ae_int32x2 *)&in1[i], 0);
            ae_int32x2 sum = AE_ADD32(a, b);
            ae_int32x2 pair0 = AE_SEL32_LL(sum, b);
            ae_int32x2 pair1 = AE_SEL32_HH(sum, b);

            AE_S32X2_I(pair0, (ae_int32x2 *)&dst[2 * i + 0], 0);
            AE_S32X2_I(pair1, (ae_int32x2 *)&dst[2 * i + 2], 0);
        }
    }
#endif
    for (; i < len; i++) {
        int32_t a = in0[i], b = in1[i];
        dst[2 * i + 0] = (uint32_t)(a + b) << shift;
        dst[2 * i + 1] = (uint32_t)b << shift;
    }
}

static void flac_decorrelate_ms_32_xtensa(uint8_t **out, int32_t **in,
                                          int channels, int len, int shift)
{
    int32_t *dst = (int32_t *)out[0];
    const int32_t *in0 = in[0];
    const int32_t *in1 = in[1];
    int i = 0;

    for (; i + 4 <= len; i += 4) {
        int32_t a0 = in0[i + 0], b0 = in1[i + 0];
        int32_t a1 = in0[i + 1], b1 = in1[i + 1];
        int32_t a2 = in0[i + 2], b2 = in1[i + 2];
        int32_t a3 = in0[i + 3], b3 = in1[i + 3];

        a0 -= b0 >> 1;
        a1 -= b1 >> 1;
        a2 -= b2 >> 1;
        a3 -= b3 >> 1;

        dst[2 * i + 0] = (uint32_t)(a0 + b0) << shift;
        dst[2 * i + 1] = (uint32_t)a0 << shift;
        dst[2 * i + 2] = (uint32_t)(a1 + b1) << shift;
        dst[2 * i + 3] = (uint32_t)a1 << shift;
        dst[2 * i + 4] = (uint32_t)(a2 + b2) << shift;
        dst[2 * i + 5] = (uint32_t)a2 << shift;
        dst[2 * i + 6] = (uint32_t)(a3 + b3) << shift;
        dst[2 * i + 7] = (uint32_t)a3 << shift;
    }
    for (; i < len; i++) {
        int32_t a = in0[i], b = in1[i];
        a -= b >> 1;
        dst[2 * i + 0] = (uint32_t)(a + b) << shift;
        dst[2 * i + 1] = (uint32_t)a << shift;
    }
}

static void flac_decorrelate_indep2_32_xtensa(uint8_t **out, int32_t **in,
                                              int channels, int len, int shift)
{
    int32_t *dst = (int32_t *)out[0];
    const int32_t *in0 = in[0];
    const int32_t *in1 = in[1];
    int i = 0;

#if defined(FF_XTENSA_HIFI_FLAC)
    if (shift == 0) {
        for (; i + 2 <= len; i += 2) {
            ae_int32x2 a = AE_L32X2_I((const ae_int32x2 *)&in0[i], 0);
            ae_int32x2 b = AE_L32X2_I((const ae_int32x2 *)&in1[i], 0);
            ae_int32x2 pair0 = AE_SEL32_LL(a, b);
            ae_int32x2 pair1 = AE_SEL32_HH(a, b);

            AE_S32X2_I(pair0, (ae_int32x2 *)&dst[2 * i + 0], 0);
            AE_S32X2_I(pair1, (ae_int32x2 *)&dst[2 * i + 2], 0);
        }
    }
#endif
    for (; i < len; i++) {
        dst[2 * i + 0] = (uint32_t)in0[i] << shift;
        dst[2 * i + 1] = (uint32_t)in1[i] << shift;
    }
}

/* =========================================================================
 * 6. Top-Level Xtensa FLAC DSP Init
 * ========================================================================= */

av_cold void ff_flacdsp_init_xtensa(FLACDSPContext *c, enum AVSampleFormat fmt,
                                    int channels)
{
    c->lpc16    = flac_lpc_16_xtensa;
    c->lpc32    = flac_lpc_32_xtensa;
    c->wasted32 = flac_wasted_32_xtensa;

    if (fmt == AV_SAMPLE_FMT_S16 && channels == 2) {
        c->decorrelate[0] = flac_decorrelate_indep2_16_xtensa;
        c->decorrelate[1] = flac_decorrelate_ls_16_xtensa;
        c->decorrelate[2] = flac_decorrelate_rs_16_xtensa;
        c->decorrelate[3] = flac_decorrelate_ms_16_xtensa;
    } else if (fmt == AV_SAMPLE_FMT_S32 && channels == 2) {
        c->decorrelate[0] = flac_decorrelate_indep2_32_xtensa;
        c->decorrelate[1] = flac_decorrelate_ls_32_xtensa;
        c->decorrelate[2] = flac_decorrelate_rs_32_xtensa;
        c->decorrelate[3] = flac_decorrelate_ms_32_xtensa;
    }
}
