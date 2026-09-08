/*
 * Xtensa HiFi optimised Opus DSP functions
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
#include <string.h>

#include "config.h"
#include "libavutil/attributes.h"
#include "libavcodec/opus/dsp.h"

#if defined(__has_include)
#  if __has_include(<xtensa/config/core-isa.h>)
#    include <xtensa/config/core-isa.h>
#  endif
#endif

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

/* =========================================================================
 * 1. Postfilter: 5-Tap Symmetric FIR Pitch Postfilter
 * ========================================================================= */

static void opus_postfilter_xtensa(float *data, int period, float *gains, int len)
{
    const float g0 = gains[0];
    const float g1 = gains[1];
    const float g2 = gains[2];
    int i = 0;

    float x4 = data[-period - 2];
    float x3 = data[-period - 1];
    float x2 = data[-period + 0];
    float x1 = data[-period + 1];

    /* Unroll by 8 to eliminate loop overhead and pipeline FPU multiply-adds */
    for (i = 0; i + 8 <= len; i += 8) {
        float x0_0 = data[i - period + 2];
        float x0_1 = data[i - period + 3];
        float x0_2 = data[i - period + 4];
        float x0_3 = data[i - period + 5];
        float x0_4 = data[i - period + 6];
        float x0_5 = data[i - period + 7];
        float x0_6 = data[i - period + 8];
        float x0_7 = data[i - period + 9];

        data[i + 0] += g0 * x2   + g1 * (x1   + x3)   + g2 * (x0_0 + x4);
        data[i + 1] += g0 * x1   + g1 * (x0_0 + x2)   + g2 * (x0_1 + x3);
        data[i + 2] += g0 * x0_0 + g1 * (x0_1 + x1)   + g2 * (x0_2 + x2);
        data[i + 3] += g0 * x0_1 + g1 * (x0_2 + x0_0) + g2 * (x0_3 + x1);
        data[i + 4] += g0 * x0_2 + g1 * (x0_3 + x0_1) + g2 * (x0_4 + x0_0);
        data[i + 5] += g0 * x0_3 + g1 * (x0_4 + x0_2) + g2 * (x0_5 + x0_1);
        data[i + 6] += g0 * x0_4 + g1 * (x0_5 + x0_3) + g2 * (x0_6 + x0_2);
        data[i + 7] += g0 * x0_5 + g1 * (x0_6 + x0_4) + g2 * (x0_7 + x0_3);

        x4 = x0_4;
        x3 = x0_5;
        x2 = x0_6;
        x1 = x0_7;
    }

    for (; i < len; i++) {
        float x0 = data[i - period + 2];
        data[i] += g0 * x2        +
                   g1 * (x1 + x3) +
                   g2 * (x0 + x4);
        x4 = x3;
        x3 = x2;
        x2 = x1;
        x1 = x0;
    }
}

/* =========================================================================
 * 2. Deemphasis: Pipelined Block IIR Single-Pole Deemphasis Filter
 * ========================================================================= */

static float opus_deemphasis_xtensa(float *y, float *x, float coeff,
                                    const float *weights, int len)
{
    const float c  = weights[0];
    const float c2 = weights[1];
    const float c3 = weights[2];
    const float c4 = weights[3];
    int i = 0;

    /*
     * 8-sample pipelined deemphasis block:
     * Break the 1-sample IIR recurrence into two 4-sample blocks.
     * FIR terms (t0..t7) depend only on input samples x and powers of c,
     * so they evaluate in parallel with the FPU multiply-accumulate pipe,
     * completely eliminating pipeline stalls.
     */
    for (i = 0; i + 8 <= len; i += 8) {
        float x0 = x[i + 0];
        float x1 = x[i + 1];
        float x2 = x[i + 2];
        float x3 = x[i + 3];

        float t0 = x0;
        float t1 = x1 + c  * x0;
        float t2 = x2 + c  * x1 + c2 * x0;
        float t3 = x3 + c  * x2 + c2 * x1 + c3 * x0;

        float x4 = x[i + 4];
        float x5 = x[i + 5];
        float x6 = x[i + 6];
        float x7 = x[i + 7];

        float t4 = x4;
        float t5 = x5 + c  * x4;
        float t6 = x6 + c  * x5 + c2 * x4;
        float t7 = x7 + c  * x6 + c2 * x5 + c3 * x4;

        float y0 = t0 + c  * coeff;
        float y1 = t1 + c2 * coeff;
        float y2 = t2 + c3 * coeff;
        float y3 = t3 + c4 * coeff;

        float y4 = t4 + c  * y3;
        float y5 = t5 + c2 * y3;
        float y6 = t6 + c3 * y3;
        float y7 = t7 + c4 * y3;

        y[i + 0] = y0;
        y[i + 1] = y1;
        y[i + 2] = y2;
        y[i + 3] = y3;
        y[i + 4] = y4;
        y[i + 5] = y5;
        y[i + 6] = y6;
        y[i + 7] = y7;

        coeff = y7;
    }

    for (; i < len; i++)
        coeff = y[i] = x[i] + coeff * c;

    return coeff;
}

/* =========================================================================
 * 3. Architecture Hook Entry Point
 * ========================================================================= */

av_cold void ff_opus_dsp_init_xtensa(OpusDSP *ctx)
{
    ctx->postfilter = opus_postfilter_xtensa;
    ctx->deemphasis = opus_deemphasis_xtensa;
}
