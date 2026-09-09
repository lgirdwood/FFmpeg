/*
 * Xtensa HiFi optimised fixed-point DSP functions
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
#include "libavutil/common.h"
#include "libavutil/fixed_dsp.h"

#if defined(__has_include)
#  if __has_include(<xtensa/config/core-isa.h>)
#    include <xtensa/config/core-isa.h>
#  endif
#endif

/*
 * 4-way unrolled fixed-point windowing and overlap-add for 32-bit integer PCM.
 * Dominant kernel for AAC-LC fixed-point IMDCT windowing.
 */
static void vector_fmul_window_xtensa(int32_t *dst, const int32_t *src0,
                                      const int32_t *src1, const int32_t *win,
                                      int len)
{
    int32_t s0_0, s0_1, s0_2, s0_3;
    int32_t s1_0, s1_1, s1_2, s1_3;
    int32_t wi_0, wi_1, wi_2, wi_3;
    int32_t wj_0, wj_1, wj_2, wj_3;
    int i, j;

    dst  += len;
    win  += len;
    src0 += len;

    for (i = -len, j = len - 1; i + 3 < 0; i += 4, j -= 4) {
        s0_0 = src0[i + 0]; s1_0 = src1[j - 0];
        wi_0 = win[i + 0];  wj_0 = win[j - 0];

        s0_1 = src0[i + 1]; s1_1 = src1[j - 1];
        wi_1 = win[i + 1];  wj_1 = win[j - 1];

        s0_2 = src0[i + 2]; s1_2 = src1[j - 2];
        wi_2 = win[i + 2];  wj_2 = win[j - 2];

        s0_3 = src0[i + 3]; s1_3 = src1[j - 3];
        wi_3 = win[i + 3];  wj_3 = win[j - 3];

        dst[i + 0] = (int32_t)(((int64_t)s0_0 * wj_0 - (int64_t)s1_0 * wi_0 + 0x40000000) >> 31);
        dst[j - 0] = (int32_t)(((int64_t)s0_0 * wi_0 + (int64_t)s1_0 * wj_0 + 0x40000000) >> 31);

        dst[i + 1] = (int32_t)(((int64_t)s0_1 * wj_1 - (int64_t)s1_1 * wi_1 + 0x40000000) >> 31);
        dst[j - 1] = (int32_t)(((int64_t)s0_1 * wi_1 + (int64_t)s1_1 * wj_1 + 0x40000000) >> 31);

        dst[i + 2] = (int32_t)(((int64_t)s0_2 * wj_2 - (int64_t)s1_2 * wi_2 + 0x40000000) >> 31);
        dst[j - 2] = (int32_t)(((int64_t)s0_2 * wi_2 + (int64_t)s1_2 * wj_2 + 0x40000000) >> 31);

        dst[i + 3] = (int32_t)(((int64_t)s0_3 * wj_3 - (int64_t)s1_3 * wi_3 + 0x40000000) >> 31);
        dst[j - 3] = (int32_t)(((int64_t)s0_3 * wi_3 + (int64_t)s1_3 * wj_3 + 0x40000000) >> 31);
    }

    for (; i < 0; i++, j--) {
        int32_t s0 = src0[i];
        int32_t s1 = src1[j];
        int32_t wi = win[i];
        int32_t wj = win[j];
        dst[i] = (int32_t)(((int64_t)s0 * wj - (int64_t)s1 * wi + 0x40000000) >> 31);
        dst[j] = (int32_t)(((int64_t)s0 * wi + (int64_t)s1 * wj + 0x40000000) >> 31);
    }
}

/*
 * Scaled 16-bit windowing and overlap-add with saturation.
 */
static void vector_fmul_window_scaled_xtensa(int16_t *dst, const int32_t *src0,
                                             const int32_t *src1, const int32_t *win,
                                             int len, uint8_t bits)
{
    int32_t s0, s1, wi, wj, i, j;
    int32_t round = bits ? (1 << (bits - 1)) : 0;

    dst  += len;
    win  += len;
    src0 += len;

    for (i = -len, j = len - 1; i + 1 < 0; i += 2, j -= 2) {
        int64_t accu0_i, accu0_j, accu1_i, accu1_j;
        int32_t s0_0 = src0[i], s1_0 = src1[j], wi_0 = win[i], wj_0 = win[j];
        int32_t s0_1 = src0[i + 1], s1_1 = src1[j - 1], wi_1 = win[i + 1], wj_1 = win[j - 1];

        accu0_i = (int64_t)s0_0 * wj_0 - (int64_t)s1_0 * wi_0 + 0x40000000;
        accu0_j = (int64_t)s0_0 * wi_0 + (int64_t)s1_0 * wj_0 + 0x40000000;
        accu1_i = (int64_t)s0_1 * wj_1 - (int64_t)s1_1 * wi_1 + 0x40000000;
        accu1_j = (int64_t)s0_1 * wi_1 + (int64_t)s1_1 * wj_1 + 0x40000000;

        dst[i + 0] = av_clip_int16(((accu0_i >> 31) + round) >> bits);
        dst[j - 0] = av_clip_int16(((accu0_j >> 31) + round) >> bits);
        dst[i + 1] = av_clip_int16(((accu1_i >> 31) + round) >> bits);
        dst[j - 1] = av_clip_int16(((accu1_j >> 31) + round) >> bits);
    }

    for (; i < 0; i++, j--) {
        s0 = src0[i]; s1 = src1[j];
        wi = win[i];  wj = win[j];
        dst[i] = av_clip_int16(((((int64_t)s0 * wj - (int64_t)s1 * wi + 0x40000000) >> 31) + round) >> bits);
        dst[j] = av_clip_int16(((((int64_t)s0 * wi + (int64_t)s1 * wj + 0x40000000) >> 31) + round) >> bits);
    }
}

/*
 * 4-way unrolled fixed-point vector multiplication with rounding.
 */
static void vector_fmul_xtensa(int *dst, const int *src0, const int *src1, int len)
{
    int i = 0;

    for (; i + 4 <= len; i += 4) {
        int64_t a0 = (int64_t)src0[i + 0] * src1[i + 0];
        int64_t a1 = (int64_t)src0[i + 1] * src1[i + 1];
        int64_t a2 = (int64_t)src0[i + 2] * src1[i + 2];
        int64_t a3 = (int64_t)src0[i + 3] * src1[i + 3];

        dst[i + 0] = (int)((a0 + 0x40000000) >> 31);
        dst[i + 1] = (int)((a1 + 0x40000000) >> 31);
        dst[i + 2] = (int)((a2 + 0x40000000) >> 31);
        dst[i + 3] = (int)((a3 + 0x40000000) >> 31);
    }

    for (; i < len; i++) {
        int64_t accu = (int64_t)src0[i] * src1[i];
        dst[i] = (int)((accu + 0x40000000) >> 31);
    }
}

/*
 * 4-way unrolled vector multiply-add.
 */
static void vector_fmul_add_xtensa(int *dst, const int *src0, const int *src1,
                                   const int *src2, int len)
{
    int i = 0;

    for (; i + 4 <= len; i += 4) {
        int64_t a0 = (int64_t)src0[i + 0] * src1[i + 0];
        int64_t a1 = (int64_t)src0[i + 1] * src1[i + 1];
        int64_t a2 = (int64_t)src0[i + 2] * src1[i + 2];
        int64_t a3 = (int64_t)src0[i + 3] * src1[i + 3];

        dst[i + 0] = src2[i + 0] + (int)((a0 + 0x40000000) >> 31);
        dst[i + 1] = src2[i + 1] + (int)((a1 + 0x40000000) >> 31);
        dst[i + 2] = src2[i + 2] + (int)((a2 + 0x40000000) >> 31);
        dst[i + 3] = src2[i + 3] + (int)((a3 + 0x40000000) >> 31);
    }

    for (; i < len; i++) {
        int64_t accu = (int64_t)src0[i] * src1[i];
        dst[i] = src2[i] + (int)((accu + 0x40000000) >> 31);
    }
}

/*
 * 4-way unrolled vector multiply-reverse.
 */
static void vector_fmul_reverse_xtensa(int *dst, const int *src0, const int *src1, int len)
{
    int i = 0;
    src1 += len - 1;

    for (; i + 4 <= len; i += 4) {
        int64_t a0 = (int64_t)src0[i + 0] * src1[-(i + 0)];
        int64_t a1 = (int64_t)src0[i + 1] * src1[-(i + 1)];
        int64_t a2 = (int64_t)src0[i + 2] * src1[-(i + 2)];
        int64_t a3 = (int64_t)src0[i + 3] * src1[-(i + 3)];

        dst[i + 0] = (int)((a0 + 0x40000000) >> 31);
        dst[i + 1] = (int)((a1 + 0x40000000) >> 31);
        dst[i + 2] = (int)((a2 + 0x40000000) >> 31);
        dst[i + 3] = (int)((a3 + 0x40000000) >> 31);
    }

    for (; i < len; i++) {
        int64_t accu = (int64_t)src0[i] * src1[-i];
        dst[i] = (int)((accu + 0x40000000) >> 31);
    }
}

/*
 * 4-way unrolled Mid/Side stereo decorrelator.
 */
static void butterflies_fixed_xtensa(int *restrict v1s, int *restrict v2, int len)
{
    unsigned int *v1 = (unsigned int *)v1s;
    int i = 0;

    for (; i + 4 <= len; i += 4) {
        int t0 = (int)v1[i + 0] - v2[i + 0];
        int t1 = (int)v1[i + 1] - v2[i + 1];
        int t2 = (int)v1[i + 2] - v2[i + 2];
        int t3 = (int)v1[i + 3] - v2[i + 3];

        v1[i + 0] += (unsigned int)v2[i + 0];
        v1[i + 1] += (unsigned int)v2[i + 1];
        v1[i + 2] += (unsigned int)v2[i + 2];
        v1[i + 3] += (unsigned int)v2[i + 3];

        v2[i + 0] = t0;
        v2[i + 1] = t1;
        v2[i + 2] = t2;
        v2[i + 3] = t3;
    }

    for (; i < len; i++) {
        int t = (int)v1[i] - v2[i];
        v1[i] += (unsigned int)v2[i];
        v2[i] = t;
    }
}

/*
 * 4-way unrolled scalar dot product.
 */
static int scalarproduct_fixed_xtensa(const int *v1, const int *v2, int len)
{
    int64_t p = 0x40000000;
    int i = 0;

    for (; i + 4 <= len; i += 4) {
        p += (int64_t)v1[i + 0] * v2[i + 0];
        p += (int64_t)v1[i + 1] * v2[i + 1];
        p += (int64_t)v1[i + 2] * v2[i + 2];
        p += (int64_t)v1[i + 3] * v2[i + 3];
    }

    for (; i < len; i++)
        p += (int64_t)v1[i] * v2[i];

    return (int)(p >> 31);
}

av_cold void ff_fixed_dsp_init_xtensa(AVFixedDSPContext *fdsp)
{
    fdsp->vector_fmul_window        = vector_fmul_window_xtensa;
    fdsp->vector_fmul_window_scaled = vector_fmul_window_scaled_xtensa;
    fdsp->vector_fmul               = vector_fmul_xtensa;
    fdsp->vector_fmul_add           = vector_fmul_add_xtensa;
    fdsp->vector_fmul_reverse       = vector_fmul_reverse_xtensa;
    fdsp->butterflies_fixed         = butterflies_fixed_xtensa;
    fdsp->scalarproduct_fixed       = scalarproduct_fixed_xtensa;
}
