/*
 * Xtensa HiFi optimised MPEG Audio fixed-point DSP functions
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
#include "libavcodec/mpegaudio.h"
#include "libavcodec/mpegaudiodsp.h"
#include "libavcodec/mathops.h"

/* =========================================================================
 * Fixed-point helper macros and rounding
 * ========================================================================= */

#define OUT_SHIFT (WFRAC_BITS + FRAC_BITS - 15)

static av_always_inline int16_t round_sample_fixed(int64_t *sum)
{
    int32_t sum1 = (int32_t)(*sum >> OUT_SHIFT);
    *sum &= (1 << OUT_SHIFT) - 1;
    if (sum1 > 32767)
        return 32767;
    if (sum1 < -32768)
        return -32768;
    return (int16_t)sum1;
}

/* =========================================================================
 * 1. Polyphase Synthesis Window Filter (apply_window_fixed)
 * ========================================================================= */

#define DUAL_TAP(sum1, sum2, w1, w2, p, sign1, sign2)                        \
    do {                                                                      \
        int32_t val;                                                          \
        val = (p)[0 * 64]; sum1 sign1 (int64_t)(w1)[0 * 64] * val; sum2 sign2 (int64_t)(w2)[0 * 64] * val; \
        val = (p)[1 * 64]; sum1 sign1 (int64_t)(w1)[1 * 64] * val; sum2 sign2 (int64_t)(w2)[1 * 64] * val; \
        val = (p)[2 * 64]; sum1 sign1 (int64_t)(w1)[2 * 64] * val; sum2 sign2 (int64_t)(w2)[2 * 64] * val; \
        val = (p)[3 * 64]; sum1 sign1 (int64_t)(w1)[3 * 64] * val; sum2 sign2 (int64_t)(w2)[3 * 64] * val; \
        val = (p)[4 * 64]; sum1 sign1 (int64_t)(w1)[4 * 64] * val; sum2 sign2 (int64_t)(w2)[4 * 64] * val; \
        val = (p)[5 * 64]; sum1 sign1 (int64_t)(w1)[5 * 64] * val; sum2 sign2 (int64_t)(w2)[5 * 64] * val; \
        val = (p)[6 * 64]; sum1 sign1 (int64_t)(w1)[6 * 64] * val; sum2 sign2 (int64_t)(w2)[6 * 64] * val; \
        val = (p)[7 * 64]; sum1 sign1 (int64_t)(w1)[7 * 64] * val; sum2 sign2 (int64_t)(w2)[7 * 64] * val; \
    } while (0)

#define APPLY_WINDOW_LOOP(INCR_STEP)                                          \
    do {                                                                      \
        for (j = 1; j < 16; j++) {                                            \
            const int32_t *p1 = synth_buf + 16 + j;                          \
            const int32_t *p2 = synth_buf + 48 - j;                          \
            sum2 = 0;                                                         \
            DUAL_TAP(sum, sum2, w, w2, p1, +=, -=);                          \
            DUAL_TAP(sum, sum2, w + 32, w2 + 32, p2, -=, -=);                \
            *samples = round_sample_fixed(&sum);                              \
            samples += (INCR_STEP);                                           \
            sum += sum2;                                                      \
            *samples2 = round_sample_fixed(&sum);                             \
            samples2 -= (INCR_STEP);                                          \
            w++;                                                              \
            w2--;                                                             \
        }                                                                     \
    } while (0)

static void ff_mpadsp_apply_window_fixed_xtensa(int32_t *synth_buf,
                                                int32_t *window,
                                                int *dither_state,
                                                int16_t *samples,
                                                ptrdiff_t incr)
{
    const int32_t *w, *w2, *p;
    int16_t *samples2;
    int64_t sum, sum2;
    int j;

    /* Copy to avoid circular wrap */
    memcpy(synth_buf + 512, synth_buf, 32 * sizeof(*synth_buf));

    samples2 = samples + 31 * incr;
    w = window;
    w2 = window + 31;

    /* Sample 0 (j = 0) */
    sum = *dither_state;
    p = synth_buf + 16;
    sum += (int64_t)w[0 * 64] * p[0 * 64]
         + (int64_t)w[1 * 64] * p[1 * 64]
         + (int64_t)w[2 * 64] * p[2 * 64]
         + (int64_t)w[3 * 64] * p[3 * 64]
         + (int64_t)w[4 * 64] * p[4 * 64]
         + (int64_t)w[5 * 64] * p[5 * 64]
         + (int64_t)w[6 * 64] * p[6 * 64]
         + (int64_t)w[7 * 64] * p[7 * 64];

    p = synth_buf + 48;
    sum -= (int64_t)w[32 + 0 * 64] * p[0 * 64]
         + (int64_t)w[32 + 1 * 64] * p[1 * 64]
         + (int64_t)w[32 + 2 * 64] * p[2 * 64]
         + (int64_t)w[32 + 3 * 64] * p[3 * 64]
         + (int64_t)w[32 + 4 * 64] * p[4 * 64]
         + (int64_t)w[32 + 5 * 64] * p[5 * 64]
         + (int64_t)w[32 + 6 * 64] * p[6 * 64]
         + (int64_t)w[32 + 7 * 64] * p[7 * 64];

    *samples = round_sample_fixed(&sum);
    samples += incr;
    w++;

    /* Samples 1..30 (j = 1..15, calculating 2 samples simultaneously) */
    if (incr == 2) {
        APPLY_WINDOW_LOOP(2);
    } else if (incr == 1) {
        APPLY_WINDOW_LOOP(1);
    } else {
        APPLY_WINDOW_LOOP(incr);
    }

    /* Sample 16 (j = 16) */
    p = synth_buf + 32;
    sum -= (int64_t)w[32 + 0 * 64] * p[0 * 64]
         + (int64_t)w[32 + 1 * 64] * p[1 * 64]
         + (int64_t)w[32 + 2 * 64] * p[2 * 64]
         + (int64_t)w[32 + 3 * 64] * p[3 * 64]
         + (int64_t)w[32 + 4 * 64] * p[4 * 64]
         + (int64_t)w[32 + 5 * 64] * p[5 * 64]
         + (int64_t)w[32 + 6 * 64] * p[6 * 64]
         + (int64_t)w[32 + 7 * 64] * p[7 * 64];

    *samples = round_sample_fixed(&sum);
    *dither_state = (int)sum;
}

/* =========================================================================
 * 2. 36-point Inverse MDCT (imdct36_blocks_fixed)
 * ========================================================================= */

#define FIXR(a)        ((int)((a) * (1 << 23) + 0.5))
#define FIXHR(a)       ((int)((a) * (1LL << 32) + 0.5))
#define MULH3(x, y, s) MULH((s) * (x), y)
#define MULLx(x, y, s) MULL((int)(x), (y), s)
#define SHR(a, b)      (((int)(a)) >> (b))

/* cos(pi*i/18) */
#define C1 FIXHR(0.98480775301220805936 / 2)
#define C2 FIXHR(0.93969262078590838405 / 2)
#define C3 FIXHR(0.86602540378443864676 / 2)
#define C4 FIXHR(0.76604444311897803520 / 2)
#define C5 FIXHR(0.64278760968653932632 / 2)
#define C6 FIXHR(0.5 / 2)
#define C7 FIXHR(0.34202014332566873304 / 2)
#define C8 FIXHR(0.17364817766693034885 / 2)

/* 0.5 / cos(pi*(2*i+1)/36) */
static const int32_t icos36[9] = {
    FIXR(0.50190991877167369479),
    FIXR(0.51763809020504152469),
    FIXR(0.55168895948124587824),
    FIXR(0.61038729438072803416),
    FIXR(0.70710678118654752439),
    FIXR(0.87172339781054900991),
    FIXR(1.18310079157624925896),
    FIXR(1.93185165257813657349),
    FIXR(5.73685662283492756461),
};

static const int32_t icos36h[9] = {
    FIXHR(0.50190991877167369479 / 2),
    FIXHR(0.51763809020504152469 / 2),
    FIXHR(0.55168895948124587824 / 2),
    FIXHR(0.61038729438072803416 / 2),
    FIXHR(0.70710678118654752439 / 2),
    FIXHR(0.87172339781054900991 / 2),
    FIXHR(1.18310079157624925896 / 4),
    FIXHR(1.93185165257813657349 / 4),
};

static void imdct36_xtensa(int *out, int *buf, int *in, int *win)
{
    int i, j;
    int t0, t1, t2, t3, s0, s1, s2, s3;
    int tmp[18], *tmp1, *in1;

    for (i = 17; i >= 1; i--)
        in[i] += in[i - 1];
    for (i = 17; i >= 3; i -= 2)
        in[i] += in[i - 2];

    for (j = 0; j < 2; j++) {
        tmp1 = tmp + j;
        in1 = in + j;

        t2 = in1[2 * 4] + in1[2 * 8] - in1[2 * 2];
        t3 = in1[2 * 0] + SHR(in1[2 * 6], 1);
        t1 = in1[2 * 0] - in1[2 * 6];
        tmp1[6] = t1 - SHR(t2, 1);
        tmp1[16] = t1 + t2;

        t0 = MULH3(in1[2 * 2] + in1[2 * 4],    C2, 2);
        t1 = MULH3(in1[2 * 4] - in1[2 * 8], -2 * C8, 1);
        t2 = MULH3(in1[2 * 2] + in1[2 * 8],   -C4, 2);

        tmp1[10] = t3 - t0 - t2;
        tmp1[ 2] = t3 + t0 + t1;
        tmp1[14] = t3 + t2 - t1;

        tmp1[4] = MULH3(in1[2 * 5] + in1[2 * 7] - in1[2 * 1], -C3, 2);
        t2 = MULH3(in1[2 * 1] + in1[2 * 5],    C1, 2);
        t3 = MULH3(in1[2 * 5] - in1[2 * 7], -2 * C7, 1);
        t0 = MULH3(in1[2 * 3], C3, 2);

        t1 = MULH3(in1[2 * 1] + in1[2 * 7],   -C5, 2);

        tmp1[ 0] = t2 + t3 + t0;
        tmp1[12] = t2 + t1 - t0;
        tmp1[ 8] = t3 - t1 - t0;
    }

    i = 0;
    for (j = 0; j < 4; j++) {
        t0 = tmp[i];
        t1 = tmp[i + 2];
        s0 = t1 + t0;
        s2 = t1 - t0;

        t2 = tmp[i + 1];
        t3 = tmp[i + 3];
        s1 = MULH3(t3 + t2, icos36h[j], 2);
        s3 = MULLx(t3 - t2, icos36[8 - j], FRAC_BITS);

        t0 = s0 + s1;
        t1 = s0 - s1;
        out[(9 + j) * SBLIMIT] = MULH3(t1, win[9 + j], 1) + buf[4 * (9 + j)];
        out[(8 - j) * SBLIMIT] = MULH3(t1, win[8 - j], 1) + buf[4 * (8 - j)];
        buf[4 * (9 + j)]       = MULH3(t0, win[MDCT_BUF_SIZE / 2 + 9 + j], 1);
        buf[4 * (8 - j)]       = MULH3(t0, win[MDCT_BUF_SIZE / 2 + 8 - j], 1);

        t0 = s2 + s3;
        t1 = s2 - s3;
        out[(9 + 8 - j) * SBLIMIT] = MULH3(t1, win[9 + 8 - j], 1) + buf[4 * (9 + 8 - j)];
        out[j * SBLIMIT]           = MULH3(t1, win[j], 1)         + buf[4 * j];
        buf[4 * (9 + 8 - j)]       = MULH3(t0, win[MDCT_BUF_SIZE / 2 + 9 + 8 - j], 1);
        buf[4 * j]                 = MULH3(t0, win[MDCT_BUF_SIZE / 2 + j], 1);
        i += 4;
    }

    s0 = tmp[16];
    s1 = MULH3(tmp[17], icos36h[4], 2);
    t0 = s0 + s1;
    t1 = s0 - s1;
    out[(9 + 4) * SBLIMIT] = MULH3(t1, win[9 + 4], 1) + buf[4 * (9 + 4)];
    out[(8 - 4) * SBLIMIT] = MULH3(t1, win[8 - 4], 1) + buf[4 * (8 - 4)];
    buf[4 * (9 + 4)]       = MULH3(t0, win[MDCT_BUF_SIZE / 2 + 9 + 4], 1);
    buf[4 * (8 - 4)]       = MULH3(t0, win[MDCT_BUF_SIZE / 2 + 8 - 4], 1);
}

static void ff_imdct36_blocks_fixed_xtensa(int *out, int *buf, int *in,
                                           int count, int switch_point, int block_type)
{
    int j;
    for (j = 0; j < count; j++) {
        int win_idx = (switch_point && j < 2) ? 0 : block_type;
        int *win = ff_mdct_win_fixed[win_idx + (4 & -(j & 1))];

        imdct36_xtensa(out, buf, in, win);

        in  += 18;
        buf += ((j & 3) != 3 ? 1 : (72 - 3));
        out++;
    }
}

/* =========================================================================
 * 3. Architecture Dispatch Init Hook
 * ========================================================================= */

av_cold void ff_mpadsp_init_xtensa(MPADSPContext *s)
{
    s->apply_window_fixed = ff_mpadsp_apply_window_fixed_xtensa;
    s->imdct36_blocks_fixed = ff_imdct36_blocks_fixed_xtensa;
}
