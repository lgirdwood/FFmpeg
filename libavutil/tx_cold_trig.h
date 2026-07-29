#ifndef AVUTIL_TX_COLD_TRIG_H
#define AVUTIL_TX_COLD_TRIG_H
/* SOF: hardcoded cosine master table + exact grid-snap lookups that replace the
 * soft-float double cos()/sin() calls in av_tx COLD twiddle generation
 * (SR power-of-two tables, MDCT/RDFT exptabs, mdct_gen_exp).  Every argument in
 * those av_cold init paths is an exact integer multiple of 2*pi/16384 for the
 * power-of-two sizes AAC uses (1024/128), so the lround() snap is bit-faithful
 * (in fact more accurate than the target soft-float cos()).  Hot per-frame
 * transforms and the non-power-of-two 5/7/9 leaf tables are NOT routed here. */
#include <math.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#ifndef M_PI_2
#define M_PI_2 1.57079632679489661923
#endif
#define SOF_TX_TRIG_MASTER 16384
extern const double ff_sof_tx_master_cos[SOF_TX_TRIG_MASTER/4 + 1];
static inline double ff_sof_tx_cos(double x)
{
    /* HW single-precision reduction: no soft-float double mul / lround.  Cold-init
     * arguments are all small (<~pi/2), so (float)x * C stays integer-exact and
     * lroundf() yields the identical grid index as the former lround(double). */
    long k = lroundf((float)x * (float)(SOF_TX_TRIG_MASTER / (2.0 * M_PI)));
    unsigned m = (unsigned)(((k % SOF_TX_TRIG_MASTER) + SOF_TX_TRIG_MASTER) % SOF_TX_TRIG_MASTER);
    if (m <= 4096)  return  ff_sof_tx_master_cos[m];
    if (m <= 8192)  return -ff_sof_tx_master_cos[8192 - m];
    if (m <= 12288) return -ff_sof_tx_master_cos[m - 8192];
    return                  ff_sof_tx_master_cos[SOF_TX_TRIG_MASTER - m];
}
static inline double ff_sof_tx_sin(double x)
{
    return ff_sof_tx_cos(x - M_PI_2);
}
/* HW single-precision variants for the float-transform cold table generators.
 * Cold-init arguments are all small (<~pi/2), so (float)x * C is integer-exact
 * and lroundf() yields the identical grid index as the double path. The only
 * double touched is the master-table read, cast once to float per call. */
static inline float ff_sof_tx_cosf(float x)
{
    long k = lroundf(x * (float)(SOF_TX_TRIG_MASTER / (2.0 * M_PI)));
    unsigned m = (unsigned)(((k % SOF_TX_TRIG_MASTER) + SOF_TX_TRIG_MASTER) % SOF_TX_TRIG_MASTER);
    if (m <= 4096)  return  (float)ff_sof_tx_master_cos[m];
    if (m <= 8192)  return -(float)ff_sof_tx_master_cos[8192 - m];
    if (m <= 12288) return -(float)ff_sof_tx_master_cos[m - 8192];
    return                  (float)ff_sof_tx_master_cos[SOF_TX_TRIG_MASTER - m];
}
static inline float ff_sof_tx_sinf(float x)
{
    return ff_sof_tx_cosf(x - (float)M_PI_2);
}
#endif /* AVUTIL_TX_COLD_TRIG_H */
