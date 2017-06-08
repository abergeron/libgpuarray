#ifndef GPUARRAY_REDUCTION_H
#define GPUARRAY_REDUCTION_H
/** \file reduction.h
 * \brief Custom reduction operations generator
 */

#include <gpuarray/buffer.h>
#include <gpuarray/elemwise.h>

#ifdef __cplusplus
extern "C" {
#endif
#ifdef CONFUSE_EMACS
}
#endif

struct _GpuReduction;

/**
 * Reduction generator structure.
 *
 * The contents are private.
 */
typedef struct _GpuReduction GpuReduction;

/**
 * Create a new GpuReduction.
 *
 * \param gr the new GpuReduction object
 * \param ctx context to create in
 * \param preamble kernel preamble (can be NULL)
 * \param expr reduction expression (using a and b as inputs)
 * \param init_val value that would be neutral for the reduction
 * \param typecode type of the array to reduce (and the result)
 * \param nd number of dimensions of input
 * \param axes axes to keep on (if NULL reduce on all axes)
 * \param flags Must be 0
 *
 * \return GA_NO_ERROR if the operation was successful
 * \return an error code otherwise
 */
GPUARRAY_PUBLIC int GpuReduction_new(GpuReduction **gr,
                                     gpucontext *ctx,
                                     const char *preamble,
                                     const char *expr,
                                     const char *init_val,
                                     int typecode,
                                     unsigned int nd,
                                     unsigned char *axes,
                                     int flags);

/**
 * Free all storage associated with a GpuReduction.
 *
 * \param gr the GpuReduction object to free.
 */
GPUARRAY_PUBLIC void GpuReduction_free(GpuReduction *gr);
#endif
