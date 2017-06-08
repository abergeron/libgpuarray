#include "gpuarray/reduction.h"

#include "util/error.h"

struct _GpuReduction {
  const char *expr; /* reduction expression */
  const char *init_val; /* initial (neutral) value for the expression */
  const char *preamble; /* preamble code */
  unsigned char *axes;
  int input_type; /* type of input buffer */
  int work_type; /* type of reduction buffer */
  int output_type; /* type of output buffer */
  unsigned int nd; /* number of dimensions of input */
};

int GpuReduction_new(GpuReduction **gr, gpucontext *ctx,
                     const char *preamble, const char *expr,
                     const char *init_val, int typecode,
                     unsigned int nd, unsigned char *axes, int flags) {
  GpuElemwise *res;
  if (ctx == NULL)
    return error_set(global_err, GA_VALUE_ERROR, "context is NULL");
  if (gr == NULL)
    return error_set(ctx->err, GA_INVALID_ERROR, "Result pointer is NULL");
  if (flags != 0)
    return error_set(ctx->err, GA_VALUE_ERROR, "flags is not 0");
  if (expr == NULL)
    return error_set(ctx->err, GA_VALUE_ERROR, "expr is NULL");
  if (init_val == NULL)
    return error_set(ctx->err, GA_VALUE_ERROR, "init_val is NULL");
  if (nd == 0)
    return error_set(ctx->err, GA_VALUE_ERROR, "Can't reduce a 0-d object");
  if (gpuarray_get_type(typecode)->typecode != typecode)
    return error_set(ctx->err, GA_VALUE_ERROR, "Invalid typecode");

  res = calloc(sizeof(*res), 1);
  if (res == NULL)
    return error_sys(ctx->err, "calloc");
  res->expr = expr;
  res->init_val = init_val;
  res->preamble = preamble;
  if (axes == NULL)
    res->axes = calloc(sizeof(unsigned char), nd);
  else
    res->axes = memdup(axes, nd * sizeof(unsigned char));
  if (res->axes == NULL) {
    free(res);
    return error_sys(ctx->err, "calloc/memdup");
  }
  res->input_type = typecode;
  res->work_type = typecode;
  res->output_type = typecode;
  res->nd = nd;

  *gr = res;
  return GA_NO_ERROR;
}

void GpuReduction_free(GpuReduction *gr) {
  if (gr) {
    free(gr->axes);
  }
  free(gr);
}
