#include "gpuarray/reduction.h"

#include "util/error.h"

struct _GpuReduction {
  const char *expr; /* reduction expression */
  const char *init_val; /* initial (neutral) value for the expression */
  const char *preamble; /* preamble code */
  char *redux;
  int input_dtype; /* type of input buffer */
  int work_dtype; /* type of reduction buffer */
  int output_dtype; /* type of output buffer */
  unsigned int nd; /* number of dimensions of input */
};

static int gen_reduction_basic_kernel(GpuKernel *k, gpucontext *ctx,
                                      char **err_str,
                                      const char *preamble,
                                      const char *map_expr,
                                      const char *expr,
                                      const char *init_val,
                                      unsigned int nd,
                                      char *redux,
                                      int input_dtype,
                                      int work_dtype,
                                      int output_dtype) {
  strb sb = STRB_STATIC_INIT;
  unsigned int i, _i, j;
  int *ktypes;
  unsigned int p;
  int flags = GA_USE_CLUDA;
  int res;

  p = 1 + 6 + nd * 4;
  ktypes = calloc(p, sizeof(int));
  if (ktypes == NULL)
    return error_sys(ctx->err, "calloc");

  p = 0;

  if (preamble)
    strb_appends(&sb, preamble);
  if (map_expr != NULL)
    strb_appendf(&sb, "\n#define PRE(a) (%s)\n", map_expr);
  else
    strb_appends(&sb, "\n#define PRE(a) (a)\n");
  strb_appendf(&sb, "#define REDUCE(a, b) (%s)\n", expr);
  strb_appends(&sb, "\nKERNEL void redux(const ga_size n");
  ktypes[p++] = GA_SIZE;
  for (i = 0; i < nd; i++) {
    strb_appendf(&sb, ", const ga_size dim%u", i);
    ktypes[p++] = GA_SIZE;
  }
  strb_appendf(&sb, ", GLOBAL_MEM %s * out, const ga_size out_offset",
               ctype(output_dtype));
  ktypes[p++] = GA_BUFFER;
  ktypes[p++] = GA_SIZE;
  for (i = 0; i < nd; i++) {
    strb_appendf(&sb, ", const ga_ssize out_str_%u", i);
    ktypes[p++] = GA_SSIZE;
  }
  strb_appendf(&sb, ", GLOBAL_MEM %s * inp, const ga_size inp_offset",
               ctype(input_dtype));
  ktypes[p++] = GA_BUFFER;
  ktypes[p++] = GA_SIZE;
  for (i = 0; i < nd; i++) {
    strb_appendf(&sb, ", const ga_ssize inp_str_%u", i);
    ktypes[p++] = GA_SSIZE;
  }
  strb_appendf(&sb, "GA_DECL_SHARED_PARAM(%s, ldata)", ctype(work_dtype));
  strb_appendf(&sb, ") {\n"
               "GA_DECL_SHARED_BODY(%s, ldata)\n"
               "const ga_size outIdx = GID_0 * GDIM_1 + GID_1;"
               "const ga_size idx = LID_0;\n"
               "ga_size i;\n", ctype(work_dtype));

  /* Prepare index of the non-reduced dimensions */
  strb_appends(&sb, "i = outIdx;\n");
  j = 0;
  for (_i = nd; _i > 0; _i--) {
    i = _i - 1;
    if (!redux[i]) {
      j++; /* For the indexing of the output at the end */
      strb_appendf(&sb, "const ga_size pos%u = i %% dim%u;\n", i, i);
    }
  }

  strb_appendf(&sb, "%s acc = %s;\n", ctype(work_dtype), init_val);
  strb_appends(&sb, "for (i = idx; i < n; i += LDIM_0) {\n"
               "ga_size ii = i;\nga_size pos;\n"
               "ga_size inp_p = inp_offset;\n");

  /* Do indexing on input */
  for (_i = nd; _i > 0; _i--) {
    i = _i - 1;
    if (redux[i]) {
      if (i > 0)
        strb_appendf(&sb, "pos = ii %% dim%u;\nii /= dim%u;\n", i, i);
      else
        strb_appends(&sb, "pos = ii;\n");
      strb_appendf(&sb, "inp_p += pos * inp_str_%u;\n", i);
    } else {
      strb_appendf(&sb, "inp_p += pos%u * inp_str_%u;\n", i, i);
    }
  }
  strb_appendf(&sb, "%s inp = *(GLOBAL_MEM %s *)(((GLOBAL_MEM char *)inp_data) + inp_p)", ctype(input_dtype), ctype(input_dtype));

  /* Finalize the reduction */
  strb_appends(&sb, "acc = REDUCE((acc), (PRE(inp)));\n"
               "}\n"
               "ldata[LID_0] = acc;\n"
               "ga_size cur_size = LDIM_0;\n"
               "while (cur_size > 1) {\n"
               "local_barrier();\n"
               "if (LID_0 < cur_size) {\n"
               "ldata[LID_0] = REDUCE(ldata[LID_0], ldata[LID_0 + cur_size]);\n"
               "}\n}\n"
               "local_barrier();\n");

  /* Store the result in the output buffer */
  strb_appends(&sb, "if (LID_0 == 0) {\n"
               "ga_size out_p = out_offset;\n");
  for (_i = nd; _i > 0; _i--) {
    i = _i - 1;
    if (!redux[i]) {
      strb_appendf(&sb, "out_p += pos%u * out_str_%u;\n", i, j);
      j--;  /* j is initialized in the loop that creates pos%u above */

    }
  }
  strb_appendf(&sb, "*(GLOBAL_MEM %s *)(((GLOBAL_MEM char *)out_data) + out_p) = ldata[0];\n", ctype(output_dtype));
  strb_appends(&sb, "}\n}\n");
  /* Kernel complete */

  /* TODO: HERE */
}

int GpuReduction_new(GpuReduction **gr, gpucontext *ctx,
                     const char *preamble, const char *expr,
                     const char *init_val, int typecode,
                     unsigned int nd, int axis, int flags) {
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
  if (axes == NULL) {
    size_t i;
    res->axes = calloc(sizeof(unsigned int), nd);
    if (res->axes != NULL)
      for (i = 0; i < nd; i++)
        res->axes[i] = i;
  } else {
    res->axes = memdup(axes, axes_len * sizeof(unsigned int));
  }
  if (res->axes == NULL) {
    free(res);
    return error_sys(ctx->err, "calloc/memdup");
  }
  res->input_dtype = typecode;
  res->work_dtype = typecode;
  res->output_dtype = typecode;
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

int GpuReduction_call(GpuReduction *gr, GpuArray *input, GpuArray *output) {

}
