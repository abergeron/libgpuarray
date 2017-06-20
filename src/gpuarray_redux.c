#include "gpuarray/reduction.h"

#include "gpuarray/kernel.h"

#include "util/error.h"

#include "private.h"

#include <assert.h>

struct _GpuReduction {
  const char *expr; /* reduction expression */
  const char *map_expr; /* pre scalar expression */
  const char *init_val; /* initial (neutral) value for the expression */
  const char *preamble; /* preamble code */
  GpuKernel *knd; /* array of kernels */
  size_t *kls; /* local and shared memory sizes for the kernels above */
  size_t *rdims; /* pre-allocated reduction dims for collapsing */
  ssize_t *rstrs; /* pre-allocated reduction strides for collapsing */
  size_t *odims; /* pre-allocated reduction dims for collapsing */
  ssize_t *ostrs[2]; /* pre-allocated reduction strides for collapsing */
  unsigned int nd; /* allocated size of variable length fields */
  int input_dtype; /* type of input buffer */
  int work_dtype; /* type of reduction buffer */
  int output_dtype; /* type of output buffer */
};

static inline int k_initialized(GpuKernel *k) {
  return k->k != NULL;
}

static inline const char *ctype(int typecode) {
  return gpuarray_get_type(typecode)->cluda_name;
}

static inline int redux_ok(uint32_t redux, unsigned int i) {
  return redux & (1U << i);
}

/* Size of the array of kernels to store the possibilities */
static inline unsigned int ndsz(unsigned int n) {
  assert(n <= 32);
  return n * (n + 1) / 2;
}

static int gen_reduction_basic_kernel(GpuKernel *k, gpucontext *ctx,
                                      char **err_str,
                                      const char *preamble,
                                      const char *map_expr,
                                      const char *expr,
                                      const char *init_val,
                                      unsigned int nd,
                                      uint32_t redux,
                                      int input_dtype,
                                      int work_dtype,
                                      int output_dtype) {
  strb sb = STRB_STATIC_INIT;
  unsigned int i, _i, j;
  int *ktypes;
  unsigned int p;
  int flags = GA_USE_CLUDA;
  int res;

  flags |= gpuarray_type_flags(input_dtype, work_dtype, output_dtype, -1);

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
    if (!redux_ok(redux, i)) {
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
    if (redux_ok(redux, i)) {
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
    if (!redux_ok(redux, i)) {
      strb_appendf(&sb, "out_p += pos%u * out_str_%u;\n", i, j);
      j--;  /* j is initialized in the loop that creates pos%u above */

    }
  }
  strb_appendf(&sb, "*(GLOBAL_MEM %s *)(((GLOBAL_MEM char *)out_data) + out_p) = ldata[0];\n", ctype(output_dtype));
  strb_appends(&sb, "}\n}\n");
  /* Kernel complete */

  if (strb_error(&sb)) {
    res = error_sys(ctx->err, "strb");
    goto bail;
  }

  res = GpuKernel_init(k, ctx, 1, (const char **)&sb.s, &sb.l, "redux",
                       p, ktypes, flags, err_str);

 bail:
  free(ktypes);
  strb_clear(&sb);
  return res;
}

#define MUL_NO_OVERFLOW ((size_t)1 << (sizeof(size_t) * 4))

static int reallocaz(void **p, size_t elsz, size_t old, size_t new) {
  char *res;

  assert(old <= new);

  if ((new >= MUL_NO_OVERFLOW || elsz >= MUL_NO_OVERFLOW) &&
      new > 0 && SIZE_MAX / new < elsz) {
    return 1;
  }
  res = realloc(*p, elsz*new);
  if (res == NULL) return 1;
  memset(res + (elsz*old), 0, elsz*(new-old));
  *p = (void *)res;
  return 0;
}

static int gr_grow(GpuReduction *gr, unsigned int nd, gpucontext *ctx) {
  assert(nd < gr->nd);

  if (reallocaz((void **)&gr->rdims, sizeof(size_t), gr->nd*2, nd*2) ||
      reallocaz((void **)&gr->rstrs, sizeof(ssize_t), gr->nd*2, nd*2) ||
      reallocaz((void **)&gr->odims, sizeof(size_t), gr->nd*2, nd*2) ||
      reallocaz((void **)&gr->ostrs[0], sizeof(ssize_t), gr->nd*2, nd*2) ||
      reallocaz((void **)&gr->ostrs[1], sizeof(ssize_t), gr->nd*2, nd*2) ||
      reallocaz((void **)&gr->knd, sizeof(GpuKernel), ndsz(gr->nd), ndsz(nd)))
    return error_sys(ctx->err, "reallocaz");
  gr->nd = nd;
  return GA_NO_ERROR;
}

static int get_kernel_nd(GpuKernel **k, gpucontext *ctx, GpuReduction *gr,
                         unsigned int nd, unsigned int rnd) {
  uint32_t redux = (1U << rnd) - 1;
  char *err_str = NULL;
  unsigned int kidx = ndsz(nd) + rnd;
  int err = GA_NO_ERROR;

  if (!k_initialized(&gr->knd[kidx])) {
    err = gen_reduction_basic_kernel(&gr->knd[ndsz(nd) + rnd], ctx, &err_str,
                                     gr->preamble, gr->map_expr, gr->expr,
                                     gr->init_val, nd, redux,
                                     gr->input_dtype, gr->work_dtype,
                                     gr->output_dtype);
    if (err_str) {
      fprintf(stderr, "GpuReduction kernel error:\n%s\n", err_str);
      free(err_str);
    }
  }

  if (err == GA_NO_ERROR)
    *k = &gr->knd[kidx];
  return err;
}

static int do_schedule(GpuReduction *gr, GpuKernel *k,
                       unsigned int rnd, unsigned int ond,
                       size_t *gs, size_t *ls, size_t *shared) {
  size_t nr;
  size_t no;
  size_t maxl0;
  size_t maxl;
  size_t prefl;
  size_t maxg0;
  size_t maxg1;
  size_t lsz;
  size_t esz;
  unsigned int i;

  nr = 1;
  for (i = 0; i < rnd; i++) nr *= gr->rdims[i];

  no = 1;
  for (i = 0; i < ond; i++) no *= gr->odims[i];

  GA_CHECK(gpukernel_property(k->k, GA_CTX_PROP_MAXGSIZE0, &maxg0));
  GA_CHECK(gpukernel_property(k->k, GA_CTX_PROP_MAXGSIZE1, &maxg1));
  GA_CHECK(gpukernel_property(k->k, GA_KERNEL_PROP_MAXLSIZE, &maxl));
  GA_CHECK(gpukernel_property(k->k, GA_KERNEL_PROP_PREFLSIZE, &prefl));
  GA_CHECK(gpukernel_property(k->k, GA_CTX_PROP_MAXLSIZE0, &maxl0));
  GA_CHECK(gpukernel_property(k->k, GA_CTX_PROP_LMEMSIZE, &lsz));
  esz = gpuarray_get_elsize(gr->work_dtype);

  /**** Choose local size ****/
  maxl = (maxl > maxl0) ? maxl0 : maxl;

  /* If there is not enough space in local memory for a full maxl
   * threadblock, we compute the max that would fit in local memory
   * given the itemsize */
  if ((maxl * esz) > lsz)
    maxl = lsz / esz;

  /* Round down to closest multiple of prefl */
  ls[0] = (maxl / prefl) * prefl;
  ls[1] = 1;

  /**** Chose global size ****/
  /* Cheap out for now, will fix later for bigger arrays */
  if (no > maxg0)
    return error_set(GpuKernel_context(k)->err, GA_UNSUPPORTED_ERROR,
                     "Reduction output is too large to handle");
  gs[0] = no;
  gs[1] = 1;

  *shared = ls[0] * esz;

  return GA_NO_ERROR;
}

static int do_call(GpuReduction *gr, GpuKernel *k,
                   unsigned int rnd, unsigned int ond,
                   gpudata *input, size_t ioff,
                   gpudata *output, size_t ooff,
                   size_t *gs, size_t *ls, size_t shared) {
  size_t nr;
  unsigned int p;
  unsigned int i;

  nr = 1;
  for (i = 0; i < rnd; i++) nr *= gr->rdims[i];

  p = 0;
  GA_CHECK(GpuKernel_setarg(k, p++, &nr));
  for (i = 0; i < rnd; i++)
    GA_CHECK(GpuKernel_setarg(k, p++, &gr->rdims[i]));
  for (i = 0; i < ond; i++)
    GA_CHECK(GpuKernel_setarg(k, p++, &gr->odims[i]));
  GA_CHECK(GpuKernel_setarg(k, p++, output));
  GA_CHECK(GpuKernel_setarg(k, p++, &ooff));
  for (i = 0; i < ond; i++)
    GA_CHECK(GpuKernel_setarg(k, p++, &gr->ostrs[1][i]));
  GA_CHECK(GpuKernel_setarg(k, p++, input));
  GA_CHECK(GpuKernel_setarg(k, p++, &ioff));
  for (i = 0; i < ond; i++)
    GA_CHECK(GpuKernel_setarg(k, p++, &gr->rstrs[i]));
  for (i = 0; i < ond; i++)
    GA_CHECK(GpuKernel_setarg(k, p++, &gr->ostrs[0][i]));


  return GpuKernel_call(k, 2, gs, ls, shared, NULL);
}


int GpuReduction_new(GpuReduction **gr, gpucontext *ctx,
                     const char *preamble, const char *expr,
                     const char *init_val, int typecode,
                     unsigned int init_nd, int flags) {
  GpuReduction *res;
  GpuKernel *k;
  unsigned int i, j;
  int err;

  if (ctx == NULL)
    return error_set(global_err, GA_VALUE_ERROR, "context is NULL");
  if (gr == NULL)
    return error_set(ctx->err, GA_INVALID_ERROR, "Result pointer is NULL");
  if (flags != 0)
    return error_set(ctx->err, GA_VALUE_ERROR, "flags is not 0");
  if (expr == NULL)
    return error_set(ctx->err, GA_VALUE_ERROR, "expr is NULL");
  if (init_nd > 32)
    return error_set(ctx->err, GA_UNSUPPORTED_ERROR, "Reduction supports 32 dimensions maximum");
  if (init_val == NULL)
    return error_set(ctx->err, GA_VALUE_ERROR, "init_val is NULL");
  if (gpuarray_get_type(typecode)->typecode != typecode)
    return error_set(ctx->err, GA_VALUE_ERROR, "Invalid typecode");

  if (init_nd < 2) init_nd = 2;

  res = calloc(sizeof(*res), 1);
  if (res == NULL)
    return error_sys(ctx->err, "calloc");
  res->expr = expr;
  res->map_expr = NULL;
  res->init_val = init_val;
  res->preamble = preamble;
  res->input_dtype = typecode;
  res->work_dtype = typecode;
  res->output_dtype = typecode;

  res->nd = 0;
  res->knd = NULL;
  res->rdims = NULL;
  res->rstrs = NULL;

  err = gr_grow(res, init_nd, ctx);
  if (err != GA_NO_ERROR) {
    GpuReduction_free(res);
    return err;
  }

  /* Initialize the kernels according to nd */
  for (i = 0; i < res->nd; i++) {
    for (j = 1; j <= i; j++) {
      err = get_kernel_nd(&k, ctx, res, i, j);
      if (err != GA_NO_ERROR) {
        GpuReduction_free(res);
        return err;
      }
    }
  }

  *gr = res;
  return GA_NO_ERROR;
}

void GpuReduction_free(GpuReduction *gr) {
  if (gr) {
    unsigned int i;
    for (i = 0; i < gr->nd; i++) {
      if (k_initialized(&gr->knd[i]))
        GpuKernel_clear(&gr->knd[i]);
    }
    free(gr->knd);
    free(gr->rstrs);
    free(gr->rdims);
  }
  free(gr);
}

int GpuReduction_call(GpuReduction *gr, GpuArray *input, uint32_t redux,
                      GpuArray *output) {
  gpucontext *ctx = GpuKernel_context(&gr->knd[0]);
  GpuKernel *k;
  size_t nprocs;
  size_t gs[2];
  size_t ls[2];
  size_t shared;
  unsigned int i, rnd, ond;

  if (input->nd > 32)
    return error_set(ctx->err, GA_UNSUPPORTED_ERROR, "Reduction supports 32 dimensions maximum");

  if (input->nd > gr->nd)
    GA_CHECK(gr_grow(gr, input->nd, ctx));

  rnd = 0;
  ond = 0;
  for (i = 0; i < input->nd; i++) {
    if (redux_ok(redux, i)) {
      gr->rdims[rnd] = input->dimensions[i];
      gr->rstrs[rnd] = input->strides[i];
      rnd++;
    } else {
      gr->odims[ond] = input->dimensions[i];
      gr->ostrs[0][ond] = input->strides[i];
      ond++;
    }
  }

  if (output->nd != ond)
    return error_fmt(ctx->err, GA_VALUE_ERROR, "Expected %u dims on output, got %u", ond, output->nd);
  for (i = 0; i < ond; i++) {
    if (gr->odims[i] != output->dimensions[i])
      return error_fmt(ctx->err, GA_VALUE_ERROR, "Expected size %u for dim %u, got %u", gr->odims[i], i, output->dimensions[i]);
    gr->ostrs[1][i] = output->strides[i];
  }

  if (rnd > 1)
    gpuarray_elemwise_collapse(1, &rnd, gr->rdims, &(gr->rstrs));
  if (ond > 1)
    gpuarray_elemwise_collapse(2, &ond, gr->odims, gr->ostrs);

  GA_CHECK(get_kernel_nd(&k, ctx, gr, rnd+ond, rnd));

  GA_CHECK(gpukernel_property(k->k, GA_CTX_PROP_NUMPROCS, &nprocs));
  GA_CHECK(do_schedule(gr, k, rnd, ond,
                       gs, ls, &shared));

  if ((gs[0] * gs[1]) < nprocs) {
    // Do 2 stage call later
  }

  return do_call(gr, k, rnd, ond,
                 input->data, input->offset,
                 output->data, output->offset,
                 gs, ls, shared);
}
