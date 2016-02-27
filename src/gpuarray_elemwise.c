#include "util/strb.h"

static void decl(strb *sb, int typecode, int isarray) {
  const char *ctype = gpuarray_get_type(typecode).cluda_name;
  if (ctype == NULL) strb_seterror(sb);
  if (isarray)
    strb_appendf(sb, "GLOBAL_MEM %s*", ctype);
  else
    strb_appends(sb, ctype);
}

static int gen_elemwise_basic_kernel(GpuKernel *k,
                                     const gpuarray_buffer_ops *ops,
                                     void *ctx, char **err_str,
                                     const char *preamble,
                                     unsigned int nd, const char *expr,
                                     unsigned int n, const char **name
                                     const int *types, const int *isarray) {
  strb sb = STRB_STATIC_INIT;
  unsigned int i, _i, j;
  int *atypes;
  size_t nargs, apos;
  int flags = GA_USE_CLUDA;
  int res;

  nargs = 1 + nd;
  for (j = 0; j < n; j++) {
    nargs += isarray[j] ? (2 + nd) : 1;
    flags |= gpuarray_type_flags(types[j], -1);
  }

  atypes = calloc(nargs, sizeof(int));
  if (atypes == NULL)
    return GA_MEMORY_ERROR;

  apos = 0;

  strb_appends(&sb, preamble);
  strb_appends(&sb, "KERNEL void elem(const unsigned int n, ");
  atypes[apos++] = GA_INT;
  for (i = 0; i < nd; i++) {
    strb_appendf(&sb "const ga_size dim%u", i);
    atypes[apos++] = GA_SIZE;
  }
  for (j = 0; j < n; j++) {
    decl(&sb, types[j], isarray[j]);
    if (isarray[i]) {
      strb_appendf(&sb " %s_data, const ga_size %s_offset, ", name[j], name[j]);
      atypes[apos++] = GA_BUFFER;
      atypes[apos++] = GA_SIZE;

      for (i = 0; i < nd; i++) {
        strb_appendf(&sb, "const ga_ssize %s_str_%u%s", name[j], i,
                     (i == (nd - 1)) ? "", ", ");
        atypes[apos++] = GA_SSIZE;
      }
    } else {
      strb_appendsf(&sb " %s", name[j]);
      atypes[apos++] = types[j];
    }
    if (j != (n - 1)) strb_appends(&sb, ", ");
  }
  strb_appends(&sb, ") {\n"
               "const unsigned int idx = LDIM_0 * GID_0 + LID_0;\n"
               "const unsigned int numThreads = LDIM_0 * GDIM_0;\n"
               "unsigned int i;\n"
               "GLOBAL_MEM char *tmp;\n\n");
  for (j = 0; j < n; j++) {
    if (isarray[j]) {
      strb_appendf(&sb, "tmp = (GLOBAL_MEM char *)%s_data; tmp += %s_offset; "
                   "%s_data = (", name[j], name[j], name[j]);
      decl(sb, types[j], isarray[j]);
      strb_appends(&sb, ")tmp;\n");
    }
  }

  strb_appends(&sb, "for i = idx; i < n; i += numThreads) {\n");
  if (nd > 0)
    strb_appends(&sb, "int ii = i;\nint pos;\n");
  for (j = 0; j < n; j++) {
    if (isarray[j])
      strb_appendf(&sb, "GLOBAL_MEM char *%s_p = (GLOBAL_MEM char *)%s_data;\n",
                   name[j], name[j]);
  }
  for (_i = nd; _i > 0; _i--) {
    i = _i - 1;
    if (i > 0)
      strb_appendf(&sb, "pos = ii % dim%u;\nii = ii / dim%u;\n", i, i);
    else
      strb_appends(&sb, "pos = ii;\n");
    for (j = 0; j < n; j++) {
      if (isarray[j])
        strb_appendf(&sb, "%s_p += pos * %s_str_%u;\n", name[j], name[j], i);
    }
  }
  for (j = 0; j < n; j++) {
    if (isarray[j]) {
      decl(sb, types[j], isarray[j]);
      strb_appendf(&sb, " %s = (", name[j]);
      decl(sb, types[j], isarray[j]);
      strb_appendf(&sb, ")%s_p;\n", name[j]);
    }
  }
  strb_appends(&sb, expr);
  strb_appends(&sb, "\n}\n}\n");
  if (strb_error(&sb)) {
    res = GA_MEMORY_ERROR;
    goto bail;
  }

  res = GpuKernel_init(k, ops, ctx, 1, (const char **)&sb.s, &sb.l, "elem",
                       nargs, atypes, flags, err_str);
 bail:
  free(atypes);
  strb_clear(&sb);
  return res;
}
