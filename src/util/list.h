#ifndef UTIL_LIST_H
#define UTIL_LIST_H

#include <gpuarray/config.h>

typedef struct _list list;

GPUARRAY_LOCAL list *list_new(size_t a);

#define list_get(a, b) _list_get(a, b, #a)
GPUARRAY_LOCAL void *_list_get(list *l, size_t i, const char *name);
GPUARRAY_LOCAL void list_set(list *l, size_t i, void *val);

GPUARRAY_LOCAL int list_append(list *l, void *val);
GPUARRAY_LOCAL int list_insert(list *l, size_t i, void *val);

GPUARRAY_LOCAL size_t list_size(list *l);

#endif
