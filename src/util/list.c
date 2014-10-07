#include <assert.h>

#include "util/list.h"
#include "util/halloc.h"
#include "util/debug.h"

#include "private.h"


struct _list {
  void **values;
  size_t size;
  size_t alloc;
};

list *list_new(size_t a) {
  list *l = h_malloc(sizeof(*l));
  if (l == NULL) {
    DPRINT("failed to alloc list struct\n");
    return NULL;
  }
  if (a < 8) a = 8;
  l->values = h_calloc(a, sizeof(void *));
  if (l->values == NULL) {
    DPRINT("failed to alloc list backing for %llu elems\n", (unsigned long long)a);
    return NULL;
  }
  hattach(l->values, l);
  l->size = 0;
  l->alloc = a;
  return l;
}

static int list_grow(list *l, size_t n) {
  if (l->alloc == 0 && n < 16) n = 16;
  if (l->alloc > n) n = l->alloc;
  if (SIZE_MAX - l->alloc < n) return -1;
  if (h_reallocarray(l->values, l->alloc+n, sizeof(void *)))
    return -1;
  l->alloc += n;
  return 0;
}

static int list_ensure(list *l, size_t s) {
  if (l->alloc - l->size < s) return list_grow(l, s);
  return 0;
}

void *_list_get(list *l, size_t i, const char *name) {
  if (i >= l->size) {
    DPRINT("list index %llu too big for %s\n", (unsigned long long)i, name);
    return NULL;
  }
  return l->values[i];
}

void list_set(list *l, size_t i, void *val) {
  assert(i < l->size);
  hattach(l->values[i], NULL);
  l->values[i] = val;
  hattach(val, l);
}

int list_append(list *l, void *val) {
  if (list_ensure(l, 1)) return -1;
  l->values[l->size++] = val;
  hattach(val, l);
  return 0;
}

int list_insert(list *l, size_t i, void *val) {
  assert(i < l->size);
  if (list_ensure(l, 1)) return -1;
  memmove(&l->values[i+1], &l->values[i], sizeof(void *) * (l->size - i));
  l->values[i] = val;
  hattach(val, l);
  l->size++;
  return 0;
}

size_t list_size(list *l) {
  return l->size;
}
