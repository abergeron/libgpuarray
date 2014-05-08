/* Null cache module that doesn't do anything expect fill the linking
   requirements */

#include "gpuarray/error.h"

int cache_get(const char *kind, const char *devname, const char *code,
              void **bin, size_t *sz) {
  return GA_MISC_ERROR;
}

int cache_put(const char *kind, const char *devname, const char *code,
              void *bin, size_t sz) {
  return GA_MISC_ERROR;
}
