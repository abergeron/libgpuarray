#ifndef UTIL_HASH_H
#define UTIL_HASH_H

#include <gpuarray/config.h>

typedef struct _hash hash;

/*
 * This hash set supports NULL as value but it is also the marker for
 * 'not found' used by hash_find() and hash_del().
 */

GPUARRAY_LOCAL hash *hash_new(uint32_t size);

GPUARRAY_LOCAL void *hash_find(hash *h, const char *key);

/*
 * hash_add does not check for duplicate keys and a new add with the
 * same key will shadow an old one until hash_del where the old value
 * will be visible again.
 */
GPUARRAY_LOCAL int hash_add(hash *h, const char *key, void *val);
GPUARRAY_LOCAL void hash_del(hash *h, const char *key);

GPUARRAY_LOCAL void hash_visit(hash *h, void (*f)(const char *k, void *v));

GPUARRAY_LOCAL size_t hash_size(hash *h);

#endif
