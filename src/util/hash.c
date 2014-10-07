#include "util/halloc.h"
#include "util/hash.h"
#include "util/debug.h"
#include "private.h"
#include <assert.h>


typedef struct _node {
  const char *key;
  void *val;
  struct _node *next;
} node;

static void node_init(node *n, const char *key, void *val) {
  n->key = key;
  n->val = val;
  hattach(val, n);
  n->next = NULL;
}

static node *node_alloc(const char *key, void *val) {
  node *res = h_malloc(sizeof(node));
  if (res != NULL)
    node_init(res, key, val);
  return res;
}

struct _hash {
  node **keyval;
  uint32_t nbuckets;
  uint32_t size;
};

static uint32_t roundup2(uint32_t s) {
  s--;
  s |= s >> 1;
  s |= s >> 2;
  s |= s >> 4;
  s |= s >> 8;
  s |= s >> 16;
  s++;
  return s;
}

/* djb hash */
static uint32_t hashfn(const char *s) {
  const unsigned char *str = (const unsigned char *)s;
  uint32_t hash = 5381;
  int c;

  while ((c = *str++))
    hash = ((hash << 5) + hash) + c; /* hash * 33 + c */

  return hash;
}

static int hash_init(hash *h, size_t size) {
  h->nbuckets = roundup2(size);
  h->keyval = h_calloc(h->nbuckets, sizeof(*h->keyval));
  hattach(h->keyval, h);
  h->size = 0;
  return h->keyval == NULL;
}

hash *hash_new(uint32_t size) {
  hash *h = h_malloc(sizeof(hash));
  if (h == NULL) return NULL;
  if (size == 0)
    size = 16;
  else
    size = size + (size/6);

  if (hash_init(h, size)) return NULL;
  return h;
}

void *hash_find(hash *h, const char *key) {
  uint32_t p = hashfn(key) & (h->nbuckets - 1);
  node *n;
  if (h->keyval[p] != NULL) {
    n = h->keyval[p];
    do {
      if (strcmp(n->key, key) == 0)
        return n->val;
      n = n->next;
    } while (n != NULL);
  }
  return NULL;
}

static uint32_t hash_insert(hash *h, node *n) {
  uint32_t p = hashfn(n->key) & (h->nbuckets - 1);
  if (h->keyval[p] == NULL) {
    h->keyval[p] = n;
  } else {
    n->next = h->keyval[p];
    h->keyval[p] = n;
  }
  h->size++;
  return p;
}

static void hash_reinsertchain(hash *h, node *n) {
  /* reinsert the leaf node first otherwise we lose the ->next chain */
  if (n->next != NULL)
    hash_reinsertchain(h, n->next);
  hash_insert(h, n);
}

static void hash_rehash(hash *h) {
  node **keyval = h->keyval;
  uint32_t n = h->nbuckets;
  uint32_t sz = h->size;
  uint32_t i;

  if (hash_init(h, h->nbuckets*2)) goto undo;

  for (i = 0; i < n; i++) {
    if (keyval[i] != NULL)
      hash_reinsertchain(h, keyval[i]);
  }
  h_free(keyval);

  /* we shouldn't lose nodes */
  assert(h->size == sz);

  return;
 undo:
  h->keyval = keyval;
  h->nbuckets = n;
  h->size = sz;
}

int hash_add(hash *h, const char *key, void *val) {
  node *n = node_alloc(key, val);
  if (n == NULL) {
    DPRINT("failed to alloc node for key %s", key);
    return -1;
  }
  hattach(n, h);
  hash_insert(h, n);
  /* hash can only grow, not shrink */
  if (h->size >= ((h->nbuckets/4)*3))
    hash_rehash(h);
  return 0;
}

void hash_del(hash *h, const char *key) {
  uint32_t p = hashfn(key) & (h->nbuckets - 1);
  node *np = h->keyval[p];
  node *n;

  /* is the bucket empty? */
  if (np == NULL) return;

  /* first element is a little special */
  if (strcmp(np->key, key) == 0) {
    h->keyval[p] = np->next;
    h_free(np);
    h->size--;
    return;
  }

  /* check the rest of the bucket */
  while (np->next != NULL) {
    n = np->next;
    if (strcmp(n->key, key) == 0) {
      np->next = n->next;
      h_free(n);
      h->size--;
      return;
    }
    np = np->next;
  }
  return;
}

void hash_visit(hash *h, void (*f)(const char *k, void *v)) {
  uint32_t i;
  node *n;
  for (i = 0; i < h->nbuckets; i++) {
    n = h->keyval[i];
    while (n != NULL) {
      f(n->key, n->val);
      n = n->next;
    }
  }
}

size_t hash_size(hash *h) {
  return h->size;
}
