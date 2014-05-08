#include <sqlite3.h>
#include <stdlib.h>

#include "private.h"

static sqlite3 *db;

static sqlite3_stmt *get;
static sqlite3_stmt *add;
static sqlite3_stmt *del;

#ifdef WIN32
const char *sep = "\\";
#else
const char *sep = "/";
#endif

static const char * cache_basedir(void) {
  const char *res;
  res = getenv("GPUARRAY_CACHE");
  if (res != NULL)
    return res;
  res = getenv("LOCALAPPDATA");
  if (res != NULL)
    return res;
  res = getenv("HOME");
  if (res != NULL)
    return res;
  res = getenv("USERPROFILE");
  if (res != NULL)
    return res;
  return NULL;
}

/* This is djb hash */
static unsigned long hash(const char *str) {
  const unsigned char *s = (const unsigned char *)str;
  unsigned long h = 5381;
  int c;

  while ((c = *s++) != '\0')
    h = ((h << 5) + h) + c;

  return h;
}

static void cache_fini(void) {
  if (get != NULL)
    sqlite3_finalize(get);
  get = NULL;
  if (add != NULL)
    sqlite3_finalize(add);
  add = NULL;
  if (del != NULL)
    sqlite3_finalize(del);
  del = NULL;
  if (db != NULL)
    sqlite3_close(db);
  db = NULL;
}

static int cache_init(void) {
  char dbname[1024];
  const char *tmp;

  if (db != NULL)
    return GA_NO_ERROR;

  if (sqlite3_threadsafe() == 0)
    return GA_UNSUPPORTED_ERROR;

  if (sqlite3_config(SQLITE_CONFIG_SERIALIZED) != SQLITE_OK)
    return GA_UNSUPPORTED_ERROR;

  tmp = cache_basedir();
  if (tmp == NULL)
    return GA_MISC_ERROR;
  if (strlcpy(dbname, tmp, sizeof(dbname)) >= sizeof(dbname))
    return GA_MISC_ERROR;
  /*
   * Allows disabling of the on-disk cache by setting
   * GPUARRAY_CACHE to ':memory:'.
   */
  if (dbname[0] != ':') {
    if (strlcat(dbname, sep, sizeof(dbname)) >= sizeof(dbname))
      return GA_MISC_ERROR;
    if (strlcat(dbname, "gpuarray.cache", sizeof(dbname)) >= sizeof(dbname))
      return GA_MISC_ERROR;
  }

  if (sqlite3_open(dbname, &db))
    goto error;
  if (sqlite3_exec(db, "PRAGMA application_id = 99845574;\n"
                   "PRAGMA page_size = 8192;", NULL, NULL, NULL))
    goto error;

  tmp = "CREATE TABLE IF NOT EXISTS cache ("
    "id INTEGER  PRIMARY KEY,"
    "kind TEXT      NOT NULL,"
    "devname TEXT   NOT NULL,"
    "hash INTEGER   NOT NULL,"
    "code BLOB      NOT NULL,"
    "bin BLOB       NOT NULL);"
    "CREATE INDEX IF NOT EXISTS cache_idx ON cache(kind, devname, hash);";

  if (sqlite3_exec(db, tmp, NULL, NULL, NULL))
    goto error;

  if (sqlite3_prepare_v2(db, "SELECT * FROM cache WHERE "
                         "(kind = ? AND devname = ? AND hash = ?);",
                         -1, &get, NULL))
    goto error;
  if (sqlite3_prepare_v2(db, "INSERT INTO "
                         "cache(kind, devname, hash, code, bin) "
                         "VALUES (?, ?, ?, ?, ?);",
                         -1, &add, NULL))
    goto error;
  if (sqlite3_prepare_v2(db, "DELETE FROM cache WHERE (id = ?);",
                         -1, &del, NULL))
    goto error;

  atexit(cache_fini);
  return GA_NO_ERROR;
 error:
  cache_fini();
  return GA_MISC_ERROR;
}

static int _sql_get(const char *kind, const char *devname, const char *code) {
  const void *tmp;
  unsigned long h = hash(code);
  int res = GA_MISC_ERROR;
  int status;
  int ntries = 0;

  if (sqlite3_bind_text(get, 1, kind, -1, SQLITE_STATIC))
    goto error;
  if (sqlite3_bind_text(get, 2, devname, -1, SQLITE_STATIC))
    goto error;
  if (sqlite3_bind_int64(get, 3, h))
    goto error;

 again:
  status = sqlite3_step(get);
  switch (status) {
  case SQLITE_BUSY:
    ntries++;
    if (ntries > 3) {
      res = GA_MISC_ERROR;
      goto error;
    }
    goto again;
    break;
  case SQLITE_ROW:
    ntries = 0;
    tmp = sqlite3_column_blob(get, 5);
    if (strncmp((const char *)tmp, code, sqlite3_column_bytes(get, 5)) == 0) {
      res = GA_NO_ERROR;
      goto end;
    } else {
      goto again;
    }
    break;
  case SQLITE_DONE:
    /* Value was not found in cache */
    res = GA_VALUE_ERROR;
    goto error;
  default:
    res = GA_MISC_ERROR;
    goto error;
  }

 error:
  sqlite3_reset(get);
 end:
  sqlite3_clear_bindings(get);
  return res;
}

static int _sql_del(sqlite3_int64 i) {
  int res = GA_MISC_ERROR;
  int status;
  int ntries = 0;

  if (sqlite3_bind_int64(del, 1, i))
    goto error;

 again:
  status = sqlite3_step(del);
  switch (status) {
  case SQLITE_BUSY:
    ntries++;
    if (ntries > 3) {
      res = GA_MISC_ERROR;
      goto error;
    }
  case SQLITE_ROW:
    goto again;
  case SQLITE_DONE:
    res = GA_NO_ERROR;
    goto end;
  default:
    goto error;
  }

 error:
 end:
  sqlite3_reset(del);
  sqlite3_clear_bindings(del);
  return res;
}

static int _sql_add(const char *kind, const char *devname,
                    const char *code, void *bin, size_t sz) {
  unsigned long h = hash(code);
  int res = GA_MISC_ERROR;
  int status;
  int ntries = 0;

  if (sqlite3_bind_text(get, 1, kind, -1, SQLITE_STATIC))
    goto error;
  if (sqlite3_bind_text(get, 2, devname, -1, SQLITE_STATIC))
    goto error;
  if (sqlite3_bind_int64(get, 3, h))
    goto error;
  if (sqlite3_bind_text(get, 4, code, -1, SQLITE_STATIC))
    goto error;
  if (sqlite3_bind_blob(get, 5, bin, sz, SQLITE_STATIC))
    goto error;

 again:
  status = sqlite3_step(add);
  switch (status) {
  case SQLITE_BUSY:
    ntries++;
    if (ntries > 3) {
      res = GA_MISC_ERROR;
      goto error;
    }
  case SQLITE_ROW:
    goto again;
  case SQLITE_DONE:
    res = GA_NO_ERROR;
    goto end;
  default:
    goto error;
  }

 error:
 end:
  sqlite3_reset(add);
  sqlite3_clear_bindings(add);
  return res;
}

int cache_put(const char *kind, const char *devname, const char *code,
              void *bin, size_t sz) {
  int err;

  err = cache_init();
  if (err != GA_NO_ERROR)
    return err;

  err = _sql_get(kind, devname, code);
  if (err == GA_NO_ERROR) {
    sqlite_int64 i = sqlite3_column_int64(get, 1);
    sqlite3_reset(get);
    _sql_del(i);
  } else if (err != GA_VALUE_ERROR) {
    return err;
  }

  return _sql_add(kind, devname, code, bin, sz);
}

int cache_get(const char *kind, const char *devname, const char *code,
              void **bin, size_t *sz) {
  const void *tmp;
  int err;

  err = cache_init();
  if (err != GA_NO_ERROR)
    return err;

  err = _sql_get(kind, devname, code);
  if (err != GA_NO_ERROR)
    return err;

  tmp = sqlite3_column_blob(get, 6);
  *sz = sqlite3_column_bytes(get, 6);
  *bin = memdup(tmp, *sz);
  sqlite3_reset(get);

 if (*bin == NULL)
    return GA_MEMORY_ERROR;
  return GA_NO_ERROR;
}
