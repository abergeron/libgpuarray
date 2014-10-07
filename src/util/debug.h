#ifndef UTIL_DEBUG_H
#define UTIL_DEBUG_H

#ifdef DEBUG
#include <stdio.h>
#define DPRINT(...) fprintf(stderr, __VA_ARGS__)
#else
#define DPRINT(...)
#endif

#endif
