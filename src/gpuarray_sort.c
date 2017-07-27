#include <assert.h>
#include <limits.h>
#include <float.h>

#include <gpuarray/sort.h>
#include <gpuarray/array.h>
#include <gpuarray/kernel.h>
#include <gpuarray/util.h>

#include "util/strb.h"
#include "private.h"

/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 * This software contains source code provided by NVIDIA Corporation.
 *
 * Read more at: http://docs.nvidia.com/cuda/eula/index.html#ixzz4lUbgXjsr
 * Follow us: @GPUComputing on Twitter | NVIDIA on Facebook
 *
 *
 */

 //#define checkErr(x) checkErrors(x, __FILE__, __LINE__)
#define checkErr(err)  if (err != GA_NO_ERROR) return err;

const int flags = GA_USE_CLUDA;

static const char *code_helper_funcs =                                                                               \
"\n#define SAMPLE_STRIDE 128 \n"                                                                                     \
"\n#define SHARED_SIZE_LIMIT  1024U \n"                                                                              \
"__device__ unsigned int iDivUp(unsigned int a, unsigned int b)"                                                     \
"{"                                                                                                                  \
"    return ((a % b) == 0) ? (a / b) : (a / b + 1); "                                                                \
"} "                                                                                                                 \
"__device__ unsigned int getSampleCount(unsigned int dividend) "                                                     \
"{ "                                                                                                                 \
"    return iDivUp(dividend, SAMPLE_STRIDE); "                                                                       \
"}"                                                                                                                  \
"\n #define W (sizeof(unsigned int) * 8) \n"                                                                         \
"__device__ unsigned int nextPowerOfTwo(unsigned int x) "                                                            \
"{"                                                                                                                  \
"    return 1U << (W - __clz(x - 1));"                                                                               \
"} "                                                                                                                 \
"template<typename T> __device__ T readArray(T *a, unsigned int pos, unsigned int length, unsigned int sortDir){"    \
"      if (pos >= length) { "                                                                                        \
"          if (sortDir) { "                                                                                          \
"             return MAX_NUM; "                                                                                      \
"          } "                                                                                                       \
"          else { "                                                                                                  \
"             return MIN_NUM; "                                                                                      \
"          } "                                                                                                       \
"      } "                                                                                                           \
"      else { "                                                                                                      \
"          return a[pos]; "                                                                                          \
"      } "                                                                                                           \
"  } "                                                                                                               \
"template<typename T> __device__ T readArray_arg(T *a, unsigned int pos, unsigned int length, unsigned int sortDir){"\
"      if (pos >= length) { "                                                                                        \
"          if (sortDir) { "                                                                                          \
"             return MAX_NUM_ARG; "                                                                                  \
"          } "                                                                                                       \
"          else { "                                                                                                  \
"             return MIN_NUM_ARG; "                                                                                  \
"          } "                                                                                                       \
"      } "                                                                                                           \
"      else { "                                                                                                      \
"          return a[pos]; "                                                                                          \
"      } "                                                                                                           \
"  } "                                                                                                               \
"template<typename T> __device__ void writeArray(T *a, unsigned int pos, T value, unsigned int length) "             \
" { "                                                                                                                \
"     if (pos >= length) "                                                                                           \
"     { "                                                                                                            \
"          return; "                                                                                                 \
"     } "                                                                                                            \
"     else { "                                                                                                       \
"         a[pos] = value; "                                                                                          \
"     } "                                                                                                            \
" }\n";

static unsigned int iDivUp(unsigned int a, unsigned int b) {
    return ((a % b) == 0) ? (a / b) : (a / b + 1);
}

static unsigned int getSampleCount(unsigned int dividend) {
    return iDivUp(dividend, SAMPLE_STRIDE);
}

static unsigned int roundDown(unsigned int numToRound, unsigned int multiple) {
  if (numToRound <= multiple)
    return numToRound;
  else
    return (numToRound / multiple) * multiple;    
}

static inline const char *ctype(int typecode) {
  return gpuarray_get_type(typecode)->cluda_name;
}

static const char *code_bin_search =                                                                              \
"template<typename T> __device__ unsigned int binarySearchInclusive(T val, T *data, unsigned int L, "             \
"                                              unsigned int stride, unsigned int sortDir){"                       \
"    if (L == 0) "                                                                                                \
"        return 0; "                                                                                              \
"    unsigned int pos = 0; "                                                                                      \
"    for (; stride > 0; stride >>= 1){ "                                                                          \
"      unsigned int newPos = min(pos + stride, L); "                                                              \
"      if ((sortDir && (data[newPos - 1] <= val)) || (!sortDir && (data[newPos - 1] >= val))){ "                  \
"          pos = newPos; "                                                                                        \
"      } "                                                                                                        \
"    } "                                                                                                          \
"    return pos; "                                                                                                \
"} "                                                                                                              \
" template<typename T> __device__ unsigned int binarySearchExclusive(T val, T *data, unsigned int L, "            \
"                                              unsigned int stride, unsigned int sortDir) "                       \
"{ "                                                                                                              \
"    if (L == 0) "                                                                                                \
"        return 0; "                                                                                              \
"    unsigned int pos = 0; "                                                                                      \
"    for (; stride > 0; stride >>= 1){ "                                                                          \
"      unsigned int newPos = min(pos + stride, L); "                                                              \
"      if ((sortDir && (data[newPos - 1] < val)) || (!sortDir && (data[newPos - 1] > val))){ "                    \
"          pos = newPos; "                                                                                        \
"      } "                                                                                                        \
"    } "                                                                                                          \
"    return pos; "                                                                                                \
"}"                                                                                                               \
"template<typename T> __device__ unsigned int binarySearchLowerBoundExclusive(T val, T *ptr, unsigned int first," \
"                                                                             unsigned int last, unsigned int sortDir) "  \
"{ "                                                                                                                      \
"    unsigned int len = last - first; "                                                                                   \
"    unsigned int half; "                                                                                                 \
"    unsigned int middle; "                                                                                               \
"    while (len > 0) { "                                                                                                  \
"        half = len >> 1; "                                                                                               \
"        middle = first; "                                                                                                \
"        middle += half; "                                                                                                \
"        if ( (sortDir && ptr[middle] < val) || (!sortDir && ptr[middle] > val) ) { "                                     \
"            first = middle; "                                                                                            \
"            ++first; "                                                                                                   \
"            len = len - half - 1; "                                                                                      \
"        } "                                                                                                              \
"        else "                                                                                                           \
"            len = half; "                                                                                                \
"    } "                                                                                                                  \
"    return first; "                                                                                                      \
"} "                                                                                                                      \
"template<typename T> __device__ unsigned int binarySearchLowerBoundInclusive(T val, T *ptr, unsigned int first,  "       \
"                                                                             unsigned int last, unsigned int sortDir) "  \
"{ "                                                                                                                      \
"    unsigned int len = last - first; "                                                                                   \
"    unsigned int half; "                                                                                                 \
"    unsigned int middle; "                                                                                               \
"    while (len > 0) { "                                                                                                  \
"        half = len >> 1; "                                                                                               \
"        middle = first; "                                                                                                \
"        middle += half; "                                                                                                \
"        if ( (sortDir && ptr[middle] <= val) || (!sortDir && ptr[middle] >= val) ) { "                                   \
"            first = middle; "                                                                                            \
"            ++first; "                                                                                                   \
"            len = len - half - 1; "                                                                                      \
"        } "                                                                                                              \
"        else "                                                                                                           \
"            len = half; "                                                                                                \
"    } "                                                                                                                  \
"    return first; "                                                                                                      \
"}\n";

#define NUMARGS_BITONIC_KERNEL 8
int type_args_bitonic[NUMARGS_BITONIC_KERNEL] = {GA_BUFFER, GA_SIZE, GA_BUFFER, GA_SIZE, GA_UINT, GA_UINT, GA_UINT, GA_UINT};
#define NUMARGS_BITONIC_KERNEL_ARG 12
int type_args_bitonic_arg[NUMARGS_BITONIC_KERNEL_ARG] = {GA_BUFFER, GA_SIZE, GA_BUFFER, GA_SIZE, GA_BUFFER, GA_SIZE, GA_BUFFER, 
                                                           GA_SIZE, GA_UINT, GA_UINT, GA_UINT, GA_UINT};
static const char *code_bitonic_smem =                                                                                                          \
" extern \"C\" __global__ void bitonicSortSharedKernel( "                                                                                       \
"      t_key *d_DstKey, "                                                                                                                       \
"      size_t dstOff,"                                                                                                                          \
"      t_key *d_SrcKey, "                                                                                                                       \
"      size_t srcOff,"\
"\n#ifdef ARGSORT\n" \
"      t_arg *d_DstArg, "\
"      size_t dstArgOff, "\
"      t_arg *d_SrcArg, "\
"      size_t srcArgOff, "                                                                                                                \
"\n#endif\n"\
"      unsigned int batchSize, "                                                                                                                \
"      unsigned int arrayLength, "                                                                                                              \
"      unsigned int elemsOff, "                                                                                                                 \
"      unsigned int sortDir "                                                                                                                   \
"  ) "                                                                                                                                          \
"  { "                                                                                                                                          \
"      d_DstKey = (t_key*) (((char*)d_DstKey)+ dstOff);"                                                                                        \
"      d_SrcKey = (t_key*) (((char*)d_SrcKey)+ srcOff);"                                                                                        \
"\n#ifdef ARGSORT\n" \
"      d_DstArg = (t_arg*) (((char*)d_DstArg)+ dstArgOff); "\
"      d_SrcArg = (t_arg*) (((char*)d_SrcArg)+ srcArgOff);"\
"      d_DstArg += elemsOff;"\
"      d_SrcArg += elemsOff;" \
"      __shared__ t_arg s_arg[SHARED_SIZE_LIMIT];" \
"\n#endif\n"\
"      d_DstKey += elemsOff;"                                                                                                                   \
"      d_SrcKey += elemsOff;"                                                                                                                   \
"      __shared__ t_key s_key[SHARED_SIZE_LIMIT]; "                                                                                             \
"      s_key[threadIdx.x] = readArray<t_key>( d_SrcKey, "                                                                                       \
"                                             blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x, "                                                   \
"                                             arrayLength * batchSize, "                                                                        \
"                                             sortDir "                                                                                         \
"                                           ); "                                                                                                \
"      s_key[threadIdx.x + (SHARED_SIZE_LIMIT / 2)] = readArray<t_key>( d_SrcKey, "                                                             \
"                                                                       blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x + (SHARED_SIZE_LIMIT / 2),"\
"                                                                       arrayLength * batchSize, "                                              \
"                                                                       sortDir "                                                               \
"                                                                     ); "                                                                      \
"\n#ifdef ARGSORT\n"
"      s_arg[threadIdx.x] = readArray_arg<t_arg>( d_SrcArg, "\
"                                             blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x, "\
"                                             arrayLength * batchSize, "\
"                                             sortDir "\
"                                            ); "\
"      s_arg[threadIdx.x + (SHARED_SIZE_LIMIT / 2)] = readArray_arg<t_arg>( d_SrcArg," \
"                                                                       blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x + (SHARED_SIZE_LIMIT / 2),"\
"                                                                       arrayLength * batchSize, "\
"                                                                       sortDir "\
"                                                                      ); "\
"\n#endif\n"                  \
"      for (unsigned int size = 2; size < SHARED_SIZE_LIMIT; size <<= 1) { "                                                                    \
"          unsigned int ddd = sortDir ^ ((threadIdx.x & (size / 2)) != 0); "                                                                    \
"          for (unsigned int stride = size / 2; stride > 0; stride >>= 1) "                                                                     \
"          { "                                                                                                                                  \
"              __syncthreads(); "                                                                                                               \
"              unsigned int pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1)); "                                                             \
"              t_key t; "                                                                                                                       \
"              if ((s_key[pos] > s_key[pos + stride]) == ddd) { "                                                                               \
"                  t = s_key[pos]; "                                                                                                            \
"                  s_key[pos] = s_key[pos + stride]; "                                                                                          \
"                  s_key[pos + stride] = t; "                                                                                                   \
"\n#ifdef ARGSORT\n" \
"                  t_arg t2 = s_arg[pos];"\
"                  s_arg[pos] = s_arg[pos + stride];" \
"                  s_arg[pos + stride] = t2;" \
"\n#endif\n" \
"              } "                                                                                                                              \
"          } "                                                                                                                                  \
"      } "                                                                                                                                      \
"      { "                                                                                                                                      \
"          for (unsigned int stride = SHARED_SIZE_LIMIT / 2; stride > 0; stride >>= 1) {"                                                       \
"              __syncthreads(); "                                                                                                               \
"              unsigned int pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1)); "                                                             \
"              t_key t; "                                                                                                                       \
"              if ((s_key[pos] > s_key[pos + stride]) == sortDir) {"                                                                            \
"                  t = s_key[pos]; "                                                                                                            \
"                  s_key[pos] = s_key[pos + stride]; "                                                                                          \
"                  s_key[pos + stride] = t; "                                                                                                   \
"\n#ifdef ARGSORT\n" \
"                  t_arg t2 = s_arg[pos];"\
"                  s_arg[pos] = s_arg[pos + stride];" \
"                  s_arg[pos + stride] = t2;" \
"\n#endif\n" \
"              } "                                                                                                                              \
"          } "                                                                                                                                  \
"      } "                                                                                                                                      \
"      __syncthreads(); "                                                                                                                       \
"      writeArray<t_key>( d_DstKey, "                                                                                                           \
"                  blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x,  "                                                                             \
"                  s_key[threadIdx.x], "                                                                                                        \
"                  arrayLength * batchSize "                                                                                                    \
"                ); "                                                                                                                           \
"      writeArray<t_key>( d_DstKey, "                                                                                                           \
"                  blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x + (SHARED_SIZE_LIMIT / 2), "                                                    \
"                  s_key[threadIdx.x + (SHARED_SIZE_LIMIT / 2)], "                                                                              \
"                  arrayLength * batchSize "                                                                                                    \
"                ); "                                                                                                                           \
"\n#ifdef ARGSORT\n" \
"       writeArray<t_arg>( d_DstArg, " \
"                   blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x, " \
"                   s_arg[threadIdx.x], " \
"                   arrayLength * batchSize " \
"                  ); " \
"      writeArray<t_arg>( d_DstArg, " \
"                                 blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x + (SHARED_SIZE_LIMIT / 2), " \
"                                 s_arg[threadIdx.x + (SHARED_SIZE_LIMIT / 2)], " \
"                                 arrayLength * batchSize " \
"                               ); " \
"\n#endif\n "\
"}\n";
static int bitonicSortShared(
    GpuArray *d_DstKey,
    GpuArray *d_SrcKey,
    GpuArray *d_DstArg,
    GpuArray *d_SrcArg,
    unsigned int batchSize,
    unsigned int arrayLength,
    unsigned int sortDir,
    unsigned int elemsOff,
    unsigned int argSortFlg,
    GpuKernel *k_bitonic,
    gpucontext *ctx
)
{
  size_t ls, gs;
  unsigned int p = 0;
  int err = GA_NO_ERROR;

  ls = SHARED_SIZE_LIMIT / 2;
  gs = batchSize;

  err = GpuKernel_setarg(k_bitonic, p++, d_DstKey->data);
  if (err != GA_NO_ERROR) return err;

  err = GpuKernel_setarg(k_bitonic, p++, &d_DstKey->offset);
  if (err != GA_NO_ERROR) return err;

  err = GpuKernel_setarg(k_bitonic, p++, d_SrcKey->data);
  if (err != GA_NO_ERROR) return err;

  err = GpuKernel_setarg(k_bitonic, p++, &d_SrcKey->offset);
  if (err != GA_NO_ERROR) return err;

  if (argSortFlg) {
    err = GpuKernel_setarg(k_bitonic, p++, d_DstArg->data);
    if (err != GA_NO_ERROR) return err;

    err = GpuKernel_setarg(k_bitonic, p++, &d_DstArg->offset);
    if (err != GA_NO_ERROR) return err;

    err = GpuKernel_setarg(k_bitonic, p++, d_SrcArg->data);
    if (err != GA_NO_ERROR) return err;

    err = GpuKernel_setarg(k_bitonic, p++, &d_SrcArg->offset);
    if (err != GA_NO_ERROR) return err;
  }
  
  err = GpuKernel_setarg(k_bitonic, p++, &batchSize);
  if (err != GA_NO_ERROR) return err;
  
  err = GpuKernel_setarg(k_bitonic, p++, &arrayLength);
  if (err != GA_NO_ERROR) return err;

  err = GpuKernel_setarg(k_bitonic, p++, &elemsOff);
  if (err != GA_NO_ERROR) return err;
  
  err = GpuKernel_setarg(k_bitonic, p++, &sortDir);
  if (err != GA_NO_ERROR) return err;

  err = GpuKernel_call(k_bitonic, 1, &gs, &ls, 0, NULL);
  if (err != GA_NO_ERROR) return err;

  return err;  
/*
  float *h_dst2 = (float *) malloc ( 16 * sizeof(float));
  err = GpuArray_read(h_dst2, 16 * sizeof(float), d_DstKey);
  if (err != GA_NO_ERROR) printf("error reading \n");

  
  int i;
  for (i = 0; i < 16; i++)
  {
      printf("%d afterbitonic %f \n", i, h_dst2[i]);
  }
  */
}

#define NUMARGS_SAMPLE_RANKS 10
const int type_args_ranks[NUMARGS_SAMPLE_RANKS] = {GA_BUFFER, GA_SIZE, GA_BUFFER, GA_SIZE, GA_BUFFER, GA_SIZE,
                                                   GA_UINT, GA_UINT, GA_UINT, GA_UINT};                                                   
static const char *code_sample_ranks =                                                                              \
"extern \"C\" __global__ void generateSampleRanksKernel("                                                           \
"    unsigned int *d_RanksA,"                                                                                       \
"    size_t rankAOff,"                                                                                              \
"    unsigned int *d_RanksB,"                                                                                       \
"    size_t rankBOff,"                                                                                              \
"    t_key *d_SrcKey,"                                                                                              \
"    size_t srcOff,"                                                                                                \
"    unsigned int stride,"                                                                                          \
"    unsigned int N,"                                                                                               \
"    unsigned int threadCount,"                                                                                     \
"    unsigned int sortDir"                                                                                          \
")"                                                                                                                 \
"{"                                                                                                                 \
"    d_RanksA = (unsigned int*) (((char*)d_RanksA)+ rankAOff);"                                                     \
"    d_RanksB = (unsigned int*) (((char*)d_RanksB)+ rankBOff);"                                                     \
"    d_SrcKey = (t_key*) (((char*)d_SrcKey)+ srcOff);"                                                              \
"    unsigned int pos = blockIdx.x * blockDim.x + threadIdx.x;"                                                     \
"    if (pos >= threadCount) {"                                                                                     \
"        return;"                                                                                                   \
"    }"                                                                                                             \
"    const unsigned int           i = pos & ((stride / SAMPLE_STRIDE) - 1);"                                        \
"    const unsigned int segmentBase = (pos - i) * (2 * SAMPLE_STRIDE);"                                             \
"    d_SrcKey += segmentBase;"                                                                                      \
"    d_RanksA += segmentBase / SAMPLE_STRIDE;"                                                                      \
"    d_RanksB += segmentBase / SAMPLE_STRIDE;"                                                                      \
"    const unsigned int segmentElementsA = stride;"                                                                 \
"    const unsigned int segmentElementsB = min(stride, N - segmentBase - stride);"                                  \
"    const unsigned int  segmentSamplesA = getSampleCount(segmentElementsA);"                                       \
"    const unsigned int  segmentSamplesB = getSampleCount(segmentElementsB);"                                       \
"    if (i < segmentSamplesA) {"                                                                                    \
"        d_RanksA[i] = i * SAMPLE_STRIDE;"                                                                          \
"        d_RanksB[i] = binarySearchExclusive<t_key>("                                                               \
"                          d_SrcKey[i * SAMPLE_STRIDE], d_SrcKey + stride,"                                         \
"                          segmentElementsB, nextPowerOfTwo(segmentElementsB), sortDir"                             \
"                      );"                                                                                          \
"    }"                                                                                                             \
"    if (i < segmentSamplesB) {"                                                                                    \
"        d_RanksB[(stride / SAMPLE_STRIDE) + i] = i * SAMPLE_STRIDE;"                                               \
"        d_RanksA[(stride / SAMPLE_STRIDE) + i] = binarySearchInclusive<t_key>("                                    \
"                                                     d_SrcKey[stride + i * SAMPLE_STRIDE], d_SrcKey + 0,"          \
"                                                     segmentElementsA, nextPowerOfTwo(segmentElementsA), sortDir"  \
"                                                 );"                                                               \
"    }"                                                                                                             \
"}\n";
static int generateSampleRanks(
  GpuSortData *msData,
  GpuArray *d_SrcKey,
  unsigned int stride,
  GpuSortConfig *msConfig,
  GpuKernel *k_ranks,
  gpucontext *ctx
)
{
  unsigned int lastSegmentElements = msConfig->Nfloor % (2 * stride);
  unsigned int threadCount = (lastSegmentElements > stride) ? 
                            (msConfig->Nfloor + 2 * stride - lastSegmentElements) / (2 * SAMPLE_STRIDE) : 
                            (msConfig->Nfloor - lastSegmentElements) / (2 * SAMPLE_STRIDE);

  size_t ls, gs;
  unsigned int p = 0;
  int err = GA_NO_ERROR;

  ls = 256;
  gs = iDivUp(threadCount, 256);

  err = GpuKernel_setarg(k_ranks, p++, msData->d_RanksA.data);
  if (err != GA_NO_ERROR) return err;

  err = GpuKernel_setarg(k_ranks, p++, &msData->d_RanksA.offset);
  if (err != GA_NO_ERROR) return err;

  err = GpuKernel_setarg(k_ranks, p++, msData->d_RanksB.data);
  if (err != GA_NO_ERROR) return err;

  err = GpuKernel_setarg(k_ranks, p++, &msData->d_RanksB.offset);
  if (err != GA_NO_ERROR) return err;
  
  err = GpuKernel_setarg(k_ranks, p++, d_SrcKey->data);
  if (err != GA_NO_ERROR) return err;

  err = GpuKernel_setarg(k_ranks, p++, &d_SrcKey->offset);
  if (err != GA_NO_ERROR) return err;

  err = GpuKernel_setarg(k_ranks, p++, &stride);
  if (err != GA_NO_ERROR) return err;

  err = GpuKernel_setarg(k_ranks, p++, &msConfig->Nfloor);
  if (err != GA_NO_ERROR) return err;

  err = GpuKernel_setarg(k_ranks, p++, &threadCount);
  if (err != GA_NO_ERROR) return err;

  err = GpuKernel_setarg(k_ranks, p++, &msConfig->sortDirFlg);
  if (err != GA_NO_ERROR) return err;

  err = GpuKernel_call(k_ranks, 1, &gs, &ls, 0, NULL);
  if (err != GA_NO_ERROR) return err;
 
  return err;
}

#define NUMARGS_RANKS_IDXS 7
const int type_args_ranks_idxs[NUMARGS_RANKS_IDXS] = {GA_BUFFER, GA_SIZE, GA_BUFFER, GA_SIZE, GA_UINT, GA_UINT, GA_UINT};
static const char *code_ranks_idxs =                                                                                           \
"extern \"C\" __global__ void mergeRanksAndIndicesKernel( "                                                                    \
"    unsigned int *d_Limits, "                                                                                                 \
"    size_t limOff,"                                                                                                           \
"    unsigned int *d_Ranks, "                                                                                                  \
"    size_t rankOff,"                                                                                                          \
"    unsigned int stride, "                                                                                                    \
"    unsigned int N, "                                                                                                         \
"    unsigned int threadCount "                                                                                                \
") "                                                                                                                           \
"{ "                                                                                                                           \
"    d_Limits = (unsigned int*) (((char*)d_Limits)+ limOff);"                                                                  \
"    d_Ranks = (unsigned int*) (((char*)d_Ranks)+ rankOff);"                                                                   \
"    unsigned int pos = blockIdx.x * blockDim.x + threadIdx.x; "                                                               \
"    if (pos >= threadCount) "                                                                                                 \
"        return; "                                                                                                             \
"    const unsigned int           i = pos & ((stride / SAMPLE_STRIDE) - 1); "                                                  \
"    const unsigned int segmentBase = (pos - i) * (2 * SAMPLE_STRIDE); "                                                       \
"    d_Ranks  += (pos - i) * 2; "                                                                                              \
"    d_Limits += (pos - i) * 2; "                                                                                              \
"    const unsigned int segmentElementsA = stride; "                                                                           \
"    const unsigned int segmentElementsB = min(stride, N - segmentBase - stride); "                                            \
"    const unsigned int  segmentSamplesA = getSampleCount(segmentElementsA); "                                                 \
"    const unsigned int  segmentSamplesB = getSampleCount(segmentElementsB); "                                                 \
"    if (i < segmentSamplesA) { "                                                                                              \
"        unsigned int dstPos = binarySearchExclusive<unsigned int>(d_Ranks[i], d_Ranks + segmentSamplesA,"                     \
"                                                                  segmentSamplesB, nextPowerOfTwo(segmentSamplesB), 1U) + i;" \
"        d_Limits[dstPos] = d_Ranks[i]; "                                                                                      \
"    } "                                                                                                                       \
"    if (i < segmentSamplesB) { "                                                                                              \
"        unsigned int dstPos = binarySearchInclusive<unsigned int>(d_Ranks[segmentSamplesA + i], d_Ranks,"                     \
"                                                                   segmentSamplesA, nextPowerOfTwo(segmentSamplesA), 1U) + i;"\
"        d_Limits[dstPos] = d_Ranks[segmentSamplesA + i]; "                                                                    \
"    } "                                                                                                                       \
"}\n";
static int mergeRanksAndIndices(
  GpuSortData *msData,
  unsigned int stride,
  GpuSortConfig *msConfig,
  GpuKernel *k_ranks_idxs,
  gpucontext *ctx
)
{
  unsigned int lastSegmentElements = msConfig->Nfloor % (2 * stride);
  unsigned int threadCount = (lastSegmentElements > stride) ? 
                             (msConfig->Nfloor + 2 * stride - lastSegmentElements) / (2 * SAMPLE_STRIDE) : 
                             (msConfig->Nfloor - lastSegmentElements) / (2 * SAMPLE_STRIDE);
  size_t ls, gs;
  unsigned int p = 0;
  int err = GA_NO_ERROR;

  ls = 256U;
  gs = iDivUp(threadCount, 256U);

  err = GpuKernel_setarg(k_ranks_idxs, p++, msData->d_LimitsA.data);
  if (err != GA_NO_ERROR) return err;

  err = GpuKernel_setarg(k_ranks_idxs, p++, &msData->d_LimitsA.offset);
  if (err != GA_NO_ERROR) return err;

  err = GpuKernel_setarg(k_ranks_idxs, p++, msData->d_RanksA.data);
  if (err != GA_NO_ERROR) return err;

  err = GpuKernel_setarg(k_ranks_idxs, p++, &msData->d_RanksA.offset);
  if (err != GA_NO_ERROR) return err;

  err = GpuKernel_setarg(k_ranks_idxs, p++, &stride);
  if (err != GA_NO_ERROR) return err;

  err = GpuKernel_setarg(k_ranks_idxs, p++, &msConfig->Nfloor);
  if (err != GA_NO_ERROR) return err;

  err = GpuKernel_setarg(k_ranks_idxs, p++, &threadCount);
  if (err != GA_NO_ERROR) return err;

  err = GpuKernel_call(k_ranks_idxs, 1, &gs, &ls, 0, NULL);
  if (err != GA_NO_ERROR) return err;

  p = 0;

  err = GpuKernel_setarg(k_ranks_idxs, p++, msData->d_LimitsB.data);
  if (err != GA_NO_ERROR) return err;

  err = GpuKernel_setarg(k_ranks_idxs, p++, &msData->d_LimitsB.offset);
  if (err != GA_NO_ERROR) return err;

  err = GpuKernel_setarg(k_ranks_idxs, p++, msData->d_RanksB.data);
  if (err != GA_NO_ERROR) return err;

  err = GpuKernel_setarg(k_ranks_idxs, p++, &msData->d_RanksB.offset);
  if (err != GA_NO_ERROR) return err;

  err = GpuKernel_call(k_ranks_idxs, 1, &gs, &ls, 0, NULL);
  if (err != GA_NO_ERROR) return err;

  return err;
}

#define NUMARGS_MERGE 11
int type_args_merge[NUMARGS_MERGE] = {GA_BUFFER, GA_SIZE, GA_BUFFER, GA_SIZE, GA_BUFFER, GA_SIZE, GA_BUFFER,
                                            GA_SIZE, GA_UINT, GA_UINT, GA_UINT};
#define NUMARGS_MERGE_ARG 15
int type_args_merge_arg[NUMARGS_MERGE_ARG] = {GA_BUFFER, GA_SIZE, GA_BUFFER, GA_SIZE, GA_BUFFER, GA_SIZE, GA_BUFFER,
                                                    GA_SIZE, GA_BUFFER, GA_SIZE, GA_BUFFER, GA_SIZE, GA_UINT, GA_UINT, GA_UINT};
static const char *code_merge =                                                                                     \
" template<typename T> __device__ void merge( "                                                                     \
"    T *dstKey, "                                                                                                   \
"    T *srcAKey, "                                                                                                  \
"    T *srcBKey, "                                                                                                  \
"\n#ifdef ARGSORT\n"                                                                                                \
"    t_arg *dstVal, "                                                                                               \
"    t_arg *srcAVal, "                                                                                              \
"    t_arg *srcBVal, "                                                                                              \
"\n#endif\n"                                                                                                        \
"    unsigned int lenA, "                                                                                           \
"    unsigned int nPowTwoLenA, "                                                                                    \
"    unsigned int lenB, "                                                                                           \
"    unsigned int nPowTwoLenB, "                                                                                    \
"    unsigned int sortDir "                                                                                         \
") "                                                                                                                \
"{ "                                                                                                                \
"    T keyA, keyB; "                                                                                                \
"\n#ifdef ARGSORT\n"                                                                                                \
"    t_arg valA, valB; "                                                                                            \
"\n#endif\n"                                                                                                        \
"    unsigned int dstPosA , dstPosB;"                                                                               \
"    if (threadIdx.x < lenA) { "                                                                                    \
"        keyA = srcAKey[threadIdx.x]; "                                                                             \
"\n#ifdef ARGSORT\n"                                                                                                \
"        valA = srcAVal[threadIdx.x]; "                                                                             \
"\n#endif\n"                                                                                                        \
"        dstPosA = binarySearchExclusive<T>(keyA, srcBKey, lenB, nPowTwoLenB, sortDir) + threadIdx.x; "             \
"    } "                                                                                                            \
"    if (threadIdx.x < lenB) { "                                                                                    \
"        keyB = srcBKey[threadIdx.x]; "                                                                             \
"\n#ifdef ARGSORT\n"                                                                                                \
"        valB = srcBVal[threadIdx.x]; "                                                                             \
"\n#endif\n"                                                                                                        \
"        dstPosB = binarySearchInclusive<T>(keyB, srcAKey, lenA, nPowTwoLenA, sortDir) + threadIdx.x; "             \
"    } "                                                                                                            \
"    __syncthreads(); "                                                                                             \
"    if (threadIdx.x < lenA) { "                                                                                    \
"        dstKey[dstPosA] = keyA; "                                                                                  \
"\n#ifdef ARGSORT\n"                                                                                                \
"        dstVal[dstPosA] = valA; "                                                                                  \
"\n#endif\n"                                                                                                        \
"    } "                                                                                                            \
"    if (threadIdx.x < lenB) { "                                                                                    \
"        dstKey[dstPosB] = keyB; "                                                                                  \
"\n#ifdef ARGSORT\n"                                                                                                \
"        dstVal[dstPosB] = valB; "                                                                                  \
"\n#endif\n"                                                                                                        \
"    } "                                                                                                            \
"} "                                                                                                                \
"extern \"C\" __global__ void mergeElementaryIntervalsKernel( "                                                     \
"    t_key *d_DstKey, "                                                                                             \
"    size_t dstOff,"                                                                                                \
"    t_key *d_SrcKey, "                                                                                             \
"    size_t srcOff,"                                                                                                \
"\n#ifdef ARGSORT\n"                                                                                                \
"    t_arg *d_DstArg, "                                                                                             \
"    size_t dstArgOff, "                                                                                            \
"    t_arg *d_SrcArg, "                                                                                             \
"    size_t srcArgOff, "                                                                                            \
"\n#endif\n"                                                                                                        \
"    unsigned int *d_LimitsA, "                                                                                     \
"    size_t limAOff,"                                                                                               \
"    unsigned int *d_LimitsB, "                                                                                     \
"    size_t limBOff,"                                                                                               \
"    unsigned int stride, "                                                                                         \
"    unsigned int N, "                                                                                              \
"    unsigned int sortDir"                                                                                          \
") "                                                                                                                \
"{ "                                                                                                                \
"    d_DstKey = (t_key*) (((char*)d_DstKey)+ dstOff);"                                                              \
"    d_SrcKey = (t_key*) (((char*)d_SrcKey)+ srcOff);"                                                              \
"    d_LimitsA = (unsigned int*) (((char*)d_LimitsA)+ limAOff);"                                                    \
"    d_LimitsB = (unsigned int*) (((char*)d_LimitsB)+ limBOff);"                                                    \
"\n#ifdef ARGSORT\n"                                                                                                \
"    d_DstArg = (t_arg*) (((char*)d_DstArg)+ dstArgOff); "                                                          \
"    d_SrcArg = (t_arg*) (((char*)d_SrcArg)+ srcArgOff);"                                                           \
"    __shared__ t_arg s_arg[2 * SAMPLE_STRIDE]; "                                                                   \
"\n#endif\n"                                                                                                        \
"    __shared__ t_key s_key[2 * SAMPLE_STRIDE]; "                                                                   \
"    const unsigned int   intervalI = blockIdx.x & ((2 * stride) / SAMPLE_STRIDE - 1); "                            \
"    const unsigned int segmentBase = (blockIdx.x - intervalI) * SAMPLE_STRIDE; "                                   \
"    d_SrcKey += segmentBase; "                                                                                     \
"    d_DstKey += segmentBase; "                                                                                     \
"\n#ifdef ARGSORT\n"                                                                                                \
"    d_DstArg += segmentBase; "                                                                                     \
"    d_SrcArg += segmentBase; "                                                                                     \
"\n#endif\n"                                                                                                        \
"    __shared__ unsigned int startSrcA, startSrcB, lenSrcA, lenSrcB, startDstA, startDstB; "                        \
"    if (threadIdx.x == 0) { "                                                                                      \
"        unsigned int segmentElementsA = stride; "                                                                  \
"        unsigned int segmentElementsB = min(stride, N - segmentBase - stride); "                                   \
"        unsigned int  segmentSamplesA = getSampleCount(segmentElementsA); "                                        \
"        unsigned int  segmentSamplesB = getSampleCount(segmentElementsB); "                                        \
"        unsigned int   segmentSamples = segmentSamplesA + segmentSamplesB; "                                       \
"        startSrcA    = d_LimitsA[blockIdx.x]; "                                                                    \
"        startSrcB    = d_LimitsB[blockIdx.x]; "                                                                    \
"        unsigned int endSrcA = (intervalI + 1 < segmentSamples) ? d_LimitsA[blockIdx.x + 1] : segmentElementsA; "  \
"        unsigned int endSrcB = (intervalI + 1 < segmentSamples) ? d_LimitsB[blockIdx.x + 1] : segmentElementsB; "  \
"        lenSrcA      = endSrcA - startSrcA; "                                                                      \
"        lenSrcB      = endSrcB - startSrcB; "                                                                      \
"        startDstA    = startSrcA + startSrcB; "                                                                    \
"        startDstB    = startDstA + lenSrcA; "                                                                      \
"    } "                                                                                                            \
"    __syncthreads(); "                                                                                             \
"    if (threadIdx.x < lenSrcA) { "                                                                                 \
"        s_key[threadIdx.x +             0] = d_SrcKey[0 + startSrcA + threadIdx.x]; "                              \
"\n#ifdef ARGSORT\n"                                                                                                \
"        s_arg[threadIdx.x +             0] = d_SrcArg[0 + startSrcA + threadIdx.x]; "                              \
"\n#endif\n"                                                                                                        \
"    } "                                                                                                            \
"    if (threadIdx.x < lenSrcB) { "                                                                                 \
"        s_key[threadIdx.x + SAMPLE_STRIDE] = d_SrcKey[stride + startSrcB + threadIdx.x]; "                         \
"\n#ifdef ARGSORT\n"                                                                                                \
"        s_arg[threadIdx.x + SAMPLE_STRIDE] = d_SrcArg[stride + startSrcB + threadIdx.x]; "                         \
"\n#endif\n"                                                                                                        \
"    } "                                                                                                            \
"    __syncthreads(); "                                                                                             \
"    merge<t_key>( "                                                                                                \
"        s_key, "                                                                                                   \
"        s_key + 0, "                                                                                               \
"        s_key + SAMPLE_STRIDE, "                                                                                   \
"\n#ifdef ARGSORT\n"                                                                                                \
"        s_arg, "                                                                                                   \
"        s_arg + 0, "                                                                                               \
"        s_arg + SAMPLE_STRIDE, "                                                                                   \
"\n#endif\n"                                                                                                        \
"        lenSrcA, SAMPLE_STRIDE, "                                                                                  \
"        lenSrcB, SAMPLE_STRIDE, "                                                                                  \
"        sortDir "                                                                                                  \
"    ); "                                                                                                           \
"    __syncthreads(); "                                                                                             \
"    if (threadIdx.x < lenSrcA) { "                                                                                 \
"        d_DstKey[startDstA + threadIdx.x] = s_key[threadIdx.x]; "                                                  \
"\n#ifdef ARGSORT\n"                                                                                                \
"        d_DstArg[startDstA + threadIdx.x] = s_arg[threadIdx.x];"                                                   \
"\n#endif\n"                                                                                                        \
"    } "                                                                                                            \
"    if (threadIdx.x < lenSrcB) { "                                                                                 \
"        d_DstKey[startDstB + threadIdx.x] = s_key[lenSrcA + threadIdx.x]; "                                        \
"\n#ifdef ARGSORT\n"                                                                                                \
"        d_DstArg[startDstB + threadIdx.x] = s_arg[lenSrcA + threadIdx.x];"                                         \
"\n#endif\n"                                                                                                        \
"    } "                                                                                                            \
"}\n";
static int mergeElementaryIntervals(
    GpuArray *d_DstKey,
    GpuArray *d_SrcKey,
    GpuArray *d_DstArg,
    GpuArray *d_SrcArg,
    GpuSortData *msData,
    unsigned int stride,
    GpuSortConfig *msConfig,
    GpuKernel *k_merge,
    gpucontext *ctx
)
{
  unsigned int lastSegmentElements = msConfig->Nfloor % (2 * stride);
  unsigned int mergePairs = (lastSegmentElements > stride) ? 
                            getSampleCount(msConfig->Nfloor) : 
                            (msConfig->Nfloor - lastSegmentElements) / SAMPLE_STRIDE;

  size_t ls, gs;
  unsigned int p = 0;
  int err = GA_NO_ERROR;

  ls = SAMPLE_STRIDE;
  gs = mergePairs;

  err = GpuKernel_setarg(k_merge, p++, d_DstKey->data);
  if (err != GA_NO_ERROR) return err;

  err = GpuKernel_setarg(k_merge, p++, &d_DstKey->offset);
  if (err != GA_NO_ERROR) return err;

  err = GpuKernel_setarg(k_merge, p++, d_SrcKey->data);
  if (err != GA_NO_ERROR) return err;

  err = GpuKernel_setarg(k_merge, p++, &d_SrcKey->offset);
  if (err != GA_NO_ERROR) return err;

  if (msConfig->argSortFlg) {
    err = GpuKernel_setarg(k_merge, p++, d_DstArg->data);
    if (err != GA_NO_ERROR) return err;

    err = GpuKernel_setarg(k_merge, p++, &d_DstArg->offset);
    if (err != GA_NO_ERROR) return err;

    err = GpuKernel_setarg(k_merge, p++, d_SrcArg->data);
    if (err != GA_NO_ERROR) return err;

    err = GpuKernel_setarg(k_merge, p++, &d_SrcArg->offset);
    if (err != GA_NO_ERROR) return err;
  }

  err = GpuKernel_setarg(k_merge, p++, msData->d_LimitsA.data);
  if (err != GA_NO_ERROR) return err;

  err = GpuKernel_setarg(k_merge, p++, &msData->d_LimitsA.offset);
  if (err != GA_NO_ERROR) return err;

  err = GpuKernel_setarg(k_merge, p++, msData->d_LimitsB.data);
  if (err != GA_NO_ERROR) return err;

  err = GpuKernel_setarg(k_merge, p++, &msData->d_LimitsB.offset);
  if (err != GA_NO_ERROR) return err;

  err = GpuKernel_setarg(k_merge, p++, &stride);
  if (err != GA_NO_ERROR) return err;

  err = GpuKernel_setarg(k_merge, p++, &msConfig->Nfloor);
  if (err != GA_NO_ERROR) return err;

  err = GpuKernel_setarg(k_merge, p++, &msConfig->sortDirFlg);
  if (err != GA_NO_ERROR) return err;

  err = GpuKernel_call(k_merge, 1, &gs, &ls, 0, NULL);
  if (err != GA_NO_ERROR) return err;
 
  return err;
}

#define NUMARGS_MERGE_GLB 8
int type_args_merge_glb[NUMARGS_MERGE_GLB] = {GA_BUFFER, GA_SIZE, GA_BUFFER, GA_SIZE, 
                                              GA_UINT, GA_UINT, GA_UINT, GA_UINT};
#define NUMARGS_MERGE_GLB_ARG 12
int type_args_merge_glb_arg[NUMARGS_MERGE_GLB_ARG] = {GA_BUFFER, GA_SIZE, GA_BUFFER, GA_SIZE, GA_BUFFER, GA_SIZE, 
                                                      GA_BUFFER, GA_SIZE, GA_UINT, GA_UINT, GA_UINT, GA_UINT};

static const char *code_merge_glb =                                                                                     \
"extern \"C\" __global__ void mergeGlobalMemKernel( "                                                                   \
"    t_key *d_DstKey, "                                                                                                 \
"    size_t dstOff, "                                                                                                   \
"    t_key *d_SrcKey, "                                                                                                 \
"    size_t srcOff, "                                                                                                   \
"\n#ifdef ARGSORT\n"                                                                                                    \
"    t_arg *d_DstArg, "                                                                                                 \
"    size_t dstArgOff, "                                                                                                \
"    t_arg *d_SrcArg, "                                                                                                 \
"    size_t srcArgOff, "                                                                                                \
"\n#endif\n"                                                                                                            \
"    unsigned int segmentSizeA, "                                                                                       \
"    unsigned int segmentSizeB, "                                                                                       \
"    unsigned int N, "                                                                                                  \
"    unsigned int sortDir "                                                                                             \
") "                                                                                                                    \
"{ "                                                                                                                    \
"    d_DstKey = (t_key*) (((char*)d_DstKey)+ dstOff);"                                                                  \
"    d_SrcKey = (t_key*) (((char*)d_SrcKey)+ srcOff);"                                                                  \
"\n#ifdef ARGSORT\n"                                                                                                    \
"    d_DstArg = (t_arg*) (((char*)d_DstArg)+ dstArgOff); "                                                              \
"    d_SrcArg = (t_arg*) (((char*)d_SrcArg)+ srcArgOff);"                                                               \
"\n#endif\n"                                                                                                            \
"    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; "                                                        \
"    t_key *segmentPtrA = d_SrcKey; "                                                                                   \
"    t_key *segmentPtrB = d_SrcKey + segmentSizeA; "                                                                    \
"    unsigned int idxSegmentA = idx % segmentSizeA; "                                                                   \
"    unsigned int idxSegmentB = idx - segmentSizeA; "                                                                   \
"    if (idx >= N) "                                                                                                    \
"        return; "                                                                                                      \
"    t_key value = d_SrcKey[idx]; "                                                                                     \
"\n#ifdef ARGSORT\n"                                                                                                    \
"    t_arg arg = d_SrcArg[idx]; "                                                                                       \
"\n#endif\n"                                                                                                            \
"    unsigned int dstPos; "                                                                                             \
"    if (idx < segmentSizeA) { "                                                                                        \
"        dstPos = binarySearchLowerBoundExclusive<t_key>(value, segmentPtrB, 0, segmentSizeB, sortDir) + idxSegmentA;"  \
"    } "                                                                                                                \
"    else { "                                                                                                           \
"        dstPos = binarySearchLowerBoundInclusive<t_key>(value, segmentPtrA, 0, segmentSizeA, sortDir) + idxSegmentB;"  \
"    } "                                                                                                                \
"    d_DstKey[dstPos] = value; "                                                                                        \
"\n#ifdef ARGSORT\n"                                                                                                    \
"    d_DstArg[dstPos] = arg; "                                                                                          \
"\n#endif\n"                                                                                                            \
"}\n";

static int mergeGlobalMem(
    GpuArray *d_DstKey,
    GpuArray *d_SrcKey,
    GpuArray *d_DstArg,
    GpuArray *d_SrcArg,
    GpuSortConfig *msConfig,
    GpuKernel *k_merge_global,
    gpucontext *ctx
)
{
  size_t ls, gs;
  unsigned int p = 0;
  int err = GA_NO_ERROR;
  unsigned int NleftC = (unsigned int)msConfig->Nleft;

  ls = 256;
  gs = iDivUp(msConfig->dims, (unsigned int)ls);

  err = GpuKernel_setarg(k_merge_global, p++, d_DstKey->data);
  if (err != GA_NO_ERROR) return err;

  err = GpuKernel_setarg(k_merge_global, p++, &d_DstKey->offset);
  if (err != GA_NO_ERROR) return err;

  err = GpuKernel_setarg(k_merge_global, p++, d_SrcKey->data);
  if (err != GA_NO_ERROR) return err;

  err = GpuKernel_setarg(k_merge_global, p++, &d_SrcKey->offset);
  if (err != GA_NO_ERROR) return err;

  if (msConfig->argSortFlg) {
    err = GpuKernel_setarg(k_merge_global, p++, d_DstArg->data);
    if (err != GA_NO_ERROR) return err;

    err = GpuKernel_setarg(k_merge_global, p++, &d_DstArg->offset);
    if (err != GA_NO_ERROR) return err;

    err = GpuKernel_setarg(k_merge_global, p++, d_SrcArg->data);
    if (err != GA_NO_ERROR) return err;

    err = GpuKernel_setarg(k_merge_global, p++, &d_SrcArg->offset);
    if (err != GA_NO_ERROR) return err;
  }

  err = GpuKernel_setarg(k_merge_global, p++, &msConfig->Nfloor);
  if (err != GA_NO_ERROR) return err;

  err = GpuKernel_setarg(k_merge_global, p++, &NleftC);
  if (err != GA_NO_ERROR) return err;

  err = GpuKernel_setarg(k_merge_global, p++, &msConfig->dims);
  if (err != GA_NO_ERROR) return err;

  err = GpuKernel_setarg(k_merge_global, p++, &msConfig->sortDirFlg);
  if (err != GA_NO_ERROR) return err;

  err = GpuKernel_call(k_merge_global, 1, &gs, &ls, 0, NULL);
  if (err != GA_NO_ERROR) return err;

  return err;
}

// Generate type specific GPU code
static int genMergeSortTypeCode(strb *str, GpuSortConfig *msConfig)
{
  if (msConfig->argSortFlg) {
    strb_appends(str, "\n#define ARGSORT\n");
    strb_appendf(str, "typedef %s t_arg;\n", ctype(msConfig->typecodeArg));
    strb_appendf(str, "#define MAX_NUM_ARG %u \n#define MIN_NUM_ARG %u \n",  (msConfig->typecodeArg==GA_ULONG) ? ULONG_MAX : UINT_MAX, 0);
  }
  // Generate typedef for the data type to be sorted    
  strb_appendf(str, "typedef %s t_key;\n", ctype(msConfig->typecodeKey));

  // Generate macro for MIN and MAX value of a given data type
  switch (msConfig->typecodeKey){
  case GA_UINT:
    strb_appendf(str, "#define MAX_NUM %u \n#define MIN_NUM %u \n", UINT_MAX, 0);
    break;
  case GA_INT:
    strb_appendf(str, "#define MAX_NUM %d \n#define MIN_NUM %d \n", INT_MAX, INT_MIN);
    break;
  case GA_FLOAT:
    strb_appendf(str, "#define MAX_NUM %f \n#define MIN_NUM %f \n", FLT_MAX, -FLT_MAX);
    break;
  case GA_DOUBLE:
    strb_appendf(str, "#define MAX_NUM %g \n#define MIN_NUM %g \n", DBL_MAX, -DBL_MAX);
    break;
  case GA_UBYTE:
    strb_appendf(str, "#define MAX_NUM %u \n#define MIN_NUM %u \n", UCHAR_MAX, 0);
    break;
  case GA_BYTE:
    strb_appendf(str, "#define MAX_NUM %d \n#define MIN_NUM %d \n", SCHAR_MAX, SCHAR_MIN);
    break;
  case GA_USHORT:
    strb_appendf(str, "#define MAX_NUM %u \n#define MIN_NUM %u \n", USHRT_MAX, 0);
    break;
  case GA_SHORT:
    strb_appendf(str, "#define MAX_NUM %d \n#define MIN_NUM %d \n", SHRT_MAX, SHRT_MIN);
    break;
  default:
    return GA_IMPL_ERROR;
    break;
  }
  return strb_error(str);
}

#define NSTR_BITONIC 3
#define NSTR_RANKS 4
#define NSTRINGS_RKS_IDX 4
#define NSTRINGS_MERGE 4
#define NSTRINGS_MERGE_GLB 4
static int compileKernels(GpuKernel *k_bitonic, GpuKernel *k_ranks, GpuKernel *k_ranks_idxs, GpuKernel *k_merge,
                           GpuKernel *k_merge_global, gpucontext *ctx, GpuSortConfig *msConfig)
{
  char *err_str = NULL;
  int err = GA_NO_ERROR;

  size_t lens_bitonic[NSTR_BITONIC]         = {0, strlen(code_helper_funcs), strlen(code_bitonic_smem)};
  size_t lens_ranks[NSTR_RANKS]             = {0, strlen(code_helper_funcs), strlen(code_bin_search), strlen(code_sample_ranks)};
  size_t lens_rks_idx[NSTRINGS_RKS_IDX]     = {0, strlen(code_helper_funcs), strlen(code_bin_search), strlen(code_ranks_idxs)};
  size_t lens_merge[NSTRINGS_MERGE]         = {0, strlen(code_helper_funcs), strlen(code_bin_search), strlen(code_merge)};
  size_t lens_merge_glb[NSTRINGS_MERGE_GLB] = {0, strlen(code_helper_funcs), strlen(code_bin_search), strlen(code_merge_glb)};

  const char *codes_bitonic[NSTR_BITONIC]         = {NULL, code_helper_funcs, code_bitonic_smem};
  const char *codes_ranks[NSTR_RANKS]             = {NULL, code_helper_funcs, code_bin_search, code_sample_ranks};
  const char *codes_rks_idx[NSTRINGS_RKS_IDX]     = {NULL, code_helper_funcs, code_bin_search, code_ranks_idxs};
  const char *codes_merge[NSTRINGS_MERGE]         = {NULL, code_helper_funcs, code_bin_search, code_merge};
  const char *codes_merge_glb[NSTRINGS_MERGE_GLB] = {NULL, code_helper_funcs, code_bin_search, code_merge_glb};

  unsigned int nargs;
  int *types;

  strb sb = STRB_STATIC_INIT;
  err = genMergeSortTypeCode(&sb, msConfig);
  if (err != GA_NO_ERROR) return err;

  // Compile Bitonic sort Kernel  
  lens_bitonic[0] = sb.l;
  codes_bitonic[0] = sb.s;
  if (msConfig->argSortFlg) {
    nargs = NUMARGS_BITONIC_KERNEL_ARG;
    types = type_args_bitonic_arg;
  }
  else {
    nargs = NUMARGS_BITONIC_KERNEL;
    types = type_args_bitonic;
  }
  err = GpuKernel_init( k_bitonic,
                        ctx,
                        NSTR_BITONIC,
                        codes_bitonic,
                        lens_bitonic,
                        "bitonicSortSharedKernel",
                        nargs,
                        types,
                        flags,
                        &err_str
                      );
  if (err != GA_NO_ERROR) {
   printf("error kernel init: %s \n", gpuarray_error_str(err));
   printf("error backend: %s \n", err_str);
   return err;
  }

  // Compile ranks kernel
  lens_ranks[0]  = sb.l;
  codes_ranks[0] = sb.s;
  err = GpuKernel_init( k_ranks,
                        ctx,
                        NSTR_RANKS, 
                        codes_ranks,
                        lens_ranks,
                        "generateSampleRanksKernel",
                        NUMARGS_SAMPLE_RANKS,
                        type_args_ranks,
                        flags,
                        &err_str
                      );
  if (err != GA_NO_ERROR) {
    printf("error kernel init: %s \n", gpuarray_error_str(err));
    printf("error backend: %s \n", err_str);
    return err;
  }

  // Compile ranks and idxs kernel
  lens_rks_idx[0]  = sb.l;
  codes_rks_idx[0] = sb.s;
  err = GpuKernel_init( k_ranks_idxs,
                        ctx, 
                        NSTRINGS_RKS_IDX, 
                        codes_rks_idx,
                        lens_rks_idx,
                        "mergeRanksAndIndicesKernel",
                        NUMARGS_RANKS_IDXS,
                        type_args_ranks_idxs,
                        flags,
                        &err_str
                      );
  if (err != GA_NO_ERROR) {
    printf("error kernel init: %s \n", gpuarray_error_str(err)); 
    printf("error backend: %s \n", err_str);
    return err;
  }

  if (msConfig->argSortFlg) {
    nargs = NUMARGS_MERGE_ARG;
    types = type_args_merge_arg;
  }
  else {
    nargs = NUMARGS_MERGE;
    types = type_args_merge;
  }
  // Compile merge kernel
  lens_merge[0]  = sb.l;
  codes_merge[0] = sb.s;
  err = GpuKernel_init( k_merge,
                        ctx,
                        NSTRINGS_MERGE, 
                        codes_merge,
                        lens_merge,
                        "mergeElementaryIntervalsKernel",
                        nargs,
                        types,
                        flags,
                        &err_str
                      );
  if (err != GA_NO_ERROR) {
    printf("error kernel init: %s \n", gpuarray_error_str(err));
    printf("error backend: %s \n", err_str);
    return err;
  }

  if (msConfig->argSortFlg) {
    nargs = NUMARGS_MERGE_GLB_ARG;
    types = type_args_merge_glb_arg;
  }
  else {
    nargs = NUMARGS_MERGE_GLB;
    types = type_args_merge_glb;
  }
  // Compile merge global kernel
  lens_merge_glb[0]  = sb.l;
  codes_merge_glb[0] = sb.s;
  err = GpuKernel_init( k_merge_global,
                        ctx, 
                        NSTRINGS_MERGE_GLB, 
                        codes_merge_glb,
                        lens_merge_glb,
                        "mergeGlobalMemKernel",
                        nargs,
                        types,
                        flags,
                        &err_str
                      );
  if (err != GA_NO_ERROR) {
    printf("error kernel init: %s \n", gpuarray_error_str(err));
    printf("error backend: %s \n", err_str);
    return err;
  }
  return err;
}

static int sort(
  GpuArray *d_DstKey,
  GpuArray *d_SrcKey,
  GpuArray *d_DstArg,
  GpuArray *d_SrcArg,
  GpuSortBuff *msBuff,
  GpuSortData  *msData,
  GpuSortConfig *msConfig,
  gpucontext *ctx
)
{
  int err = GA_NO_ERROR;
  size_t lstCopyOffDst, lstCopyOffSrc;
  unsigned int stageCount = 0;
  unsigned int stride;

  GpuArray *ikey, *okey, *iarg, *oarg, *t, *t2;
  GpuKernel k_bitonic, k_ranks, k_ranks_idxs, k_merge, k_merge_global;
  checkErr( compileKernels(&k_bitonic, &k_ranks, &k_ranks_idxs, &k_merge, &k_merge_global, ctx, msConfig) );

  for (stride = SHARED_SIZE_LIMIT; stride < msConfig->Nfloor; stride <<= 1, stageCount++);

  if (stageCount & 1) {
    ikey = &msBuff->BufKey;
    okey = d_DstKey;
    iarg = &msBuff->BufArg;
    oarg = d_DstArg;
  }
  else {
    ikey = d_DstKey;
    okey = &msBuff->BufKey;
    iarg = d_DstArg;
    oarg = &msBuff->BufArg;
  }

  // Bitonic sort for short arrays
  if (msConfig->dims <= SHARED_SIZE_LIMIT) { 

    checkErr( bitonicSortShared(d_DstKey, d_SrcKey, d_DstArg, d_SrcArg, 1, (unsigned int)msConfig->dims, 
                                msConfig->sortDirFlg, 0, msConfig->argSortFlg, &k_bitonic, ctx)
            );
  }
  // Merge - Bitonic sort for bigger arrays
  else {

    unsigned int batchSize = msConfig->Nfloor / SHARED_SIZE_LIMIT;
    unsigned int arrayLength = SHARED_SIZE_LIMIT;
    checkErr( bitonicSortShared(ikey, d_SrcKey, iarg, d_SrcArg, batchSize, arrayLength, 
                                msConfig->sortDirFlg, 0, msConfig->argSortFlg, &k_bitonic, ctx)
            );

    for (stride = SHARED_SIZE_LIMIT; stride < msConfig->Nfloor; stride <<= 1) {
      unsigned int lastSegmentElements = msConfig->Nfloor % (2 * stride);

      //Find sample ranks and prepare for limiters merge
      checkErr( generateSampleRanks(msData, ikey, stride, msConfig, &k_ranks, ctx) );

      //Merge ranks and indices
      checkErr( mergeRanksAndIndices(msData, stride, msConfig, &k_ranks_idxs, ctx) );

      //Merge elementary intervals
      checkErr( mergeElementaryIntervals(okey, ikey, oarg, iarg, msData, stride, msConfig, &k_merge, ctx) );

      if (lastSegmentElements <= stride) {
        //Last merge segment consists of a single array which just needs to be passed through          
        lstCopyOffDst = okey->offset + ((msConfig->Nfloor - lastSegmentElements) * msConfig->typesizeKey);
        lstCopyOffSrc = ikey->offset + ((msConfig->Nfloor - lastSegmentElements) * msConfig->typesizeKey);
        checkErr( gpudata_move(okey->data, lstCopyOffDst, ikey->data, lstCopyOffSrc, 
                               lastSegmentElements * msConfig->typesizeKey)
                );
        
        if (msConfig->argSortFlg) {
          lstCopyOffDst = oarg->offset + ((msConfig->Nfloor - lastSegmentElements) * msConfig->typesizeArg);
          lstCopyOffSrc = iarg->offset + ((msConfig->Nfloor - lastSegmentElements) * msConfig->typesizeArg);
          checkErr( gpudata_move(oarg->data, lstCopyOffDst, iarg->data, lstCopyOffSrc, 
                                 lastSegmentElements * msConfig->typesizeArg)
                  );
        }
      }
      // Swap pointers
      t = ikey;
      ikey = okey;
      okey = t;
      if (msConfig->argSortFlg) {
        t2 = iarg;
        iarg = oarg;
        oarg = t2;
      }
    }
    // If the array is not multiple of 1024, sort the remaining and merge
    if (msConfig->Nleft > 0) {

      checkErr( bitonicSortShared(d_SrcKey, d_DstKey, d_SrcArg, d_DstArg, 1, msConfig->Nleft, 
                              msConfig->sortDirFlg, msConfig->Nfloor, msConfig->argSortFlg, &k_bitonic, ctx)
              );

      // Copy the leftMost segment to the output array of which contains the first sorted sequence
      lstCopyOffDst = d_DstKey->offset + (msConfig->Nfloor * msConfig->typesizeKey);
      lstCopyOffSrc = d_SrcKey->offset + (msConfig->Nfloor * msConfig->typesizeKey);
      checkErr( gpudata_move(d_DstKey->data, lstCopyOffDst, d_SrcKey->data, lstCopyOffSrc, 
                         msConfig->Nleft * msConfig->typesizeKey)
              );

      if (msConfig->argSortFlg) {
        lstCopyOffDst = d_DstArg->offset + (msConfig->Nfloor * msConfig->typesizeArg);
        lstCopyOffSrc = d_SrcArg->offset + (msConfig->Nfloor * msConfig->typesizeArg);        
        checkErr( gpudata_move(d_DstArg->data, lstCopyOffDst, d_SrcArg->data, lstCopyOffSrc, 
                           msConfig->Nleft * msConfig->typesizeArg)
                );
      }
      checkErr( mergeGlobalMem(d_SrcKey, d_DstKey, d_SrcArg, d_DstArg, msConfig, &k_merge_global, ctx) );      

      checkErr( GpuArray_copy(d_DstKey, d_SrcKey, GA_C_ORDER) );

      if (msConfig->argSortFlg) {
        checkErr( GpuArray_copy(d_DstArg, d_SrcArg, GA_C_ORDER) );
      }
    }
  }
  return err;
}

static int initArgSort(
  GpuArray *srcArg,
  GpuArray *src,
  GpuSortConfig *msConfig,
  gpucontext *ctx
)
{
  int err = GA_NO_ERROR;
  size_t dims;
  void *tmp;
  unsigned long *lPtr;
  unsigned int  *iPtr;
  unsigned long i;
  
  size_t typeSize = (msConfig->typecodeArg == GA_ULONG) ? sizeof(unsigned long) : sizeof(unsigned int);
  dims = src->dimensions[0] * typeSize;

  tmp = malloc(dims);
  lPtr = (unsigned long*)tmp;
  iPtr = (unsigned int*)tmp;
  
  for (i = 0; i < src->dimensions[0]; ++i) {
    if (msConfig->typecodeArg == GA_ULONG) 
      lPtr[i] = i;
    else
      iPtr[i] = (unsigned int)i;
  }

  err = GpuArray_empty(srcArg, ctx, msConfig->typecodeArg, src->nd, src->dimensions, GA_C_ORDER);
  if (err != GA_NO_ERROR) return err;
  
  err = GpuArray_write(srcArg, tmp, dims);
  if (err != GA_NO_ERROR) return err;

  free(tmp);
  return err;
}

static int initMergeSort(
  GpuSortData *msData,
  GpuSortConfig *msConfig,
  GpuArray *src,
  GpuArray *srcArg,
  gpucontext *ctx
)
{
  int err = GA_NO_ERROR;
  const size_t dims = msConfig->Nfloor / 128;
  unsigned int nd = src->nd;

  err = GpuArray_empty(&msData->d_RanksA, ctx, GA_UINT, nd, &dims, GA_C_ORDER);
  if (err != GA_NO_ERROR) return err;

  err = GpuArray_empty(&msData->d_RanksB, ctx, GA_UINT, nd, &dims, GA_C_ORDER);
  if (err != GA_NO_ERROR) return err;
  
  err = GpuArray_empty(&msData->d_LimitsA, ctx, GA_UINT, nd, &dims, GA_C_ORDER);
  if (err != GA_NO_ERROR) return err;

  err = GpuArray_empty(&msData->d_LimitsB, ctx, GA_UINT, nd, &dims, GA_C_ORDER);
  if (err != GA_NO_ERROR) return err;

  if (msConfig->argSortFlg) {
    initArgSort(srcArg, src, msConfig, ctx);
  }
  return err;
}

static void initMsConfig(GpuSortConfig *msConfig, GpuArray *src, GpuArray *arg, unsigned int sortDir, unsigned int argSort)
{
  msConfig->dims = src->dimensions[0];
  msConfig->Nfloor = roundDown((unsigned int)msConfig->dims, SHARED_SIZE_LIMIT);
  msConfig->Nleft = (unsigned int)msConfig->dims - msConfig->Nfloor;
  msConfig->sortDirFlg = sortDir;
  msConfig->argSortFlg = argSort;
  msConfig->typecodeKey = src->typecode;
  msConfig->typesizeKey = gpuarray_get_elsize(src->typecode);
  if (argSort) {
    assert(arg->typecode == GA_UINT || arg->typecode == GA_ULONG);
    msConfig->typecodeArg = arg->typecode;
    msConfig->typesizeArg = gpuarray_get_elsize(arg->typecode);
  }
}

static int initMsBuff(GpuSortBuff *msBuff, GpuArray *src, gpucontext *ctx, GpuSortConfig *msConfig)
{
  int err = GA_NO_ERROR;

  err = GpuArray_empty(&msBuff->BufKey, ctx, msConfig->typecodeKey, src->nd, src->dimensions, GA_C_ORDER);
  if (err != GA_NO_ERROR) return err;
  
  if (msConfig->argSortFlg) {
    err = GpuArray_empty(&msBuff->BufArg, ctx, msConfig->typecodeArg, src->nd, src->dimensions, GA_C_ORDER);
    if (err != GA_NO_ERROR) return err;
  }

  return err;
}

static void destroyMergeSort(
  GpuSortData *msData,
  GpuSortBuff *msBuff,
  GpuArray *srcArg,
  unsigned int argSort
)
{
  GpuArray_clear(&msData->d_RanksA);
  GpuArray_clear(&msData->d_RanksB);
  GpuArray_clear(&msData->d_LimitsA);
  GpuArray_clear(&msData->d_LimitsB);
  GpuArray_clear(&msBuff->BufKey);
  if (argSort) {
    GpuArray_clear(&msBuff->BufArg);
    GpuArray_clear(srcArg);
  }
}

int GpuArray_sort(
  GpuArray *dstKey,
  GpuArray *srcKey,
  unsigned int sortDir,
  GpuArray *dstArg
)
{
  int err = GA_NO_ERROR;
  gpucontext *ctx = GpuArray_context(srcKey);

  GpuArray srcArg; 
  GpuSortConfig msConfig;
  GpuSortBuff msBuff;
  GpuSortData msData;

  if (srcKey->nd > 1) return GA_IMPL_ERROR;

  initMsConfig(&msConfig, srcKey, dstArg, sortDir, dstArg != NULL ? 1 : 0);

  checkErr( initMsBuff(&msBuff, srcKey, ctx, &msConfig) );

  checkErr( initMergeSort(&msData, &msConfig, srcKey, &srcArg, ctx) );
  
  checkErr( sort(dstKey, srcKey, dstArg, &srcArg, &msBuff, &msData, &msConfig, ctx) );

  destroyMergeSort(&msData, &msBuff, &srcArg, msConfig.sortDirFlg);

  return err;
}