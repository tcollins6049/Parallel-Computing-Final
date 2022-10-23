#include <immintrin.h>
#include <stdlib.h>
#include <iostream>
#include <cstdint>
#include <string.h>
#include <sys/syscall.h>
#include <stddef.h>
#include <unistd.h>

#include "hpc_helpers.h"

void naiveMult(float * A, float * B, float * C, uint64_t M, uint64_t N, uint64_t L);
void transposeAndMult(float * A, float * B, float * C, uint64_t M, uint64_t N, uint64_t L);
void blockedMult(float * A, float * B, float * C, uint64_t M, uint64_t N, uint64_t L, uint64_t blkSz);
void openMPnaiveMult(float * A, float * B, float * C, uint64_t M, uint64_t N, uint64_t L);
void transposeAndMultMP(float * A, float * B, float * C, uint64_t M, uint64_t N, uint64_t L);
void initialize(float * array, uint64_t size);
void compare(float * array1, float * array2, uint64_t size);
void cacheFlush();

#define TESTS 4
int main () 
{
                                //M,     N,      L,      blkSz 
    uint64_t sizes[TESTS][4] = {{1 << 9, 1 << 9, 1 << 5, 16},
                                {1 << 10, 1 << 10, 1 << 5, 16},
                                {1 << 10, 1 << 10, 1 << 6, 32},
                                {1 << 11, 1 << 11, 1 << 6, 32}};

    for (int i = 0; i < TESTS; i++)
    {
        uint64_t M = sizes[i][0];
        uint64_t N = sizes[i][1];
        uint64_t L = sizes[i][2];
        uint64_t blkSz = sizes[i][3];
        if ((blkSz % 8) != 0)
        {
            printf("Error: block size %ld is not a multiple of 8\n", blkSz);
            exit(1);
        }

        printf("\n%ld by %ld TIMES %ld by %ld EQUALS %ld by %ld\n", M, L, L, N, M, N);
        printf("BLOCKSIZE EQUALS %ld\n", blkSz);

        float * A = (float *)aligned_alloc(32, sizeof(float) * M * L);
        float * B = (float *)aligned_alloc(32, sizeof(float) * L * N);    //L rows, N columns
        float * Cn = new float[M * N];                                    //naive multiply
        float * CnMP = new float[M * N];                                  //naive multiply OpenMP
        float * Ct = new float[M * N];                                    //transpose multiply
        float * CtMP = new float[M * N];                                  //transpose multiply OpenMP
        float * Cb = new float[M * N];                                    //blocked multiply
        initialize(A, M * L);
        initialize(B, L * N);

        //perform matrix multiply the naive way to check for correctness
        cacheFlush();
        TIMERSTART(naive_mult)
        naiveMult(A, B, Cn, M, N, L);
        TIMERSTOP(naive_mult)

        cacheFlush();
        TIMERSTART(openMPnaive)
        openMPnaiveMult(A, B, CnMP, M, N, L);
        TIMERSTOP(openMPnaive)
        compare(Cn, CnMP, M * N);
        SPEEDUP(openMPnaive, naive_mult)

        cacheFlush();
        TIMERSTART(transpose_and_mult)
        transposeAndMult(A, B, Ct, M, N, L);
        TIMERSTOP(transpose_and_mult)
        compare(Cn, Ct, M * N);

        cacheFlush();
        TIMERSTART(transpose_and_multMP)
        transposeAndMultMP(A, B, CtMP, M, N, L);
        TIMERSTOP(transpose_and_multMP)
        compare(Cn, CtMP, M * N);
        SPEEDUP(transpose_and_multMP, transpose_and_mult)

        /*
        cacheFlush();
        TIMERSTART(blocked_mult)
        blockedMult(A, B, Cb, M, N, L, blkSz);
        TIMERSTOP(blocked_mult)
        //make sure that the results of the blockedMult are the same as naiveMult
        compare(Cn, Cb, M * N);
        */

        delete A; 
        delete B; 
        delete Cn; 
        delete CnMP; 
        delete Ct; 
        delete CtMP; 
        delete Cb; 
    }
}

/* 
 * Initialize an array of size floats to random values between 0 and 9.
 */
void initialize(float * array, uint64_t size)
{
   for (uint64_t i = 0; i < size; i++) array[i] = rand() % 10;
}   

/* 
 * Compare two arrays.  Output a message and exit program if not the same. 
 */
void compare(float * array1, float * array2, uint64_t size)
{
    for (uint64_t i = 0; i < size; i++)
    {
        if (array1[i] != array2[i])
        {
            printf("Error: arrays do not match\n");
            printf("       index %ld: %6.2f != %6.2f\n", i, array1[i], array2[i]);
            exit(1);
        }
    }

}

/* 
 * Perform a matrix multiply A * B and store the result in array C.
 * Use the naive matrix multiply technique.
 * A is of size M by L, 
 * B is of size L by N, 
 * C is of size M by N
 */
void naiveMult(float * A, float * B, float * C, uint64_t M, uint64_t N, uint64_t L)
{
    float accum;
    uint64_t i,j,k;
    for (i = 0; i < M; i++)
    {
        for (j = 0; j < N; j++)
        {
            accum = 0;
            for (k = 0; k < L; k++)
            {
                accum += A[i*L+k]*B[k*N+j];
            }
            C[i*N+j] = accum;
        }
    }
}

/* 
 * Perform a matrix multiply A * B and store the result in array C.
 * Transpose the B array before doing the matrix multiply.
 * A is of size M by L, 
 * B is of size L by N, 
 * C is of size M by N
 */
void transposeAndMult(float * A, float * B, float * C, uint64_t M, uint64_t N, uint64_t L)
{
    uint64_t i,j,k;
    float * Bt = new float[N*L];
    for (k = 0; k < L; k++)
        for (j = 0; j < N; j++)
            Bt[j*L+k] = B[k*N+j];

    float accum;
    for (i = 0; i < M; i++)
    {
        for (j = 0; j < N; j++)
        {
            accum = 0;
            for (k = 0; k < L; k++)
                accum += A[i*L+k]*Bt[j*L+k];
            C[i*N+j] = accum;
        }
    }
    delete Bt;
}

/* 
 * Perform a matrix multiply A * B and store the result in array.
 * Use the blocking matrix multiply technique:
 * https://csapp.cs.cmu.edu/public/waside/waside-blocking.pdf
 * A is of size M by L, 
 * B is of size L by N, 
 * C is of size M by N
*/
void blockedMult(float * A, float * B, float * C, uint64_t M, uint64_t N, uint64_t L,
                    uint64_t blkSz)
{
    /* Add your blocking matrix multiply from programming assignment 1. */
   uint64_t i, j, k, kk, jj;
   double sum;
   //uint64_t el = blkSz * (L/blkSz);
   //uint64_t en = blkSz * (N/blkSz);
   //for (x = 0; x < M * N; x++) {
    //   C[x] = 0.0;
   //}

   for (kk = 0; kk < L; kk += blkSz) {
       for (jj = 0; jj < N; jj += blkSz) {
           for (i = 0; i < M; i++) {
               for (j = jj; j < jj + blkSz; j++) {
                   sum = C[i * N + j];
                   for (k = kk; k < kk + blkSz; k++) {
                       sum += A[i * L + k] * B[k * N + j];
                   }
                   C[i * N + j] = sum;
               }
           }
       }
   }  
}

void openMPnaiveMult(float * A, float * B, float * C, uint64_t M, uint64_t N, uint64_t L)
{
    uint64_t i;

    auto naiveMultMP2 = [](uint64_t i, uint64_t j, float * A, float * B, float * C, uint64_t N, uint64_t L) {
        float accum = 0;
        uint64_t k;
        for (k = 0; k < L; k++)
        {
            accum += A[i*L+k]*B[k*N+j];
        }
        C[i*N+j] = accum;
    };

    auto naiveMultMP = [&naiveMultMP2](uint64_t i, float * A, float * B, float * C, uint64_t N, uint64_t L) {
        uint64_t j;
        for (j = 0; j < N; j++)
        {
            naiveMultMP2(i, j, A, B, C, N, L);
        }
    };

    #pragma omp parallel for schedule(guided) num_threads(M/8) //Modified this line for graphs
    for (i = 0; i < M; i++)
    {
        naiveMultMP(i, A, B, C, N, L);
    }
}

void transposeAndMultMP(float * A, float * B, float * C, uint64_t M, uint64_t N, uint64_t L) {
    uint64_t i, k, j;
    float * Bt = new float[N*L];
    for (k = 0; k < L; k++)
        for (j = 0; j < N; j++)
            Bt[j*L+k] = B[k*N+j];

    auto transposeMP = [](uint64_t i, float * A, float * Bt, float * C, uint64_t N, uint64_t L) {
        uint64_t k, j;
        float accum;
        for (j = 0; j < N; j++)
        {
            accum = 0;
            for (k = 0; k < L; k++)
                accum += A[i*L+k]*Bt[j*L+k];
            C[i*N+j] = accum;
        }
    };

    #pragma omp parallel for schedule(guided) num_threads(M/8) //Modified this line for graphs
    for (i = 0; i < M; i++)
    {
        transposeMP(i, A, Bt, C, N, L);
    }
    delete Bt;
}

/*
 * Call this function before the matrix multiply to flush the cache so that none of A or B are in the
 * cache before the multiply is performed.
 * I couldn't find a fool proof way to do this. cacheflush isn't available to us. 
 * But this seems to work.
 */
void cacheFlush()
{
    /* Going to sleep for a few seconds causes a context switch */
    sleep(3);
}
