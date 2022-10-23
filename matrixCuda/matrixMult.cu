#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cuda_runtime.h>
#include "helpers.h"
#include <sys/syscall.h>
#include <unistd.h>
#define NAIVE 1
#define TILED 2
#define DEBUG 0
//#define TILE_SZ 1
//#define TILE_SZ 2
//#define TILE_SZ 4
//#define TILE_SZ 8
#define TILE_SZ 16
//#define TILE_SZ 32

// transfer constants
#define H2D (cudaMemcpyHostToDevice)
#define D2H (cudaMemcpyDeviceToHost)

__global__ 
void matrixMultNaiveKernel(float * A, float * B, float * C, int width);

__global__ 
void matrixMultTiledKernel(float * A, float * B, float * C, int width);

static void parseArgs(int argc, char * argv[], int * which, bool * doTime);
static void naiveMult(float * A, float * B, float * C, uint64_t M, uint64_t N, uint64_t L);
static void cacheFlush();
static void printMatrix(const char * header, float * matrix, int width);
static void initialize(float * array, uint64_t size);
static void compare(float * result1, float * result2, int n);

float matrixMult(float * A, float * B, float * C, int width, int which, int M, int N, int L);

/*
   driver for the matrix multiply program.  
*/
int main(int argc, char * argv[])
{
    //parameters for GPU version
    int which;
    bool doTime;

    //parse the command line arguments
    parseArgs(argc, argv, &which, &doTime);

    //CPU time, GPU time, speedup
    float cpu_time, gpu_time, speedup;

                                //M,     N,      L 
    uint64_t sizes[4][3] = {{1 << 6, 1 << 6, 1 << 6}, 
                                {1 << 8, 1 << 8, 1 << 8},
                                {1 << 10, 1 << 10, 1 << 10},
                                {1 << 11, 1 << 11, 1 << 11},};

    int i = 2;
    uint64_t M = sizes[i][0];
    uint64_t N = sizes[i][1];
    uint64_t L = sizes[i][2];
    uint64_t matrixDim = M;
    float * A = (float *)aligned_alloc(32, sizeof(float) * M * L);
    float * B = (float *)aligned_alloc(32, sizeof(float) * L * N);
    float * C = new float[M * N];
    float * Cn = new float[M * N];
    initialize(A, M * L);
    initialize(B, L * N);

    naiveMult(A, B, Cn, M, N, L);

    if (DEBUG)
    {
        printMatrix("CPU Result", Cn, matrixDim);
    }

    gpu_time = matrixMult(A, B, C, matrixDim, which, M, N, L);
    if (DEBUG)
        printMatrix("GPU Result", C, matrixDim);

    //compare GPU and CPU results 
    compare(C, Cn, matrixDim);
    printf("GPU result is correct.\n");

    //run the GPU version multiple times to get somewhat accurate timing
    if (doTime == true)
    {
        //Because the GPU time varies greatly, we will run the GPU code
        //multiple times and compute an average.
        //In addition, ignore the first couple of times since it takes
        //time for the GPU to "warm-up."
        printf("Timing the kernel. This may take a bit.\n");
        
        int i;
        cpu_time = 0;
        gpu_time = 0;
        for (i = 0; i < 5; i++) {
            cacheFlush();
            TIMERSTART(cpuT)
            naiveMult(A, B, Cn, M, N, L);
            TIMERSTOP(cpuT)
            cpu_time += TIMEELAPSED(cpuT)
        }
        cpu_time = cpu_time / 5.0;

        for (i = 0; i < 5; i++) {
            gpu_time += matrixMult(A, B, C, matrixDim, which, M, N, L);
        }
        gpu_time = gpu_time/5.0;

        //Output the times and the speedup
        printf("\nTiming\n");
        printf("------\n");
        printf("CPU: \t\t\t%f msec\n", cpu_time);
        printf("GPU: \t\t\t%f msec\n", gpu_time);
        speedup = cpu_time/gpu_time;
        printf("Speedup: \t\t%f\n", speedup);
    }

    //free dynamically allocated data
    free(A);
    free(B);
    free(Cn);
    free(C);
}

/*
    matrixMult
    This function
    Allocates space for matrices and then launches the appropriate kernel.
    Once the kernel is done, the function gets the result from GPU memory.
    Inputs:
    A - pointer to the float array which is the first matrix
    B - pointer to the float array which is the second matrix
    C - pointer to the float array which is the result matrix
    width - width and height of the matrices
    blkDim - Block Dimension
    which - the value of the kernel, 1 -> Naive 2 -> Tiled
    M - value of M
    N - value of N
    L - value of L
*/
float matrixMult(float * A, float * B, float * C, int width, int which, int M, int N, int L)
{
    float * gpuA, * gpuB, * gpuC;  //pointers to matrices for GPU
   
    TIMERSTART(gpuTime)

    //Allocate space in GPU memory for A matrix
    cudaMalloc((void **)&gpuA, sizeof(float) * M * L);            CUERR

    //Copy A from CPU memory to GPU memory
    cudaMemcpy(gpuA, A, sizeof(float) * M * L, H2D);          CUERR

    //Allocate space in GPU memory for B matrix
    cudaMalloc((void **)&gpuB, sizeof(float) * L * N);            CUERR

    //Copy B from CPU memory to GPU memory
    cudaMemcpy(gpuB, B, sizeof(float) * L * N, H2D);          CUERR

    //Allocate space in GPU memory for result matrix
    cudaMalloc((void **)&gpuC, sizeof(float) * M * N);           CUERR

    //Launch the appropriate kernel
    if (which == NAIVE)
    {
        dim3 block(TILE_SZ, TILE_SZ, 1);
        dim3 grid(ceil((float)width/TILE_SZ), ceil((float)width/TILE_SZ), 1);
        matrixMultNaiveKernel<<<grid, block>>>(gpuA, gpuB, gpuC, width);                CUERR
    } else {
        dim3 block(TILE_SZ, TILE_SZ, 1);
        dim3 grid(ceil((float)width/TILE_SZ), ceil((float)width/TILE_SZ), 1);
        matrixMultTiledKernel<<<grid, block>>>(gpuA, gpuB, gpuC, width);                CUERR
    }
    //wait for threads to finish
    cudaDeviceSynchronize();                                                 CUERR
    //copy C from GPU memory to CPU memory
    cudaMemcpy(C, gpuC, sizeof(float) * width * width, D2H);                CUERR

    //free dynamically  allocated memory
    cudaFree(gpuA);                                                       CUERR
    cudaFree(gpuB);                                                       CUERR 
    cudaFree(gpuC);                                                       CUERR

    //stop the timer
    TIMERSTOP(gpuTime)
    return TIMEELAPSED(gpuTime);
}

/*  
    matrixMultNaiveKernel
    This kernel performs naive matrix multiplication
    and stores the result in the C matrix.
    Each matrix is of size width by width.
    Inputs:
    A - pointer to the float array which is the first matrix
    B - pointer to the float array which is the second matrix
    C - pointer to the float array which is the result matrix
    width - width and height of the matrices
*/
__global__ 
void matrixMultNaiveKernel(float * A, float * B, float * C, int width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if ((row < width) && (col < width)) {
        float num = 0;
        for (int i = 0; i < width; ++i) {
            num += A[row * width + i] * B[i * width + col];
        }
        C[row*width+col] = num;
    }
}

/*  
    matrixMultTiledKernel
    This kernel performs tiled matrix multiplication
    and stores the result in the C matrix.
    Each matrix is of size width by width.
    Inputs:
    A - pointer to the float array which is the first matrix
    B - pointer to the float array which is the second matrix
    C - pointer to the float array which is the result matrix
    width - width and height of the matrices
*/
__global__ 
void matrixMultTiledKernel(float * A, float * B, float * C, int width) {
    __shared__ float sA[TILE_SZ][TILE_SZ];
    __shared__ float sB[TILE_SZ][TILE_SZ];
    int i;

    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int by = blockIdx.y;
    int ty = threadIdx.y;
    int row = by * TILE_SZ + ty;
    int col = bx * TILE_SZ + tx;
    float num = 0;

    for (i = 0; i < ceil(width/(float)TILE_SZ); ++i) {
        if ((row < width) && (i * TILE_SZ + tx) < width)
            sA[ty][tx] = A[row * width + i * TILE_SZ + tx];
        if ((i * TILE_SZ + ty) < width && (col < width))
            sB[ty][tx] = B[((i * TILE_SZ) + ty) * width + col];
        
        __syncthreads();

        for (int j = 0; j < TILE_SZ; ++j) {
            num += sA[ty][j] * sB[j][tx];
        }
        __syncthreads();
    }
    if ((row < width) && (col < width))
        C[row * width + col] = num;
}

/* 
    parseArgs
    This function parses the command line arguments to get
    the GPU kernel to be executed and whether timing results should be
    produced.
    Inputs:
    argc - count of the number of command line arguments
    argv - array of command line arguments
    whichP - which kernel to execute
    doTimeP - pointer to a bool that is set to true or false if timing
              is to be performed
*/
void parseArgs(int argc, char * argv[], int * whichP, bool * doTimeP)
{
    int i;
    //set the parameters to their defaults
    int which = NAIVE;
    bool doTime = false;

    //loop through the command line arguments
    for (i = 1; i < argc; i++)
    {
       if (strcmp(argv[i], "-naive") == 0)
          which = NAIVE;
       else if (strcmp(argv[i], "-tiled") == 0)
          which = TILED;
       else if (strcmp(argv[i], "-time") == 0)
          doTime = true;
    }
    (*whichP) = which;
    (*doTimeP) = doTime;
}

/* 
    Perform a matrix multiply A * B and store the result in array C.
    Use the naive matrix multiply technique.
    A is of size M by L, 
    B is of size L by N, 
    C is of size M by N
    (MODIFICATION: Made all input matrix sizes N x N)
 */
static void naiveMult(float * A, float * B, float * C, uint64_t M, uint64_t N, uint64_t L)
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
    Call this function before the matrix multiply to flush the cache so that none of A or B are in the
    cache before the multiply is performed.
    I couldn't find a fool proof way to do this. cacheflush isn't available to us. 
    But this seems to work.
*/
static void cacheFlush()
{
    // Going to sleep for a few seconds causes a context switch
    sleep(3);
}

/*
    printMatrix
    Outputs the header and a matrix. Matrix is of size width by width
*/
static void printMatrix(const char * header, float * matrix, int width)
{
    int i, j;
    printf("\n%s:\n", header);
    for (i = 0; i < width; i++)
    {
        for (j = 0; j < width; j++)
        {
            printf("%4.1f ", matrix[i * width + j]);
        }
        printf("\n");
    }
}

/* 
    Initialize an array of size floats to random values between 0 and 9.
*/
static void initialize(float * array, uint64_t size)
{
   for (uint64_t i = 0; i < size; i++) array[i] = rand() % 10;
} 

/*
    compare
    Compares the values in two matrices and outputs an
    error message and exits if the values do not match.
    Inputs
    result1, result2 - float matrices
    n - dimension of each matrix is n by n
    label - string to use in the output message if an error occurs
*/
static void compare(float * result1, float * result2, int n)
{
    int i, j;
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        { 
            float diff = abs(result1[i * n + j] - result2[i * n + j]);
            if (diff > 0) // 
            {
                printf("GPU transpose does not match CPU results.\n");
                printf("cpu result[%d, %d]: %f, gpu: result[%d, %d]: %f\n", 
                   i, j, result1[i * n + j], i, j, result2[i * n + j]);
                exit(EXIT_FAILURE);
            }
        }
    }
}