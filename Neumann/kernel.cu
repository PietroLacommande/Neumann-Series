
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <cstdlib> // For rand()
#include <ctime>   // For seeding rand()
#include <fstream>
#include <sstream>
#include <vector>

#define tileLength 16

void printMatrix(const float* matrix, int rows, int columns, const char* name) {
    printf("Matrix %s:\n", name);
    for (int i = 0; i < rows; ++i) {
        printf("{ "); // Start of row delimiter
        for (int j = 0; j < columns; ++j) {
            printf("%.8f", matrix[i * columns + j]); // Print with 2 decimal place
            if (j < columns - 1) {
                printf(" "); // Add space between elements
            }
        }
        printf(" }"); // End of row delimiter
        printf("\n"); // Newline after each row
    }
    printf("\n");
}

// CUDA kernel for matrix multiplication with shared memory
__global__ void multiplyKernel(float* c, const float* a, const float* b, int length) {
    int threadY = threadIdx.y;
    int threadX = threadIdx.x;

    int row = (tileLength * blockIdx.y) + threadY;
    int column = (tileLength * blockIdx.x) + threadX;

    __shared__ float tileA[tileLength][tileLength];
    __shared__ float tileB[tileLength][tileLength];

    float sum = 0;

    int numTiles = (length + tileLength - 1) / tileLength;

    for (int t = 0; t < numTiles; t++) {
        // Data loading
        if (row < length && (t * tileLength + threadX) < length) {
            tileA[threadY][threadX] = a[row * length + (t * tileLength + threadX)];
        }
        else {
            tileA[threadY][threadX] = 0.0f;
        }

        if (column < length && (t * tileLength + threadY) < length) {
            tileB[threadY][threadX] = b[(t * tileLength + threadY) * length + column];
        }
        else {
            tileB[threadY][threadX] = 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < tileLength; k++) {
            sum += tileA[threadY][k] * tileB[k][threadX];
        }

        __syncthreads();
    }

    if (row < length && column < length) {
        c[(row * length) + column] = sum;
    }
}


// CUDA kernel to extract diagonal and off-diagonal matrices
__global__ void extractDiagonalAndOffDiagonal(float* A, float* D, float* E, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        int idx = row * N + col;
        if (row == col) {
            // Diagonal element
            D[idx] = A[idx];
            E[idx] = 0.0f;
        }
        else {
            // Off-diagonal element
            D[idx] = 0.0f;
            E[idx] = A[idx];
        }
    }
}

// CUDA kernel to invert the diagonal matrix
__global__ void invertDiagonal(float* D, float* D_inv, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float diag = D[idx * N + idx];
        D_inv[idx * N + idx] = (diag != 0.0f) ? 1.0f / diag : 0.0f; // Handle division by zero
    }
}

// CUDA kernel to combine results into A^{-1}
__global__ void combineResults(float* D_inv, float* DinvEDinv, float* DinvE2Dinv, float* A_inv, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        A_inv[row * N + col] = D_inv[row * N + col] - DinvEDinv[row * N + col] + DinvE2Dinv[row * N + col];
    }
}

// Host function for matrix inversion using Neumann series
void matrixInversionNeumann(float* A, float* A_inv, int N) {
    size_t size = N * N * sizeof(float);
    float* d_A, * d_D, * d_E, * d_D_inv, * d_DinvE, * d_DinvEDinv, * d_DinvE2,* d_DinvE2Dinv, * d_A_inv;

    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_D, size);
    cudaMalloc((void**)&d_E, size);
    cudaMalloc((void**)&d_D_inv, size);
    cudaMalloc((void**)&d_DinvE, size);
    cudaMalloc((void**)&d_DinvEDinv, size);
    cudaMalloc((void**)&d_DinvE2, size);
    cudaMalloc((void**)&d_DinvE2Dinv, size);
    cudaMalloc((void**)&d_A_inv, size);

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);

    dim3 blockSize(tileLength, tileLength);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y);

    // Step 1: Extract D and E
    extractDiagonalAndOffDiagonal << <gridSize, blockSize >> > (d_A, d_D, d_E, N);
    cudaDeviceSynchronize();

     // Step 2: Compute D^{-1}
    invertDiagonal << < gridSize, blockSize >> > (d_D, d_D_inv, N);
    cudaDeviceSynchronize();

    // Step 3: Compute D^{-1}E
    multiplyKernel << <gridSize, blockSize >> > (d_DinvE, d_D_inv, d_E, N);
    cudaDeviceSynchronize();

    // Step 4: Compute D^{-1}ED^{-1}
    multiplyKernel << <gridSize, blockSize >> > (d_DinvEDinv, d_DinvE, d_D_inv, N);
    

    // Step 5: Compute (D^{-1}E)^2D^{-1}
    multiplyKernel << <gridSize, blockSize >> > (d_DinvE2, d_DinvE, d_DinvE, N);
    cudaDeviceSynchronize();
    multiplyKernel << <gridSize, blockSize >> > (d_DinvE2Dinv, d_DinvE2, d_D_inv, N);
    cudaDeviceSynchronize();

    // Step 6: Combine results to compute A^{-1}
    combineResults << <gridSize, blockSize >> > (d_D_inv, d_DinvEDinv, d_DinvE2Dinv, d_A_inv, N);
    cudaDeviceSynchronize();

    cudaMemcpy(A_inv, d_A_inv, size, cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_D);
    cudaFree(d_E);
    cudaFree(d_D_inv);
    cudaFree(d_DinvE);
    cudaFree(d_DinvE2);
    cudaFree(d_DinvEDinv);
    cudaFree(d_DinvE2Dinv);
    cudaFree(d_A_inv);
}

// Function to read a matrix from a .txt file
void readMatrixFromFile(const char* filename, float* matrix, int rows, int cols) {
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        fprintf(stderr, "Error: Could not open file %s\n", filename);
        exit(EXIT_FAILURE);
    }

    std::string line;
    int row = 0;

    while (std::getline(infile, line) && row < rows) {
        std::istringstream iss(line);
        std::string value;
        int col = 0;

        while (std::getline(iss, value, ',') && col < cols) {
            matrix[row * cols + col] = std::stof(value); // Convert string to float
            col++;
        }
        row++;
    }

    infile.close();
}

int main() {
    const int N = 64; // Matrix size
    float A[N * N];   // Host matrix input
    float A_inv[N * N]; // Host matrix result

    // Step 1: Read matrix from file
    const char* filename = "C:/Users/lacommap/Desktop/Neumann-Series/MatrixA.txt";
    readMatrixFromFile(filename, A, N, N);

    printf("Original Matrix A:\n");
    printMatrix(A, N, N, "A");

    // Step 2: Perform Neumann series inversion
    matrixInversionNeumann(A, A_inv, N);

    printf("Approximated Inverse Matrix A^-1:\n");
    printMatrix(A_inv, N, N, "A^-1");

    return 0;
}


