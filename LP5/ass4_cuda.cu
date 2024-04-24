// Program1

#include <cuda_runtime.h>
#include <iostream>

// CUDA kernel to perform matrix multiplication
__global__ void matmul(int *A, int *B, int *C, int N)
{
    int Row = blockIdx.y * blockDim.y + threadIdx.y; // Calculate the row index
    int Col = blockIdx.x * blockDim.x + threadIdx.x; // Calculate the column index

    // Ensure the thread is within the matrix dimensions
    if (Row < N && Col < N)
    {
        int Pvalue = 0; // Initialize the element of the resulting matrix

        // Perform dot product of row of A and column of B to compute element of C
        for (int k = 0; k < N; k++)
        {
            Pvalue += A[Row * N + k] * B[k * N + Col];
        }

        C[Row * N + Col] = Pvalue; // Store the computed element in the result matrix
    }
}

int main()
{
    int N = 512;                    // Size of the matrices
    int size = N * N * sizeof(int); // Size of each matrix in bytes
    int *A, *B, *C;                 // Host matrices
    int *dev_A, *dev_B, *dev_C;     // Device matrices
    cudaMallocHost(&A, size);       // Allocate pinned memory for matrix A
    cudaMallocHost(&B, size);       // Allocate pinned memory for matrix B
    cudaMallocHost(&C, size);       // Allocate pinned memory for matrix C
    cudaMalloc(&dev_A, size);       // Allocate memory for matrix A on the device
    cudaMalloc(&dev_B, size);       // Allocate memory for matrix B on the device
    cudaMalloc(&dev_C, size);       // Allocate memory for matrix C on the device

    // Initialize matrices A and B
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            A[i * N + j] = i * N + j; // Initialize matrix A with row-major order
            B[i * N + j] = j * N + i; // Initialize matrix B with column-major order
        }
    }

    // Copy matrices A and B from host to device
    cudaMemcpy(dev_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_B, B, size, cudaMemcpyHostToDevice);

    // Define thread block dimensions and grid dimensions
    dim3 dimBlock(16, 16);                        // 16x16 thread block
    dim3 dimGrid(N / dimBlock.x, N / dimBlock.y); // N/16 x N/16 grid

    // Launch the kernel for matrix multiplication
    matmul<<<dimGrid, dimBlock>>>(dev_A, dev_B, dev_C, N);

    // Copy the result matrix C from device to host
    cudaMemcpy(C, dev_C, size, cudaMemcpyDeviceToHost);

    // Print the first 10x10 elements of the result matrix C
    for (int i = 0; i < 10; i++)
    {
        for (int j = 0; j < 10; j++)
        {
            std::cout << C[i * N + j] << " ";
        }
        std::cout << std::endl;
    }

    // Free memory on the device
    cudaFree(dev_A);
    cudaFree(dev_B);
    cudaFree(dev_C);
    // Free pinned memory on the host
    cudaFreeHost(A);
    cudaFreeHost(B);
    cudaFreeHost(C);

    return 0;
}

// Program 2

#include <iostream>
#include <cuda_runtime.h>

using namespace std;

// CUDA kernel to add two vectors
__global__ void addVectors(int *A, int *B, int *C, int n)
{
    // Calculate the global index of the thread
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        // Add corresponding elements from A and B and store the result in C
        C[i] = A[i] + B[i];
    }
}

int main()
{
    int n = 1000000;            // Size of the vectors
    int *A, *B, *C;             // Host vectors
    int size = n * sizeof(int); // Size of each vector in bytes

    // Allocate pinned memory on the host for vectors A, B, and C
    cudaMallocHost(&A, size);
    cudaMallocHost(&B, size);
    cudaMallocHost(&C, size);

    // Initialize vectors A and B
    for (int i = 0; i < n; i++)
    {
        A[i] = i;
        B[i] = i * 2;
    }

    // Allocate memory for vectors A, B, and C on the device
    int *dev_A, *dev_B, *dev_C;
    cudaMalloc(&dev_A, size);
    cudaMalloc(&dev_B, size);
    cudaMalloc(&dev_C, size);

    // Copy vectors A and B from host to device
    cudaMemcpy(dev_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_B, B, size, cudaMemcpyHostToDevice);

    // Launch the kernel to add vectors A and B
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    addVectors<<<numBlocks, blockSize>>>(dev_A, dev_B, dev_C, n);

    // Copy vector C from device to host
    cudaMemcpy(C, dev_C, size, cudaMemcpyDeviceToHost);

    // Print the first 10 elements of the result vector C
    for (int i = 0; i < 10; i++)
    {
        cout << C[i] << " ";
    }
    cout << endl;

    // Free memory on the device
    cudaFree(dev_A);
    cudaFree(dev_B);
    cudaFree(dev_C);
    // Free pinned memory on the host
    cudaFreeHost(A);
    cudaFreeHost(B);
    cudaFreeHost(C);

    return 0;
}
