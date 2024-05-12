#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <random>
#include <chrono>
#include <time.h>

using namespace std;

// Let's write the code for CUDA
// Writing code for applying non-linear activation function like ReLU and tanh on a float matrix using cuda
__global__ void ReLU_kernel(float *input, float *output, int rows, int cols){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if(i < rows && j < cols){
        output[i * cols + j] = max(0.0, input[i * cols + j]);
    }
}

__global__ void tanh_kernel(float *input, float *output, int rows, int cols){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if(i < rows && j < cols){
        output[i * cols + j] = tanh(input[i * cols + j]);
    }
}

float *ReLU_gpu(float *input, int rows, int cols){
    int output_size = rows * cols;
    int input_bytes = rows * cols * sizeof(float);
    int output_bytes = output_size * sizeof(float);

    // Allocate memory on GPU
    float *d_input, *d_output;
    cudaMalloc((void **)&d_input, input_bytes);
    cudaMalloc((void **)&d_output, output_bytes);

    // Copy input from host to device
    cudaMemcpy(d_input, input, input_bytes, cudaMemcpyHostToDevice);

    // Define the block and grid dimensions
    dim3 blockSize(16, 16);
    dim3 gridSize((rows + blockSize.x - 1) / blockSize.x, (cols + blockSize.y - 1) / blockSize.y);

    // Launch ReLU kernel
    ReLU_kernel<<<gridSize, blockSize>>>(d_input, d_output, rows, cols);

    // Allocate memory for result on host
    float *output = (float *)malloc(output_bytes);

    // Copy the output
    cudaMemcpy(output, d_output, output_bytes, cudaMemcpyDeviceToHost);

    // Free Device memory
    cudaFree(d_input);
    cudaFree(d_output);

    return output;
}


float *tanh_gpu(float *input, int rows, int cols){
    int output_size = rows * cols;
    int input_bytes = rows * cols * sizeof(float);
    int output_bytes = output_size * sizeof(float);

    // Allocate memory on GPU
    float *d_input, *d_output;
    cudaMalloc((void **)&d_input, input_bytes);
    cudaMalloc((void **)&d_output, output_bytes);

    // Copy input from host to device
    cudaMemcpy(d_input, input, input_bytes, cudaMemcpyHostToDevice);

    // Define the block and grid dimensions
    dim3 blockSize(16, 16);
    dim3 gridSize((rows + blockSize.x - 1) / blockSize.x, (cols + blockSize.y - 1) / blockSize.y);

    // Launch tanh kernel
    tanh_kernel<<<gridSize, blockSize>>>(d_input, d_output, rows, cols);

    // Allocate memory for result on host
    float *output = (float *)malloc(output_bytes);

    // Copy the output
    cudaMemcpy(output, d_output, output_bytes, cudaMemcpyDeviceToHost);

    // Free Device memory
    cudaFree(d_input);
    cudaFree(d_output);

    return output;
}

// CPU codes

void ReLU_mat(float ** input, int row, int cols){
    for(int i = 0; i < row; i++){
        for(int j = 0; j < cols; j++){
            if(input[i][j] < 0){
                input[i][j] = 0;
            }
        }
    }
}

void tanh_mat(float ** input, int row, int cols){
    for(int i = 0; i < row; i++){
        for(int j = 0; j < cols; j++){
            input[i][j] = tanh(input[i][j]);
        }
    }
}

// Testing and Comparison

int main(){
    // Define input matrix
    int rows = 100;
    int cols = 100;
    float **input = new float*[rows];
    for(int i = 0; i < rows; i++){
        input[i] = new float[cols];
    }
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    // We need a flattened one for the GPU
    float *input_flattened = new float[rows * cols];
    for(int i = 0; i < rows; i++){
        for(int j = 0; j < cols; j++){
            input[i][j] = dist(mt);
            input_flattened[i * cols + j] = input[i][j];
        }
    }

    // Apply ReLU on CPU
    auto start = chrono::high_resolution_clock::now();
    ReLU_mat(input, rows, cols);
    auto end = chrono::high_resolution_clock::now();
    // Apply ReLU on GPU
    float *output_gpu = ReLU_gpu(input_flattened, rows, cols);
    auto end_gpu = chrono::high_resolution_clock::now();
    cout<<"Time taken by CPU: "<<chrono::duration_cast<chrono::microseconds>(end - start).count()<<" microseconds"<<endl;
    cout<<"Time taken by GPU: "<<chrono::duration_cast<chrono::microseconds>(end_gpu - end).count()<<" microseconds"<<endl;
    // Difference between CPU and GPU
    float norm = 0.0f;
    for(int i = 0; i < rows; i++){
        for(int j = 0; j < cols; j++){
            norm += abs((input[i][j] - output_gpu[i * cols + j]))**2;
        }
    }
    norm = sqrt(norm);
    cout<<"Norm of the difference between CPU and GPU output: "<<norm<<endl;


    // Apply the tanh function on CPU
    // Need to reinitialize the input matrix
    for(int i = 0; i < rows; i++){
        for(int j = 0; j < cols; j++){
            input[i][j] = input_flattened[i * cols + j];
        }
    }
    start = chrono::high_resolution_clock::now();
    tanh_mat(input, rows, cols);
    end = chrono::high_resolution_clock::now();
    // Apply tanh on GPU
    output_gpu = tanh_gpu(input_flattened, rows, cols);
    end_gpu = chrono::high_resolution_clock::now();
    cout<<"Time taken by CPU: "<<chrono::duration_cast<chrono::microseconds>(end - start).count()<<" microseconds"<<endl;
    cout<<"Time taken by GPU: "<<chrono::duration_cast<chrono::microseconds>(end_gpu - end).count()<<" microseconds"<<endl;
    // Difference between CPU and GPU
    norm = 0.0f;
    for(int i = 0; i < rows; i++){
        for(int j = 0; j < cols; j++){
            norm += abs((input[i][j] - output_gpu[i * cols + j]))**2;
        }
    }
    norm = sqrt(norm);
    cout<<"Norm of the difference between CPU and GPU output: "<<norm<<endl;
    // Free the memory
    for(int i = 0; i < rows; i++){
        delete[] input[i];
    }
    delete[] input;
    delete[] input_flattened;
    delete[] output_gpu;
    return 0;
}