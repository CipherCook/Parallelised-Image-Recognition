#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <random>
#include <chrono>
#include <time.h>

using namespace std;

// Let's write the code for CUDA for softmax and sigmoid functions
// Writing code for applying softmax and sigmoid on a float matrix using cuda
// We will do the sigmoid first 

__global__ void sigmoid(float *input, float *output, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        output[index] = 1 / (1 + exp(-input[index]));
    }
}

// Now about the softmax function
// We pass in the exponent of total sum also as a parameter
__global__ void softmax_1(float *input, float *output, float *exp_total_sum, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        
        float alpha = exp(input[index]);
        atomicAdd(exp_total_sum, alpha);
        output[index] = alpha;

    }
}

__global__ void softmax_2(float *output, float *exp_total_sum, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        output[index] /= *exp_total_sum;
    }
}

// Now let's write the main function

float* sigmoid_gpu(float *input, int n) {
    int input_bytes = n * sizeof(float);
    int output_bytes = n * sizeof(float);

    // Allocate memory on GPU
    float *d_input, *d_output;
    cudaMalloc((void **)&d_input, input_bytes);
    cudaMalloc((void **)&d_output, output_bytes);

    // Copy input from host to device
    cudaMemcpy(d_input, input, input_bytes, cudaMemcpyHostToDevice);

    // Define the block and grid dimensions
    dim3 blockSize(256);
    dim3 gridSize((n + blockSize.x - 1) / blockSize.x);

    // Launch sigmoid kernel
    sigmoid<<<gridSize, blockSize>>>(d_input, d_output, n);

    // Allocate memory for result on host
    float *output = (float *)malloc(output_bytes);

    // Copy the output
    cudaMemcpy(output, d_output, output_bytes, cudaMemcpyDeviceToHost);

    // Free Device memory
    cudaFree(d_input);
    cudaFree(d_output);

    return output;
}


float* softmax_gpu(float *input, int n) {
    int input_bytes = n * sizeof(float);
    int output_bytes = n * sizeof(float);
    int exp_total_sum_bytes = sizeof(float);

    // Allocate memory on GPU
    float *d_input, *d_output, *d_exp_total_sum;
    cudaMalloc((void **)&d_input, input_bytes);
    cudaMalloc((void **)&d_output, output_bytes);
    cudaMalloc((void **)&d_exp_total_sum, exp_total_sum_bytes);

    // Copy input from host to device
    cudaMemcpy(d_input, input, input_bytes, cudaMemcpyHostToDevice);

    // Define the block and grid dimensions
    dim3 blockSize(256);
    dim3 gridSize((n + blockSize.x - 1) / blockSize.x);

    // Launch softmax kernel
    softmax_1<<<gridSize, blockSize>>>(d_input, d_output, d_exp_total_sum, n);
    softmax_2<<<gridSize, blockSize>>>(d_output, d_exp_total_sum, n);

    // Allocate memory for result on host
    float *output = (float *)malloc(output_bytes);

    // Copy the output
    cudaMemcpy(output, d_output, output_bytes, cudaMemcpyDeviceToHost);

    // Free Device memory
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_exp_total_sum);

    return output;
}


float * sigmoid_cpu(float * input, int size){
    float * output = new float[size];
    for(int i = 0; i < size; i++){
        output[i] = 1.0f / (1.0f + exp(-input[i]));
    }
    return output;
}

float * softmax_cpu(float * input, int size){
    float * output = new float[size];
    float sum = 0.0f;
    for(int i = 0; i < size; i++){
        output[i] = exp(input[i]);
        sum += output[i];
    }
    for(int i = 0; i < size; i++){
        output[i] /= sum;
    }
    return output;
}


__global__ void dummyKernel(int *result) {
    *result = threadIdx.x + blockIdx.x * blockDim.x;
}


void random_gpu() {
    // To warm up the GPU

    const int N = 1; // Number of elements

    // Allocate memory on the host
    int *h_result = new int[N];

    // Allocate memory on the device
    int *d_result;
    cudaMalloc((void **)&d_result, N * sizeof(int));

    // Invoke kernel
    dummyKernel<<<1, 1>>>(d_result);

    // Copy result back to host
    cudaMemcpy(h_result, d_result, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Output the result (optional)
    std::cout << "Result: " << *h_result << std::endl;

    // Free memory on the device
    cudaFree(d_result);

    // Free memory on the host
    delete[] h_result;
}

int main(){
    // Now let's compare the performance of the CPU and GPU implementations
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    int size = 10000;
    float * input = new float[size];
    for(int i = 0; i < size; i++){
        input[i] = dist(mt);
    }
    random_gpu();
    // Now let's test the sigmoid function
    auto start = chrono::high_resolution_clock::now();
    float * output_cpu = sigmoid_cpu(input, size);
    auto end = chrono::high_resolution_clock::now();
    float * output_gpu = sigmoid_gpu(input, size);
    auto end_gpu = chrono::high_resolution_clock::now();
    cout << "Time taken by CPU: " << chrono::duration_cast<chrono::microseconds>(end - start).count() << endl;
    cout << "Time taken by GPU: " << chrono::duration_cast<chrono::microseconds>(end_gpu - end).count() << endl;
    // Now let's test the softmax function
    auto start_s = chrono::high_resolution_clock::now();
    float * output_cpu_s = softmax_cpu(input, size);
    auto end_s = chrono::high_resolution_clock::now();
    float * output_gpu_s = softmax_gpu(input, size);
    auto end_gpu_s = chrono::high_resolution_clock::now();
    cout << "Time taken by CPU: " << chrono::duration_cast<chrono::microseconds>(end_s - start_s).count() << endl;
    cout << "Time taken by GPU: " << chrono::duration_cast<chrono::microseconds>(end_gpu_s - end_s).count() << endl;
    // Now let's compute the norm of the difference between the two outputs
    float norm = 0.0f;
    for(int i = 0; i < size; i++){
        norm += (abs(output_cpu[i] - output_gpu[i]))*(abs(output_cpu[i] - output_gpu[i]));
    }
    cout << "Norm of the difference between CPU and GPU sigmoid output: " << sqrt(norm) << endl;
    norm = 0.0f;
    for(int i = 0; i < size; i++){
        norm += (abs(output_cpu_s[i] - output_gpu_s[i]))*(abs(output_cpu_s[i] - output_gpu_s[i]));
    }
    cout << "Norm of the difference between CPU and GPU softmax output: " << sqrt(norm) << endl;

}




