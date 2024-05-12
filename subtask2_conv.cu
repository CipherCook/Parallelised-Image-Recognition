#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <random>
#include <chrono>
#include <time.h>

using namespace std;

// Let's write the code for CUDA

// Convolution function

__global__ void convolutionKernel(float *input, float *kernel, float *output, int input_size, int kernel_size, int padding) {
    int output_size = input_size - kernel_size + 1 + 2 * padding;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < output_size && j < output_size) {
        for (int k = 0; k < kernel_size; k++) {
            for (int l = 0; l < kernel_size; l++) {
                int input_row = i + k - padding;
                int input_col = j + l - padding;
                if(i+k < padding || i+k >= input_size + padding || j+l < padding || j+l >= input_size + padding){
                  continue;
                }
                output[i * output_size + j] += input[input_row * input_size + input_col] * kernel[k * kernel_size + l];
            }
        }
    }
}

float* convolutionGPU(float *input, float *kernel, int input_size, int kernel_size, int padding) {
    int output_size = input_size - kernel_size + 1 + 2 * padding;
    int input_bytes = input_size * input_size * sizeof(float);
    int kernel_bytes = kernel_size * kernel_size * sizeof(float);
    int output_bytes = output_size * output_size * sizeof(float);

    // Allocate memory on GPU
    float *d_input, *d_kernel, *d_output;
    cudaMalloc((void **)&d_input, input_bytes);
    cudaMalloc((void **)&d_kernel, kernel_bytes);
    cudaMalloc((void **)&d_output, output_bytes);

    // Copy input and kernel from host to device
    cudaMemcpy(d_input, input, input_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, kernel_bytes, cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    dim3 blockSize(16, 16);
    dim3 gridSize((output_size + blockSize.x - 1) / blockSize.x, (output_size + blockSize.y - 1) / blockSize.y);

    // Launch convolution kernel
    convolutionKernel<<<gridSize, blockSize>>>(d_input, d_kernel, d_output, input_size, kernel_size, padding);

    // Allocate memory for result on host
    float *output = (float *)malloc(output_bytes);

    // Copy result from device to host
    cudaMemcpy(output, d_output, output_bytes, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);

    return output;
}


float ** convolutionCPU(float ** input, float ** kernel, int input_size, int kernel_size, int padding){
    int output_size = input_size - kernel_size + 1 + 2*padding;
    float ** output = new float*[output_size];
    for(int i = 0; i < output_size; i++){
        output[i] = new float[output_size];
    }
    for(int i = 0; i < output_size; i++){
        for(int j = 0; j < output_size; j++){
            output[i][j] = 0.0f;
        }
    }
    // Need to delete the output once work is done
    // Let's convolve now assuming padding = 0

        for(int i = 0; i < output_size; i++){
            for(int j = 0; j< output_size; j++){

                for(int k = 0; k < kernel_size; k++){
                    for(int l = 0; l< kernel_size; l++){
                        if(i+k < padding || i+k >= input_size + padding || j+l < padding || j+l >= input_size + padding){
                            continue;
                        }
                        else{
                            output[i][j] += input[i+k-padding][j+l-padding] * kernel[k][l];
                        }
                    }
                }
            }
        }

    return output;
}




// Let's write a main function to test the convolution function
int main() {

    // Define input and kernel
    int input_size = 800;
    int kernel_size = 30;
    int padding = 1;
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    // Now let's create a random input and kernel
    float ** input = new float*[input_size];
    for(int i = 0; i < input_size; i++){
        input[i] = new float[input_size];
    }
    float ** kernel = new float*[kernel_size];
    for(int i = 0; i < kernel_size; i++){
        kernel[i] = new float[kernel_size];
    }
    // We need to flatten this for the GPU
    float * input_flattened = new float[input_size * input_size];
    float * kernel_flattened = new float[kernel_size * kernel_size];
    for(int i = 0; i < input_size; i++){
        for(int j = 0; j < input_size; j++){
            input[i][j] = dist(mt);
            input_flattened[i*input_size + j] = input[i][j];
        }
    }
    for(int i = 0; i < kernel_size; i++){
        for(int j = 0; j < kernel_size; j++){
            kernel[i][j] = dist(mt);
            kernel_flattened[i*kernel_size + j] = kernel[i][j];
        }
    }
    // Let's convolve now
    // Let's measure the time difference in the two operations
    cout<<"Starting_CPU"<<endl;
    auto start_cpu = chrono::high_resolution_clock::now();
    float ** output_cpu = convolutionCPU(input, kernel, input_size, kernel_size, padding);
    auto end_cpu = chrono::high_resolution_clock::now();
    float * output_gpu = convolutionGPU(input_flattened, kernel_flattened, input_size, kernel_size, padding);
    auto end_gpu = chrono::high_resolution_clock::now();
    // Let's print the compute the Normal of the difference between the two outputs
    float norm = 0.0f;
    int op_sz = input_size - kernel_size + 1 + 2*padding;
    for(int i = 0; i < op_sz; i++){
        for(int j = 0; j < op_sz; j++){
            norm += abs(output_cpu[i][j] - output_gpu[i*op_sz + j]);
        }
    }
    cout << "Norm of the difference between CPU and GPU output: " << norm << endl;
    cout << "Time taken by CPU: " << chrono::duration_cast<chrono::microseconds>(end_cpu - start_cpu).count() << " microseconds" << endl;
    cout << "Time taken by GPU: " << chrono::duration_cast<chrono::microseconds>(end_gpu - end_cpu).count() << " microseconds" << endl;
    // Print the CPU first line
    cout<<"CPU First Line"<<endl;
    for(int i = 0; i < op_sz; i++){
        cout<<output_cpu[0][i]<<" ";
    }
    cout<<endl;
    // Print the GPU first line
    cout<<"GPU First Line"<<endl;
    for(int i = 0; i < op_sz; i++){
        cout<<output_gpu[i]<<" ";
    }
    cout<<endl;
    // Free the memory
    for(int i = 0; i < input_size; i++){
        delete[] input[i];
    }
    delete[] input;
    for(int i = 0; i < kernel_size; i++){
        delete[] kernel[i];
    }
    delete[] kernel;
    delete[] input_flattened;
    delete[] kernel_flattened;
    for(int i = 0; i < input_size - kernel_size + 1 + 2*padding; i++){
        delete[] output_cpu[i];
    }
    delete[] output_cpu;
    delete[] output_gpu;

    return 0;
}