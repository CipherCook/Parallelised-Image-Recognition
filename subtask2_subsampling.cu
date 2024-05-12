#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <random>
#include <chrono>
#include <time.h>

using namespace std;

// Let's write the code for CUDA for Max Pooling and Average Pooling for square matrix
// Writing code for applying Max Pooling and Average Pooling on a float matrix using cuda

__global__ void MaxPooling_kernel(float *input, float *output, int input_size, int pool_size){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    // Now this thread will be responsible for doing all the computations for the output matrix (i,j) entry
    int op_size = input_size / pool_size;
    if(i < op_size && j < op_size){
        float max_val = -1000000.0f;
        for(int x = i * pool_size; x < (i + 1) * pool_size; x++){
            for(int y = j * pool_size; y < (j + 1) * pool_size; y++){
                max_val = max(max_val, input[x * input_size + y]);
            }
        }
        output[i * op_size + j] = max_val;
    }
}

__global__ void AveragePooling_kernel(float *input, float *output, int input_size, int pool_size){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    // Now this thread will be responsible for doing all the computations for the output matrix (i,j) entry
    int op_size = input_size / pool_size;
    if(i < op_size && j < op_size){
        float sum = 0;
        for(int x = i * pool_size; x < (i + 1) * pool_size; x++){
            for(int y = j * pool_size; y < (j + 1) * pool_size; y++){
                sum += input[x * input_size + y];
            }
        }
        output[i * op_size + j] = sum / (pool_size * pool_size);
    }
}

float *MaxPooling_gpu(float *input, int input_size, int pool_size){
    int output_size = input_size / pool_size;
    int input_bytes = input_size * input_size * sizeof(float);
    int output_bytes = output_size * output_size * sizeof(float);

    // Allocate memory on GPU
    float *d_input, *d_output;
    cudaMalloc((void **)&d_input, input_bytes);
    cudaMalloc((void **)&d_output, output_bytes);

    // Copy input from host to device
    cudaMemcpy(d_input, input, input_bytes, cudaMemcpyHostToDevice);

    // Define the block and grid dimensions
    dim3 blockSize(16, 16);
    dim3 gridSize((output_size + blockSize.x - 1) / blockSize.x, (output_size + blockSize.y - 1) / blockSize.y);

    // Launch MaxPooling kernel
    MaxPooling_kernel<<<gridSize, blockSize>>>(d_input, d_output, input_size, pool_size);

    // Allocate memory for result on host
    float *output = (float *)malloc(output_bytes);

    // Copy the output
    cudaMemcpy(output, d_output, output_bytes, cudaMemcpyDeviceToHost);

    // Free Device memory
    cudaFree(d_input);
    cudaFree(d_output);

    return output;
}

float *AveragePooling_gpu(float *input, int input_size, int pool_size){
    int output_size = input_size / pool_size;
    int input_bytes = input_size * input_size * sizeof(float);
    int output_bytes = output_size * output_size * sizeof(float);

    // Allocate memory on GPU
    float *d_input, *d_output;
    cudaMalloc((void **)&d_input, input_bytes);
    cudaMalloc((void **)&d_output, output_bytes);

    // Copy input from host to device
    cudaMemcpy(d_input, input, input_bytes, cudaMemcpyHostToDevice);

    // Define the block and grid dimensions
    dim3 blockSize(16, 16);
    dim3 gridSize((output_size + blockSize.x - 1) / blockSize.x, (output_size + blockSize.y - 1) / blockSize.y);

    // Launch AveragePooling kernel
    AveragePooling_kernel<<<gridSize, blockSize>>>(d_input, d_output, input_size, pool_size);

    // Allocate memory for result on host
    float *output = (float *)malloc(output_bytes);

    // Copy the output
    cudaMemcpy(output, d_output, output_bytes, cudaMemcpyDeviceToHost);

    // Free Device memory
    cudaFree(d_input);
    cudaFree(d_output);

    return output;
}

// Let's test this code against our CPU codes

float ** max_pooling(float ** input, int pool_size, int input_size){
    int output_size = input_size / pool_size;
    float ** output = new float*[output_size];
    for(int i = 0; i < output_size; i++){
        output[i] = new float[output_size];
    }
    for(int i = 0; i < output_size; i++){
        for(int j = 0; j < output_size; j++){
            output[i][j] = 0.0f;
        }
    }
    for(int i = 0; i < output_size; i++){
        for(int j = 0; j < output_size; j++){
            float max_val = -1000000.0f;
            for(int k = 0; k < pool_size; k++){
                for(int l = 0; l < pool_size; l++){
                    if(input[i*pool_size + k][j*pool_size + l] > max_val){
                        max_val = input[i*pool_size + k][j*pool_size + l];
                    }
                }
            }
            output[i][j] = max_val;
        }
    }
    return output;

}

float ** avg_pooling(float **input, int pool_size, int input_size){

    int output_size = input_size / pool_size;
    float ** output = new float*[output_size];
    for(int i = 0; i < output_size; i++){
        output[i] = new float[output_size];
    }
    for(int i = 0; i < output_size; i++){
        for(int j = 0; j < output_size; j++){
            output[i][j] = 0.0f;
        }
    }
    for(int i = 0; i < output_size; i++){
        for(int j = 0; j < output_size; j++){
            float sum = 0.0f;
            for(int k = 0; k < pool_size; k++){
                for(int l = 0; l < pool_size; l++){
                    sum += input[i*pool_size + k][j*pool_size + l];
                }
            }
            output[i][j] = sum / (pool_size * pool_size);
        }
    }
    return output;

}

int main(){
    // Let's test now 
    int input_size = 800;
    int pool_size = 4;
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    // Now let's create a random input
    float ** input = new float*[input_size];
    for(int i = 0; i < input_size; i++){
        input[i] = new float[input_size];
    }
    // We need a flattened one for the GPU
    float *input_flattened = new float[input_size * input_size];
    for(int i = 0; i < input_size; i++){
        for(int j = 0; j < input_size; j++){
            input[i][j] = dist(mt);
            input_flattened[i * input_size + j] = input[i][j];
        }
    }
    // Let's apply Max Pooling on CPU
    cout<<"Starting the Max Pooling"<<endl;
    auto start = chrono::high_resolution_clock::now();
    float **output_cpu_max = max_pooling(input, pool_size, input_size);
    auto end = chrono::high_resolution_clock::now();
    // Let's apply Max Pooling on GPU
    float *output_gpu_max = MaxPooling_gpu(input_flattened, input_size, pool_size);
    auto end_gpu = chrono::high_resolution_clock::now();

    cout<<"Starting the Average Pooling"<<endl;
    // Let's apply Average Pooling on CPU
    auto start_avg = chrono::high_resolution_clock::now();
    float **output_cpu_avg = avg_pooling(input, pool_size, input_size);
    auto end_avg = chrono::high_resolution_clock::now();
    // Let's apply Average Pooling on GPU
    float *output_gpu_avg = AveragePooling_gpu(input_flattened, input_size, pool_size);
    auto end_gpu_avg = chrono::high_resolution_clock::now();

    // Let's compare the results with the norms
    float norm_max_cpu = 0.0f;
    float norm_avg_cpu = 0.0f;
    for(int i = 0; i < input_size / pool_size; i++){
        for(int j = 0; j < input_size / pool_size; j++){
            norm_max_cpu += abs((output_cpu_max[i][j] - output_gpu_max[i * input_size / pool_size + j]))*abs((output_cpu_max[i][j] - output_gpu_max[i * input_size / pool_size + j]));
            norm_avg_cpu += abs((output_cpu_avg[i][j] - output_gpu_avg[i * input_size / pool_size + j]))*abs((output_cpu_avg[i][j] - output_gpu_avg[i * input_size / pool_size + j]));
        }
    }
    norm_max_cpu = sqrt(norm_max_cpu);
    norm_max_gpu = sqrt(norm_max_gpu);

    // Let's print the results

    cout<<"Time taken by CPU: "<<chrono::duration_cast<chrono::microseconds>(end - start).count()<<" microseconds"<<endl;
    cout<<"Time taken by GPU: "<<chrono::duration_cast<chrono::microseconds>(end_gpu - end).count()<<" microseconds"<<endl;
    cout<<"Norm of the difference between CPU and GPU output for Max Pooling: "<<norm_max_cpu<<endl;

    cout<<"Time taken by CPU: "<<chrono::duration_cast<chrono::microseconds>(end_avg - start_avg).count()<<" microseconds"<<endl;
    cout<<"Time taken by GPU: "<<chrono::duration_cast<chrono::microseconds>(end_gpu_avg - end_avg).count()<<" microseconds"<<endl;
    cout<<"Norm of the difference between CPU and GPU output for Average Pooling: "<<norm_avg_cpu<<endl;

}