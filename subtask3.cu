//OLD
#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <random>
#include <chrono>
#include <time.h>


using namespace std;
float *d_conv1_weights, *d_conv1_biases, *d_conv2_weights, *d_conv2_biases, *d_fc1_weights, *d_fc1_biases, *d_fc2_weights, *d_fc2_biases;


__global__ void softmax_1(float *input, float *output, float *exp_total_sum, int n) {
    int index = threadIdx.x;
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

__global__ void ReLU_kernel(float *input, float *output, int rows, int cols){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if(i < rows && j < cols){
        output[i * cols + j] = max(0.0, input[i * cols + j]);
    }
}

__global__ void conv1_kernel(float *input, float *output, float *d_conv1_weights, float *d_conv1_biases){
    int i = threadIdx.x;
    int j = threadIdx.y;
    int k = blockIdx.x;
    float op = 0;
    for(int x = 0; x < 5; x++){
        for(int y = 0; y < 5; y++){
            op += input[(i + x) * 28 + (j + y)] * d_conv1_weights[k * 5 * 5 + x * 5 + y];
        }
    }
    output[k * 24 * 24 + i * 24 + j] = op + d_conv1_biases[k];
}

__global__ void maxpool_1_kernel(float *input, float *output){
    int i = threadIdx.x;
    int j = threadIdx.y;
    int k = blockIdx.x;
    float max_val = -1000000.0f;
    for(int x = 0; x < 2; x++){
        for(int y = 0; y < 2; y++){
            max_val = max(max_val, input[k * 24 * 24 + (i * 2 + x) * 24 + (j * 2 + y)]);
        }
    }
    output[k * 12 * 12 + i * 12 + j] = max_val;
}

__global__ void conv2_kernel(float *input, float *output, float *d_conv2_weights, float *d_conv2_biases){
    int i = threadIdx.x;
    int j = threadIdx.y;
    int k1 = blockIdx.x;
    int k2 = blockIdx.y;
    // input size = 20 * (12 * 12) 
    // output size = 50 * (8 * 8)
    float op = 0;
    for(int x = 0; x < 5; x++){
        for(int y = 0; y < 5; y++){
            // k1 denotes the filter number of the first convolutional layer
            // k2 denotes the filter number of the second convolutional layer
            op += input[k1 * 12 * 12 + (i + x) * 12 + (j + y)] * d_conv2_weights[k2 * 20 * 5 * 5 + k1 * 5 * 5 + x * 5 + y];
        }
    }
    atomicAdd(&output[k2 * 8 * 8 + i * 8 + j],op);
    if(k1 == 19){
        atomicAdd(&output[k2 * 8 * 8 + i * 8 + j],d_conv2_biases[k2]);
    }
}

__global__ void maxpool_2_kernel(float *input, float *output){
    int i = threadIdx.x;
    int j = threadIdx.y;
    int k = blockIdx.x;
    float max_val = -1000000.0f;
    for(int x = 0; x < 2; x++){
        for(int y = 0; y < 2; y++){
            max_val = max(max_val, input[k * 8 * 8 + (i * 2 + x) * 8 + (j * 2 + y)]);
        }
    }
    output[k * 4 * 4 + i * 4 + j] = max_val;
}

__global__ void fc_1_kernel(float *input, float *output, float *d_fc1_weights, float *d_fc1_biases){
    int i = threadIdx.x;
    int j = threadIdx.y;
    int k = threadIdx.z;
    int f = blockIdx.x;
    float op = input[k*16 + i * 4 + j] * d_fc1_weights[f * 800 + k*16 + i * 4 + j];
    atomicAdd(&output[f], op);
    if(i == 0 && j==0 && k==0){
        atomicAdd(&output[f], d_fc1_biases[f]);
    }
}

__global__ void RelU_fc1(float *input){
    int i = threadIdx.x;
    if(input[i] < 0){
        input[i] = 0;
    }
}

__global__ void fc_2_kernel(float *input, float *output, float *d_fc2_weights, float *d_fc2_biases){
    int i = threadIdx.x; // 500 threads
    int j = blockIdx.x; // 10 blocks
    float op = input[i] * d_fc2_weights[j * 500 + i];
    atomicAdd(&output[j], op);
    if(i == 0){
        atomicAdd(&output[j], d_fc2_biases[j]);
    }
}

__global__ void Matrix_wt_sum(float *matrix, int size, float* output){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i == 0){
        for(int j = 0; j < size; j++){
            output[0] += matrix[j];
        }
    }
}

__global__ void print_matrix(float *matrix, int rows, int cols){
    for(int i = 0; i < rows; i++){
        for(int j = 0; j < cols; j++){
            printf("%f ", matrix[i * cols + j]);
        }
        printf("\n");
    }
}

float* neural_net_run(float *input, float* output){
    // input size = 28*28
    // output size = 10
    // Let's first do the convolutional layer
    // We will first do the convolutional layer 1
    // Make a grid of size 20
    // And every block of size 24*24
    dim3 blockSize(24, 24);
    dim3 gridSize(20);
    float *d_input, *d_output;
    cudaMalloc((void **)&d_input, 28 * 28 * sizeof(float));
    cudaMalloc((void **)&d_output, 20 * 24 * 24 * sizeof(float));
    cudaMemcpy(d_input, input, 28 * 28 * sizeof(float), cudaMemcpyHostToDevice);
    conv1_kernel<<<gridSize, blockSize>>>(d_input, d_output, d_conv1_weights, d_conv1_biases);
    // Print the matrix
    // print_matrix<<<1, 1>>>(d_output, 20, 24 * 24);
    // Now we have the output of the first convolutional layer
    // Now we will do the maxpooling layer
    // The input size is 20 * 24 * 24
    // The output size is 20 * 12 * 12
    dim3 blockSize1(12, 12);
    dim3 gridSize1(20);
    float *d_output1;
    cudaMalloc((void **)&d_output1, 20 * 12 * 12 * sizeof(float));
    maxpool_1_kernel<<<gridSize1, blockSize1>>>(d_output, d_output1);
    //print_matrix<<<1,1>>>(d_output1,20,12*12);
    // Now we have the output of the maxpooling layer
    // Now we will do the second convolutional layer
    // The input size is 20 * 12 * 12
    // The output size is 50 * 8 * 8
    dim3 blockSize2(8, 8);
    dim3 gridSize2(20, 50);
    float *d_output2;
    cudaMalloc((void **)&d_output2, 50 * 8 * 8 * sizeof(float));
    // Need to set the output to 0
    cudaMemset(d_output2, 0, 50 * 8 * 8 * sizeof(float));
    conv2_kernel<<<gridSize2, blockSize2>>>(d_output1, d_output2, d_conv2_weights, d_conv2_biases);
    //print_matrix<<<1,1>>>(d_output2,50,8*8);
    // Now we have the output of the second convolutional layer
    // Now we will do the maxpooling layer
    // The input size is 50 * 8 * 8
    // The output size is 50 * 4 * 4
    dim3 blockSize3(4, 4);
    dim3 gridSize3(50);
    float *d_output3;
    cudaMalloc((void **)&d_output3, 50 * 4 * 4 * sizeof(float));
    maxpool_2_kernel<<<gridSize3, blockSize3>>>(d_output2, d_output3);
    //print_matrix<<<1,1>>>(d_output3,50,4*4);
    // Now we have the output of the maxpooling layer
    // Now we will do the fully connected layer 1
    // The input size is 50 * 4 * 4
    // The output size is 500
    dim3 blockSize4(4, 4, 50);
    dim3 gridSize4(500);
    float *d_output4;
    cudaMalloc((void **)&d_output4, 500 * sizeof(float));
    fc_1_kernel<<<gridSize4, blockSize4>>>(d_output3, d_output4, d_fc1_weights, d_fc1_biases);
    // print_matrix<<<1,1>>>(d_output4,20,25);
    // Now we have the output of the fully connected layer 1
    // Now we will do the ReLU layer
    // The input size is 500
    // The output size is 500
    RelU_fc1<<<1, 500>>>(d_output4);
    // print_matrix<<<1,1>>>(d_output4,20,25);
    // Now we have the output of the ReLU layer
    // Now we will do the fully connected layer 2
    // The input size is 500
    // The output size is 10
    dim3 blockSize5(500);
    dim3 gridSize5(10);
    float *d_output5;
    cudaMalloc((void **)&d_output5, 10 * sizeof(float));
    fc_2_kernel<<<gridSize5, blockSize5>>>(d_output4, d_output5, d_fc2_weights, d_fc2_biases);
    // print_matrix<<<1,1>>>(d_output5,1,10);
    // Now we have the output of the fully connected layer 2
    // We will perform the softmax operation here only
    // The input size is 10
    // The output size is 10
    float *d_output6;
    cudaMalloc((void **)&d_output6, 10 * sizeof(float));
    float *d_exp_total_sum;
    cudaMalloc((void **)&d_exp_total_sum, sizeof(float));
    softmax_1<<<1, 10>>>(d_output5, d_output6, d_exp_total_sum, 10);
    softmax_2<<<1, 10>>>(d_output6, d_exp_total_sum, 10);
    // Now we have the output of the softmax layer
    // Copy the output to the host
    cudaMemcpy(output, d_output6, 10 * sizeof(float), cudaMemcpyDeviceToHost);
    // Free the device memory
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_output1);
    cudaFree(d_output2);
    cudaFree(d_output3);
    cudaFree(d_output4);
    cudaFree(d_output5);
    cudaFree(d_output6);
    cudaFree(d_exp_total_sum);
    return output;

}

void neural_net_set( float * conv1_weights, float * conv1_biases, float * conv2_weights, float * conv2_biases, float * fc1_weights, float * fc1_biases, float * fc2_weights, float * fc2_biases){
    // Need to make a neural network with 2 convolutional layers and 2 fully connected layers 
    // The first convolutional layer has an input dimension of 28*28 and uses 20 filters of kernels of size 5*5
    // Then we have a maxpooling layer of size 2*2
    // The second convolutional layer has an input dimension of 12*12 and uses 50 filters of kernels of size 20*5*5
    // Then we have a maxpooling layer of size 2*2
    // Then we have a fully connected layer with 500 neurons
    // Here we have a ReLU layer
    // Then we have a fully connected layer with 10 neurons
    // Here we have a softmax layer
    int conv1_weight_size = 20 * 5 * 5 * sizeof(float);
    int conv1_bias_size = 20 * sizeof(float);
    int conv2_weight_size = 50 * 20 * 5 * 5 * sizeof(float);
    int conv2_bias_size = 50 * sizeof(float);
    int fc1_weight_size = 500 * 800 * sizeof(float);
    int fc1_bias_size = 500 * sizeof(float);
    int fc2_weight_size = 10 * 500 * sizeof(float);
    int fc2_bias_size = 10 * sizeof(float);

    cudaMalloc((void **)&d_conv1_weights, conv1_weight_size);
    cudaMalloc((void **)&d_conv1_biases, conv1_bias_size);
    cudaMalloc((void **)&d_conv2_weights, conv2_weight_size);
    cudaMalloc((void **)&d_conv2_biases, conv2_bias_size);
    cudaMalloc((void **)&d_fc1_weights, fc1_weight_size);
    cudaMalloc((void **)&d_fc1_biases, fc1_bias_size);
    cudaMalloc((void **)&d_fc2_weights, fc2_weight_size);
    cudaMalloc((void **)&d_fc2_biases, fc2_bias_size);

    cudaMemcpy(d_conv1_weights, conv1_weights, conv1_weight_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_conv1_biases, conv1_biases, conv1_bias_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_conv2_weights, conv2_weights, conv2_weight_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_conv2_biases, conv2_biases, conv2_bias_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_fc1_weights, fc1_weights, fc1_weight_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_fc1_biases, fc1_biases, fc1_bias_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_fc2_weights, fc2_weights, fc2_weight_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_fc2_biases, fc2_biases, fc2_bias_size, cudaMemcpyHostToDevice);
}




void neural_net_check(){
    // Let's compute the sum of the weight of the conv1 matrix
    float *d_output;
    cudaMalloc((void **)&d_output, sizeof(float));
    float *output = new float[1];
    output[0] = 0.0f;
    cudaMemcpy(d_output, output, sizeof(float), cudaMemcpyHostToDevice);
    Matrix_wt_sum<<<1, 1>>>(d_conv1_weights, 20 * 5 * 5, d_output);
    cudaMemcpy(output, d_output, sizeof(float), cudaMemcpyDeviceToHost);
    cout << "Sum of the weights of the conv1 matrix: " << output[0] << endl;
    delete[] output;
    cudaFree(d_output);
}

void neural_net_free(){
    cudaFree(d_conv1_weights);
    cudaFree(d_conv1_biases);
    cudaFree(d_conv2_weights);
    cudaFree(d_conv2_biases);
    cudaFree(d_fc1_weights);
    cudaFree(d_fc1_biases);
    cudaFree(d_fc2_weights);
    cudaFree(d_fc2_biases);
}

int main(){
    // Let's check whether the neural network is reading the weights and biases correctly
    float *conv1_weights = new float[20 * 5 * 5];
    float *conv1_biases = new float[20];
    float *conv2_weights = new float[50 * 20 * 5 * 5];
    float *conv2_biases = new float[50];
    float *fc1_weights = new float[500 * 800];
    float *fc1_biases = new float[500];
    float *fc2_weights = new float[10 * 500];
    float *fc2_biases = new float[10];
    // Read the weights from the files and store them
    FILE *f = fopen("/home/cse/btech/cs1210082/380_A2/trained_weights/conv1.txt", "r");
    for(int i = 0; i < 20 * 5 * 5; i++){
        fscanf(f, "%f", &conv1_weights[i]);
    }
    for(int i = 0; i < 20; i++){
        fscanf(f, "%f", &conv1_biases[i]);
    }
    fclose(f);
    FILE *f1 = fopen("/home/cse/btech/cs1210082/380_A2/trained_weights/conv2.txt", "r");
    for(int i = 0; i < 50 * 20 * 5 * 5; i++){
        fscanf(f1, "%f", &conv2_weights[i]);
    }
    for(int i = 0; i < 50; i++){
        fscanf(f1, "%f", &conv2_biases[i]);
    }
    fclose(f1);
    FILE *f2 = fopen("/home/cse/btech/cs1210082/380_A2/trained_weights/fc1.txt", "r");
    for(int i = 0; i < 500 * 800; i++){
        fscanf(f2, "%f", &fc1_weights[i]);
    }
    for(int i = 0; i < 500; i++){
        fscanf(f2, "%f", &fc1_biases[i]);
    }
    fclose(f2);
    FILE *f3 = fopen("/home/cse/btech/cs1210082/380_A2/trained_weights/fc2.txt", "r");
    for(int i = 0; i < 10 * 500; i++){
        fscanf(f3, "%f", &fc2_weights[i]);
    }
    for(int i = 0; i < 10; i++){
        fscanf(f3, "%f", &fc2_biases[i]);
    }
    fclose(f3);
        // Set all to 1 
    // for(int i = 0; i < 20 * 5 * 5; i++){
    //     conv1_weights[i] = 1.0f;
    // }
    // for(int i = 0; i < 20; i++){
    //     conv1_biases[i] = 1.0f;
    // }
    // for(int i = 0; i < 50 * 20 * 5 * 5; i++){
    //     conv2_weights[i] = 0.01f;
    // }
    // for(int i = 0; i < 50; i++){
    //     conv2_biases[i] = 0.01f;
    // }
    // for(int i = 0; i < 500 * 800; i++){
    //     fc1_weights[i] = 0.001f;
    // }
    // for(int i = 0; i < 500; i++){
    //     fc1_biases[i] = 0.01f;
    // }
    // for(int i = 0; i < 10 * 500; i++){
    //     fc2_weights[i] = 0.001f;
    // }
    // for(int i = 0; i < 10; i++){
    //     fc2_biases[i] = 0.01f;
    // }
    neural_net_set(conv1_weights, conv1_biases, conv2_weights, conv2_biases, fc1_weights, fc1_biases, fc2_weights, fc2_biases);
    // Now let's run the neural network
    float *input = new float[28 * 28];
    float *output = new float[10];
    cout<<"I am here "<<'\n';
    auto start = chrono::high_resolution_clock::now();
    for(int i=0;i<1000;i++){
        // run for 1000 images
        string filename = "/home/cse/btech/cs1210082/380_A2/tst_imgs/" + to_string(10005000+i) + ".txt";
        // cout<<filename<<'\n';
        FILE *f_a = fopen(filename.c_str(), "r");
        for(int j = 0; j < 28 * 28; j++){
            fscanf(f_a, "%f", &input[j]);
        }
        fclose(f_a);
        cout<<i<<'\n';
        neural_net_run(input, output);
    }
    auto end = chrono::high_resolution_clock::now();
    cout << "Time taken by the neural network: " << chrono::duration_cast<chrono::microseconds>(end - start).count() << endl;
    neural_net_free();

}