#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <time.h>

using namespace std;

// Convolution function
float ** convolution(float ** input, float ** kernel, int input_size, int kernel_size, int padding){
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
            input[i][j] = tanhf(input[i][j]);
        }
    }
}

float ** max_pooling(float ** input, int pool_size, int input_size){
    // stride = 1
    int output_size = input_size - pool_size + 1;
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
                    max_val = max(max_val, input[i+k][j+l]);
                }
            }
            output[i][j] = max_val;
        }
    }
    return output;

}

float ** avg_pooling(float **input, int pool_size, int input_size){

    int output_size = input_size - pool_size + 1;
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
                    sum += input[i+k][j+l];
                }
            }
            output[i][j] = sum / (pool_size * pool_size);
        }
    }
    return output;

}

float * sigmoid(float * input, int size){
    float * output = new float[size];
    for(int i = 0; i < size; i++){
        output[i] = 1.0f / (1.0f + exp(-input[i]));
    }
    return output;
}

float * softmax(float * input, int size){
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


void print_mat(float ** input, int row, int cols){
    for(int i = 0; i < row; i++){
        for(int j = 0; j < cols; j++){
            cout << input[i][j] << " ";
        }
        cout << endl;
    }
}

int main(int argc, char *argv[]) {
    int arg1 = atoi(argv[1]);
    // Now we need to make cases
    switch (arg1) {
        case 1: {
            // Read N,M,P
            int N, M, P;
            N = atoi(argv[2]);
            M = atoi(argv[3]);
            P = atoi(argv[4]);
            // Read input matrix
            float **input = new float *[N];
            for (int i = 0; i < N; i++) {
                input[i] = new float[N];
            }
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    input[i][j] = atof(argv[5 + i * N + j]);
                }
            }
            // Read kernel matrix
            float **kernel = new float *[M];
            for (int i = 0; i < M; i++) {
                kernel[i] = new float[M];
            }
            for (int i = 0; i < M; i++) {
                for (int j = 0; j < M; j++) {
                    kernel[i][j] = atof(argv[5 + N * N + i * M + j]);
                }
            }
            // Now perform the convolution
            float **output = convolution(input, kernel, N, M, P);
            // Print the output
            int output_size = N - M + 1 + 2 * P;
            for (int i = 0; i < output_size; i++) {
                for (int j = 0; j < output_size; j++) {
                    cout << output[i][j] << " ";
                }
                // cout << '\n';
            }
            break;
        }
        case 2:{
            // read the first input
            int arg2 = atoi(argv[2]);
            if(arg2 == 0){
                // Read N and M 
                int N, M;
                N = atoi(argv[3]);
                M = atoi(argv[4]);
                // Apply the ReLU function
                float **input = new float*[N];
                for(int i = 0; i < N; i++){
                    input[i] = new float[M];
                }
                for(int i = 0; i < N; i++){
                    for(int j = 0; j < M; j++){
                        input[i][j] = atof(argv[5 + i*M + j]);
                    }
                }
                ReLU_mat(input, N, M);
                for(int i = 0; i < N; i++){
                    for(int j = 0; j < M; j++){
                        cout << input[i][j] << " ";
                    }
                    
                }
                // cout<<'\n';
            }
            else{
                int N, M;
                N = atoi(argv[3]);
                M = atoi(argv[4]);
                // Apply the ReLU function
                float **input = new float*[N];
                for(int i = 0; i < N; i++){
                    input[i] = new float[M];
                }
                for(int i = 0; i < N; i++){
                    for(int j = 0; j < M; j++){
                        input[i][j] = atof(argv[5 + i*M + j]);
                    }
                }
                tanh_mat(input, N, M);
                for(int i = 0; i < N; i++){
                    for(int j = 0; j < M; j++){
                        cout << input[i][j] << " ";
                    }
                    
                }
                // cout<<'\n';
            }
            break;
        }
        case 3:{
            int arg2 = atoi(argv[2]);
            int N,M;
            M = atoi(argv[3]);
            N = atoi(argv[4]);
            float **input = new float*[N];
            for(int i = 0; i < N; i++){
                input[i] = new float[N];
            }
            for(int i = 0; i < N; i++){
                for(int j = 0; j < N; j++){
                    input[i][j] = atof(argv[5 + i*N + j]);
                }
            }
            if(arg2 == 0){
                // apply max pooling
                int pool_size = M;
                float **output = max_pooling(input, pool_size, N);
                int op_size = N - M + 1;
                for(int i = 0; i < op_size; i++){
                    for(int j = 0; j < op_size; j++){
                        cout << output[i][j] << " ";
                    }
                    // cout << '\n';
                }
            }
            else{
                // apply avg pooling
                int pool_size = M;
                float **output = avg_pooling(input, pool_size, N);
                int op_size = N - M + 1;
                for(int i = 0; i < op_size; i++){
                    for(int j = 0; j < op_size; j++){
                        cout << output[i][j] << " ";
                    }
                    // cout << '\n';
                }
            
            }
            break;
        }
        case 4:{
            int arg2 = atoi(argv[2]);
            int n = argc;
            float *input = new float[n-3];
            for(int i = 0; i < n-3; i++){
                input[i] = atof(argv[3 + i]);
            }
            if(arg2 == 0){
                float *output = sigmoid(input, n-3);
                for(int i = 0; i < n-3; i++){
                    cout << output[i] << " ";
                }
            }
            else{
                float *output = softmax(input, n-3);
                for(int i = 0; i < n-3; i++){
                    cout << output[i] << " ";
                }
            }
            break;
        }
        default:
            break;
    }

    return 0;
}