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
            input[i][j] = tanh(input[i][j]);
        }
    }
}

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

int main(){
    float * ip = new float[4];
    ip[0] = 2.0f;
    ip[1] = 3.0f;
    ip[2] = 5.0f;
    ip[3] = 6.0f;
    float * op = sigmoid(ip, 4);
    for(int i = 0; i < 4; i++){
        cout << op[i] << " ";
    }
    cout << endl;
    float * op2 = softmax(ip, 4);
    for(int i = 0; i < 4; i++){
        cout << op2[i] << " ";
    }
    cout << endl;
    delete [] ip;
    delete [] op;
    delete [] op2;
    return 0;
}