#include <iostream>
#include <math.h>
#include <filesystem>
#include <fstream>
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
                            // cout<<"here"<<endl;
                            // cout<<i<<" "<<j<<" "<<k<<" "<<l<<endl;
                            // cout<<kernel[k][l]<<endl;
                            // cout<<input[i+k-padding][j+l-padding]<<" "<<kernel[k][l]<<endl;
                            output[i][j] += input[i+k-padding][j+l-padding] * kernel[k][l];
                            // cout<<"here_too"<<endl;
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

// Need to implement a deep neural network with some pre trained weights and biases for the MNIST dataset


// Layer,Input dimension,output dimension,Input Channels,Output Channels,Kernel,Stride,Padding,Has Relu ?,No of Weights,Bias,Total Weights
// Conv_1,28,24,1,20,5,1,0,0,500,20,520
// Pool_1,24,12,20,20,2,2,0,0,-,-,-
// Conv_2,12,8,20,50,5,1,0,0,25000,50,25050
// Pool_2,8,4,50,50,2,2,0,0,-,-,-
// FC_1,4,1,50,500,4,1,0,1,400000,500,400500
// FC_2,1,1,500,10,1,1,0,0,5000,10,5010

// MNIST Input image: 28x28 pixels, grayscale so number of channels 1
 
// Conv_1:
// Input dimension N 28x28
// Input channels 1
// Output channels 20, so number of filters 20 -- each filter will produce 1 output channel
// Kernel K = 5, so each filter is a 5x5 square
// Stride 1, padding 0, so output dimension (N-K+1) = 28-5+1 = 24
// Number of parameters = 20 (number of filters) * 5 * 5 (square kernel) * 1 (input channels) + 20 (bias terms, one for each filter) = 20 * 26 = 520.
// File conv1.txt has 520 values, last 20 being the bias values. 

// Pool_1:
// Input dimension N = 24x24
// Input channels 20
// Pooling with kernel K = 2, stride 2, so output is 12x12
// Output channel 20
// Max pooling, so no weights 

// Conv_2:
// Input dimension N 12x12
// Input channels 20
// Output channels 50, so number of filters 50 -- each filter will produce 1 output channel
// Kernel K = 5, so each filter is a 5x5 square
// Stride 1, padding 0, so output dimension (N-K+1) = 12-5+1 = 8
// Number of parameters = 50 (number of filters) * 5 * 5 (square kernel) * 20 (input channels) + 50 (bias terms, one for each filter) = 50 * 501 = 25050.
// File conv2.txt has 25050 values, last 50 being the bias values. 

// Pool_2:
// Input dimension N = 8x8
// Input channels 50
// Pooling with kernel K = 2, stride 2, so output is 4x4
// Output channel 50
// Max pooling, so no weights 

// FC_1:
// Input dimension N 4x4
// Input channels 50
// Output channels 500, so number of filters 500 -- each filter will produce 1 output channel
// Kernel K = 4, so each filter is a 4x4 square
// Stride 1, padding 0, so output dimension (N-K+1) = 4-4+1 = 1
// Number of parameters = 500 (number of filters) * 4 * 4 (square kernel) * 50 (input channels) + 500 (bias terms, one for each filter) = 500 * 801 = 400500.
// File fc1.txt has 400500 values, last 500 being the bias values.
// Has a relu layer.
 
// FC_2:
// Input dimension N 1x1
// Input channels 500
// Output channels 10, so number of filters 10 -- each filter will produce 1 output channel
// Kernel K = 1, so each filter is a 1x1 square
// Stride 1, padding 0, so output size (N-K+1) = 1-1+1 = 1
// Number of parameters = 10 (number of filters) * 1 * 1 (square kernel) * 500 (input channels) + 10 (bias terms, one for each filter) = 10 * 501 = 5010.
// File fc2.txt has 5010 values, last 10 being the bias values.


// Let's start by first assuming we have the input image in a 2D array of size 28x28
// We will then read the weights and biases from the files and perform the operations as described above

// Assume that the input images are stored in a 2D array input_image of size 28x28 in the file input.txt
// Assume that the weights and biases are stored in the files conv1.txt, conv2.txt, fc1.txt, fc2.txt within the trained_weights folder

float* neural_net(float ** input_image, float *** conv1_weights, float * conv1_biases, float **** conv2_weights, float * conv2_biases, float **** fc1_weights, float * fc1_biases, float ** fc2_weights, float * fc2_biases){
    // Perform the operations as described above
    // Conv_1
    float *** conv1_output = new float**[20];
    for(int i = 0; i < 20; i++){
        conv1_output[i] = convolution(input_image, conv1_weights[i], 28, 5, 0);
        for(int j = 0; j < 24; j++){
            for(int k = 0; k < 24; k++){
                conv1_output[i][j][k] += conv1_biases[i];
            }
        }
    }
    // Max Pooling
    float *** pool1_output = new float**[20];
    for(int i = 0; i < 20; i++){
        pool1_output[i] = max_pooling(conv1_output[i], 2, 24);
    }
    // Conv_2
    float *** conv2_output = new float**[50];
    for(int i=0; i<50; i++){
        conv2_output[i] = new float*[8];
        for(int j=0; j<8; j++){
            conv2_output[i][j] = new float[8];
        }
        // set to zero
        for(int j=0; j<8; j++){
            for(int k=0; k<8; k++){
                conv2_output[i][j][k] = 0.0f;
            }
        }
        // Add for every filter
        for(int j=0; j<20; j++){
            float ** temp = convolution(pool1_output[j], conv2_weights[i][j], 12, 5, 0);
            for(int k=0; k<8; k++){
                for(int l=0; l<8; l++){
                    conv2_output[i][k][l] += temp[k][l];
                }
            }
        }
        // Add bias
        for(int j=0; j<8; j++){
            for(int k=0; k<8; k++){
                conv2_output[i][j][k] += conv2_biases[i];
            }
        }
    }
    // Max Pooling
    float *** pool2_output = new float**[50];
    for(int i=0; i<50; i++){
        pool2_output[i] = max_pooling(conv2_output[i], 2, 8);
    }
    // Apply the fully connected layer
    float * fc1_output = new float[500];
    for(int i=0; i<500; i++){
        fc1_output[i] = 0.0f;
    }
    for(int i=0; i<500; i++){
        for(int j=0; j<50; j++){
            float ** temp = convolution(pool2_output[j], fc1_weights[i][j], 4, 4, 0);
            fc1_output[i] += temp[0][0];
        }
        fc1_output[i] += fc1_biases[i];
    }
    // Apply ReLU
    ReLU_mat(&fc1_output, 1, 500);
    // Apply the second fully connected layer
    float * fc2_output = new float[10];
    for(int i=0; i<10; i++){
        fc2_output[i] = 0.0f;
    }
    for(int i=0; i<10; i++){
        for(int j=0; j<500; j++){
            fc2_output[i] += fc1_output[j] * fc2_weights[i][j];
        }
        fc2_output[i] += fc2_biases[i];
    }
    // Apply softmax
    float * output = softmax(fc2_output, 10);
    return output;
}





int main(){
    // Read the input image from the file input.txt
    // float ** input_image = new float*[28];
    // for(int i = 0; i < 28; i++){
    //     input_image[i] = new float[28];
    // }
    // FILE * input_file = fopen("input.txt", "r");
    // for(int i = 0; i < 28; i++){
    //     for(int j = 0; j < 28; j++){
    //         fscanf(input_file, "%f", &input_image[i][j]);
    //     }
    // }

    // fclose(input_file);
    // cout<<"Input Image completed"<<endl;
    // Read the weights and biases for the first convolution layer
    float *** conv1_weights = new float**[20];
    for(int i = 0; i < 20; i++){
        conv1_weights[i] = new float*[5];
        for(int j = 0; j < 5; j++){
            conv1_weights[i][j] = new float[5];
        }
    }
    float * conv1_biases = new float[20];
    FILE * conv1_file = fopen("trained_weights/conv1.txt", "r");
    for(int i = 0; i < 20; i++){
        for(int j = 0; j < 5; j++){
            for(int k = 0; k < 5; k++){
                fscanf(conv1_file, "%f", &conv1_weights[i][j][k]);
            }
        }
    }
    for(int i = 0; i < 20; i++){
        fscanf(conv1_file, "%f", &conv1_biases[i]);
    }
    fclose(conv1_file);
    // cout<<"Conv1 completed"<<endl;
    // Read the weights and biases for the second convolution layer
    float **** conv2_weights = new float***[50];
    for(int i=0; i<50; i++){
        conv2_weights[i] = new float**[20];
        for(int j=0; j<20; j++){
            conv2_weights[i][j] = new float*[5];
            for(int k=0; k<5; k++){
                conv2_weights[i][j][k] = new float[5];
            }
        }
    }
    float * conv2_biases = new float[50];
    FILE * conv2_file = fopen("trained_weights/conv2.txt", "r");
    for(int i=0; i<50; i++){
        for(int j=0; j<20; j++){
            for(int k=0; k<5; k++){
                for(int l=0; l<5; l++){
                    fscanf(conv2_file, "%f", &conv2_weights[i][j][k][l]);
                }
            }
        }
    }
    for(int i=0; i<50; i++){
        fscanf(conv2_file, "%f", &conv2_biases[i]);
    }
    fclose(conv2_file);
    // cout<<"Conv2 completed"<<endl;
    // Read the weights and biases for the first fully connected layer
    float **** fc1_weights = new float***[500];
    for(int i=0;i<500;i++){
        fc1_weights[i] = new float**[50];
        for(int j=0;j<50;j++){
            fc1_weights[i][j] = new float*[4];
            for(int k=0;k<4;k++){
                fc1_weights[i][j][k] = new float[4];
            }
        }
    }
    float * fc1_biases = new float[500];
    FILE * fc1_file = fopen("trained_weights/fc1.txt", "r");
    for(int i=0;i<500;i++){
        for(int j=0;j<50;j++){
            for(int k=0;k<4;k++){
                for(int l=0;l<4;l++){
                    fscanf(fc1_file, "%f", &fc1_weights[i][j][k][l]);
                }
            }
        }
    }
    for(int i=0;i<500;i++){
        fscanf(fc1_file, "%f", &fc1_biases[i]);
    }
    fclose(fc1_file);
    // cout<<"FC1 completed"<<endl;
    // Read the weights and biases for the second fully connected layer
    float ** fc2_weights = new float*[10];
    for(int i=0;i<10;i++){
        fc2_weights[i] = new float[500];
    }
    float * fc2_biases = new float[10];
    FILE * fc2_file = fopen("trained_weights/fc2.txt", "r");
    for(int i=0;i<10;i++){
        for(int j=0;j<500;j++){
            fscanf(fc2_file, "%f", &fc2_weights[i][j]);
        }
    }
    for(int i=0;i<10;i++){
        fscanf(fc2_file, "%f", &fc2_biases[i]);
    }
    fclose(fc2_file);


    std::string folder = "tst_imgs";
    std::filesystem::directory_iterator it(folder);
    int count = 0;
    while(it != std::filesystem::directory_iterator()){
        count++;
        it++;
        if(count == 1000){
            break;
        }
        // Now load the image 
        float ** input_image = new float*[28];
        for(int i = 0; i < 28; i++){
            input_image[i] = new float[28];
        }
        std::string file = it->path();
        std::ifstream input_file(file);
        for(int i = 0; i < 28; i++){
            for(int j = 0; j < 28; j++){
                input_file >> input_image[i][j];
            }
        }
        input_file.close();
        float * output = neural_net(input_image, conv1_weights, conv1_biases, conv2_weights, conv2_biases, fc1_weights, fc1_biases, fc2_weights, fc2_biases);
        // for(int i = 0; i < 10; i++){
        //     cout << output[i] << " ";
        // }
        // cout << endl;

    }

    // Need to do the following:
    // Choose files from the tst_imgs folder 100 and apply the neural net on them
    
    // float * output = neural_net(input_image, conv1_weights, conv1_biases, conv2_weights, conv2_biases, fc1_weights, fc1_biases, fc2_weights, fc2_biases);

    // cout<<"FC2 completed"<<endl;
    // Perform the operations as described above
    // Conv_1
    // float *** conv1_output = new float**[20];
    // for(int i = 0; i < 20; i++){
    //     conv1_output[i] = convolution(input_image, conv1_weights[i], 28, 5, 0);
    //     for(int j = 0; j < 24; j++){
    //         for(int k = 0; k < 24; k++){
    //             conv1_output[i][j][k] += conv1_biases[i];
    //         }
    //     }
    // }
    // // cout<<"Conv1 output completed"<<endl;
    // // Max Pooling
    // float *** pool1_output = new float**[20];
    // for(int i = 0; i < 20; i++){
    //     pool1_output[i] = max_pooling(conv1_output[i], 2, 24);
    // }
    // // cout<<"Pool1 output completed"<<endl;
    // // Conv_2
    // float *** conv2_output = new float**[50];
    // for(int i=0; i<50; i++){
    //     conv2_output[i] = new float*[8];
    //     for(int j=0; j<8; j++){
    //         conv2_output[i][j] = new float[8];
    //     }
    //     // set to zero
    //     for(int j=0; j<8; j++){
    //         for(int k=0; k<8; k++){
    //             conv2_output[i][j][k] = 0.0f;
    //         }
    //     }
    //     // Add for every filter
    //     for(int j=0; j<20; j++){
    //         float ** temp = convolution(pool1_output[j], conv2_weights[i][j], 12, 5, 0);
    //         for(int k=0; k<8; k++){
    //             for(int l=0; l<8; l++){
    //                 conv2_output[i][k][l] += temp[k][l];
    //             }
    //         }
    //     }
    //     // Add bias
    //     for(int j=0; j<8; j++){
    //         for(int k=0; k<8; k++){
    //             conv2_output[i][j][k] += conv2_biases[i];
    //         }
    //     }
    // }
    // // cout<<"Conv2 output completed"<<endl;
    // // Max Pooling
    // float *** pool2_output = new float**[50];
    // for(int i=0; i<50; i++){
    //     pool2_output[i] = max_pooling(conv2_output[i], 2, 8);
    // }
    // // cout<<"Pool2 output completed"<<endl;
    // // Apply the fully connected layer
    // float * fc1_output = new float[500];
    // for(int i=0; i<500; i++){
    //     fc1_output[i] = 0.0f;
    // }
    // for(int i=0; i<500; i++){
    //     for(int j=0; j<50; j++){
    //         for(int k=0; k<4; k++){
    //             for(int l=0; l<4; l++){
    //                 fc1_output[i] += pool2_output[j][k][l] * fc1_weights[i][j][k][l];
    //             }
    //         }
    //     }
    //     fc1_output[i] += fc1_biases[i];
    // }
    // // cout<<"FC1 output completed"<<endl;
    // // Apply ReLU
    // ReLU_mat(&fc1_output, 1, 500);
    // // cout<<"ReLU output completed"<<endl;
    // // Apply the second fully connected layer
    // float * fc2_output = new float[10];
    // for(int i=0; i<10; i++){
    //     fc2_output[i] = 0.0f;
    // }
    // for(int i=0; i<10; i++){
    //     for(int j=0; j<500; j++){
    //         fc2_output[i] += fc1_output[j] * fc2_weights[i][j];
    //     }
    //     fc2_output[i] += fc2_biases[i];
    // }
    // cout<<"FC2 output completed"<<endl;
    // Apply softmax
    // float * output = softmax(fc2_output, 10);
    // cout<<"Softmax output completed"<<endl;
    // // Print the output
    // for(int i = 0; i < 10; i++){
    //     cout << output[i] << " ";
    // }
    // cout << endl;
}
