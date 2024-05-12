
import numpy as np

def tanh_matrix(input_matrix):
    # Apply tanh function element-wise to the input matrix
    output_matrix = np.tanh(input_matrix)
    return output_matrix

# Example input matrix
L = [
    [-0.193, 0.249, -0.927, 0.658],
    [-0.07, 0.272, 0.544, -0.122],
    [0.923, 0.709, -0.56, -0.835]
]
input_matrix = np.array(L)
# Get the tanh matrix
output_matrix = tanh_matrix(input_matrix)

# Print the output matrix
print("Input Matrix:")
print(input_matrix)
print("\nTanh Matrix:")
print(output_matrix)

[[-0.19063882  0.24397842 -0.72919213  0.57703088]
 [-0.06988589  0.26548486  0.49600984 -0.1213983 ]
 [ 0.72731355  0.61004938 -0.50797743 -0.68315164]]