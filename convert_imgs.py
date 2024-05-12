import os
import sys
sys.path.append('/Library/Frameworks/Python.framework/Versions/3.11/lib/python3.11/site-packages')
import cv2 
import numpy as np
np.set_printoptions(linewidth=np.inf,formatter={'float': '{: 0.6f}'.format})
# Create directory if it doesn't exist
output_dir = 'tst_imgs_marked'
os.makedirs(output_dir, exist_ok=True)

# List all files in the test folder
test_folder = 'test'
i = 1e7+9
for filename in os.listdir(test_folder):
    # Read image
    i+=1
    img = cv2.imread(os.path.join(test_folder, os.path.splitext(filename)[0] + '.png'))
    # print(os.path.join(test_folder, os.path.splitext(filename)[0] + '.txt'))
    if img is None:
        print(f"Error: Unable to read {filename}")
        continue
    
    if img.shape != [28,28]:
        img2 = cv2.resize(img,(28,28))
    
    img = img2.reshape(28,28,-1);
    output_path = os.path.join(output_dir, os.path.splitext(filename)[0] + '.txt')
    #revert the image,and normalize it to 0-1 range
    img =  img/255.0
    # Need to write the image to the file
    np.savetxt(output_path, img[:,:,0], fmt='%0.6f', delimiter=' ', newline='\n')

    print(f"Converted and saved: {output_path}")

print("Conversion completed.")
