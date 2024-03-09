# import os
# import shutil

# def rename_images(folder_path):
#     # List all files in the folder
#     files = os.listdir(folder_path)
    
#     # Iterate over each file
#     for file in files:
#         # Check if the file is an image (you can customize this check based on your image file extensions)
#         if file.endswith('.jpg') or file.endswith('.png') or file.endswith('.jpeg'):
#             # Construct the new file name by adding 'a' to the beginning
#             new_name = 'a_' + file
            
#             # Get the full paths of the original and new files
#             old_path = os.path.join(folder_path, file)
#             new_path = os.path.join(folder_path, new_name)
            
#             # Rename the file
#             os.rename(old_path, new_path)
            
#             print(f'Renamed: {old_path} -> {new_path}')

# # Replace 'folder_path' with the path to your folder containing images
# folder_path = 'C:\\Users\\duciu\Downloads\\archive (10)\\images\images\\train\\angry'
# rename_images(folder_path)

import matplotlib.pyplot as plt
import cv2

image = cv2.imread('happy1.jpg', cv2.IMREAD_GRAYSCALE)

image = cv2.resize(image, (48, 48))

plt.imshow(image, cmap='gray')
plt.show()