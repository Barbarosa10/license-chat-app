from cnn_architecture import CNNArchitecture
import torch


import cv2
import base64
import numpy as np
import matplotlib.pyplot as plt

def image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        # Read the image file
        img_data = img_file.read()
        # Convert image data to base64 encoding
        base64_encoded_image = base64.b64encode(img_data).decode('utf-8')
        return base64_encoded_image

def base64_to_gray_image(base64_encoded_image_data):
    # Decode base64 image data
    binary_image_data = base64.b64decode(base64_encoded_image_data)

    # Convert to numpy array
    nparr = np.frombuffer(binary_image_data, np.uint8)

    # Read image in grayscale
    gray_image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

    return gray_image

def crop_face(image):
    # Load face cascade classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Crop face regions
    cropped_faces = []
    for (x, y, w, h) in faces:
        face = image[y:y+h, x:x+w]
        cropped_faces.append(face)

    return cropped_faces

def preprocess_image(image):
    # Resize image to 28x28 pixels
    resized_image = cv2.resize(image, (28, 28))
    # Normalize pixel values
    normalized_image = resized_image / 255.0

    # Add channel dimension
    normalized_image = np.expand_dims(normalized_image, axis=0)

    return normalized_image

# # Example usage
image_path = "p3.jpg"  # Path to your image file
base64_image = image_to_base64(image_path)
print(base64_image)

# Example usage
base64_encoded_image_data = base64_image  # Your base64 encoded image data
gray_image = base64_to_gray_image(base64_encoded_image_data)
faces = crop_face(gray_image)



label_mapping = {0: "Negative", 1: "Positive"} 

# Preprocess each face
preprocessed_faces = [preprocess_image(face) for face in faces]
for image in preprocessed_faces:


    image_tensor = torch.tensor(image / 255.0, dtype=torch.float32)
    image_tensor = image_tensor.view(image_tensor.shape[0], 1, image_tensor.shape[2], image_tensor.shape[1])
    
    model = CNNArchitecture(28, 28)
    model.load_state_dict(torch.load('CNN_model_6.pt', map_location=torch.device('cpu')))
    # model.eval()
    # Forward pass
    with torch.no_grad():
        output = model(image_tensor)
    print(f'output: {output}')
    probabilities = torch.softmax(output, dim=1)
    print(f'probabilities: {probabilities}')
    _, predicted_class = torch.max(probabilities, 1)

    predicted_label = label_mapping[predicted_class.item()]


    print(predicted_label)

    plt.imshow(image[0], cmap='gray')  # Assuming images are grayscale
    plt.show()


# def Get_sentiment_for_image(base64_encoded_image_data):
#     gray_image = base64_to_gray_image(base64_encoded_image_data)
#     faces = crop_face(gray_image)

#     label_mapping = {0: "Negative", 1: "Positive"} 

#     # Preprocess each face
#     preprocessed_faces = [preprocess_image(face) for face in faces]
    
#     for image in preprocessed_faces:


#         image_tensor = torch.tensor(image / 255.0, dtype=torch.float32)
#         image_tensor = image_tensor.view(image_tensor.shape[0], 1, image_tensor.shape[2], image_tensor.shape[1])
        
#         model = CNNArchitecture(28, 28)
#         model.load_state_dict(torch.load('CNN_model_6.pt', map_location=torch.device('cpu')))
#         # model.eval()
#         # Forward pass
#         with torch.no_grad():
#             output = model(image_tensor)
#         print(f'output: {output}')
#         probabilities = torch.softmax(output, dim=1)
#         print(f'probabilities: {probabilities}')
#         _, predicted_class = torch.max(probabilities, 1)

#         predicted_label = label_mapping[predicted_class.item()]


#         return predicted_label

