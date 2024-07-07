import base64
import numpy as np
from PIL import Image
from io import BytesIO
from image_processor.CNN.cnn_architecture import CNNArchitecture
from text_processor.BERT.bert_architecture import BertArchitecture
from text_processor.BERT.bert_dataset import BertDataset
from transformers import AutoModel, BertTokenizerFast
import torch
import cv2
import matplotlib.pyplot as plt

Model = BertArchitecture(AutoModel.from_pretrained('bert-base-uncased'))
Model.load_state_dict(torch.load('text_processor\BERT\saved_weights_sentiment_chat1.pt', map_location=torch.device('cpu')))
Tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

model = CNNArchitecture(28, 28)
model.load_state_dict(torch.load('image_processor\CNN\CNN_model_10.pt', map_location=torch.device('cpu')))

def image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        img_data = img_file.read()
        base64_encoded_image = base64.b64encode(img_data).decode('utf-8')

        return base64_encoded_image

def base64_to_gray_image(base64_encoded_image_data):
    binary_image_data = base64.b64decode(base64_encoded_image_data)
    nparr = np.frombuffer(binary_image_data, np.uint8)
    gray_image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

    return gray_image

def crop_face(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))
    cropped_faces = []

    for (x, y, w, h) in faces:
        face = image[y:y+h, x:x+w]
        cropped_faces.append(face)

    return cropped_faces

def preprocess_image(image):
    resized_image = cv2.resize(image, (28, 28))
    normalized_image = resized_image / 255.0
    normalized_image = np.expand_dims(normalized_image, axis=0)

    return normalized_image

def get_sentiment_for_image(base64_encoded_image_data):
    gray_image = base64_to_gray_image(base64_encoded_image_data)
    faces = crop_face(gray_image)

    label_mapping = {0: "Sad", 1: "Happy"} 

    preprocessed_faces = [preprocess_image(face) for face in faces]

    for image in preprocessed_faces:
        image_tensor = torch.tensor(image / 255.0, dtype=torch.float32)
        image_tensor = image_tensor.view(image_tensor.shape[0], 1, image_tensor.shape[2], image_tensor.shape[1])
        
        with torch.no_grad():
            output = model(image_tensor)
        print(f'output: {output}')
        probabilities = torch.softmax(output, dim=1)
        print(f'probabilities: {probabilities}')
        _, predicted_class = torch.max(probabilities, 1)
        predicted_label = label_mapping[predicted_class.item()]

        return predicted_label

def get_sentiment_for_text(Review):
    if not isinstance(Review, list):
        Review = [Review]
 
    inputs = Tokenizer.batch_encode_plus(Review,
                                        padding=True,
                                        truncation=True,
                                        add_special_tokens=True,
                                        max_length=25,
                                        return_tensors="pt")
    with torch.no_grad():
        outputs = Model(inputs['input_ids'], inputs['attention_mask'])

    print(outputs)

    label_mapping = {0: "Sad", 1: "Happy"}
    predicted_class = torch.argmax(outputs, dim=1).item()
    predicted_label = label_mapping[predicted_class]

    return predicted_label

def process_image(image_data):
    sentiment = get_sentiment_for_image(image_data)
    print(f'SENTIMENT FOR IMAGE: {sentiment}')
    return sentiment

def process_text(text_data):
    sentiment = get_sentiment_for_text(text_data)
    print(f'SENTIMENT FOR TEXT: {sentiment}')
    return sentiment