from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import base64
import numpy as np

from PIL import Image
from io import BytesIO
import base64

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/api/process-data', methods=['POST'])
@cross_origin()
def process_video_frame_and_text():
    try:
        request_data = request.get_json()

        #get data
        base64_encoded_image_data = request_data['videoFrame']
        text_data = request_data['text']

        # Decode Base64 to binary
        binary_image_data = base64.b64decode(base64_encoded_image_data.split(',')[1])

        #Process image and text
        process_video(binary_image_data)
        process_text(text_data)
        return jsonify({'message': str("BINE")})

    except Exception as e:
        return jsonify({'error': str(e)})

def process_video(image_data):
    print(image_data)

    # Open the decoded image using Pillow
    image = Image.open(BytesIO(image_data))

    # Display the image
    image.show()

def process_text(text_data):
    print(text_data)

if __name__ == '__main__':
    app.run(debug=True, port=8001)
