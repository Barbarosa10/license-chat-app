from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from request_processor import process_image, process_text

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/api/process-data', methods=['POST'])
@cross_origin()
def process_video_frame_and_text():
    try:
        request_data = request.get_json()

        base64_encoded_image_data = request_data['videoFrame']
        text_data = request_data['text']

        image_sentiment = process_image(base64_encoded_image_data.split(',')[1])
        text_sentiment = process_text(text_data)
        return jsonify({'image_sentiment': str(image_sentiment), 'text_sentiment': str(text_sentiment)})

    except Exception as e:
        return jsonify({'error': str(e)})
        
if __name__ == '__main__':
    app.run(debug=True, port=8001)