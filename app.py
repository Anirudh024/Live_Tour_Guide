from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Initialize Gemini with API key from .env
api_key = os.getenv('GOOGLE_API_KEY')
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in .env file!")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=api_key,
    temperature=0.3,
    max_output_tokens=200
)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # Expecting the full data URL from the frontend (e.g., "data:image/jpeg;base64,...")
        image_data_url = request.json.get('image')
        
        if not image_data_url:
            return jsonify({'insight': 'Error: No image data received'}), 400

        # Create the message using the image_url type
        # LangChain's Google integration expects a data URL string for local/base64 images
        msg = HumanMessage(
            content=[
                {
                    "type": "text", 
                    "text": "Provide one key insight about this video frame. Focus on main objects, actions, scene, or changes. Be concise."
                },
                {
                    "type": "image_url", 
                    "image_url": {"url": image_data_url}
                }
            ]
        )
        
        response = llm.invoke([msg])
        return jsonify({'insight': response.content})
    
    except Exception as e:
        return jsonify({'insight': f'Error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)