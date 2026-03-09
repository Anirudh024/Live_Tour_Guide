from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os
import requests
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

load_dotenv()

app = Flask(__name__)
CORS(app)

api_key = os.getenv('GOOGLE_API_KEY')

# INCREASED max_output_tokens to 2048 to allow for the full 9-point narrative
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=api_key,
    temperature=0.4,
    max_output_tokens=4096 
)

def get_historical_image(query):
    """Fetches a real historical/representative image from Wikipedia."""
    try:
        search_url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={query}&format=json"
        search_data = requests.get(search_url).json()
        if not search_data['query']['search']: return None
        
        title = search_data['query']['search'][0]['title']
        img_url = f"https://en.wikipedia.org/w/api.php?action=query&titles={title}&prop=pageimages&format=json&pithumbsize=600"
        img_data = requests.get(img_url).json()
        pages = img_data['query']['pages']
        for p in pages:
            return pages[p].get('thumbnail', {}).get('source')
    except:
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        image_data_url = request.json.get('image')
        
        msg = HumanMessage(
            content=[
                {
                    "type": "text", 
                    "text": """
You are a world-class AI Tour Guide. Provide a detailed 9-point deep-dive into this image.
Include: 1. Scene, 2. History, 3. Architecture, 4. Narrative, 5. Recommendations, 6. Photo Tips, 7. Navigation, 8. Historical photo of the landmark, 9. Closing.

Format with Markdown. At the very end, add: 
LANDMARK_IDENTIFIED: [Single Name of Landmark]
"""
                },
                {
                    "type": "image_url", 
                    "image_url": {"url": image_data_url}
                }
            ]
        )
        
        response = llm.invoke([msg])
        raw_text = response.content
        
        # Extract landmark and fetch image
        hist_img = None
        if "LANDMARK_IDENTIFIED:" in raw_text:
            text_parts = raw_text.split("LANDMARK_IDENTIFIED:")
            landmark_name = text_parts[1].strip()
            hist_img = get_historical_image(landmark_name)
            clean_text = text_parts[0]
        else:
            clean_text = raw_text

        return jsonify({
            'insight': clean_text,
            'historical_image': hist_img
        })
    
    except Exception as e:
        return jsonify({'insight': f'Error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)