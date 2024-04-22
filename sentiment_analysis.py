import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS class
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download VADER lexicon
nltk.download('vader_lexicon')


app = Flask(__name__)
CORS(app)  # Enable CORS for your Flask app
sid = SentimentIntensityAnalyzer()

@app.route("/analyze", methods=["POST"])
def analyze_sentiment():
    # Initialize VADER
    content = request.json["content"]
    print(content)
    result = sid.polarity_scores(content)
    print(result)
    return jsonify({"sentiment": result['compound']})

@app.route("/", methods=["GET"])
def home():
    # Initialize VADER
    print("Welcome to sentiment analysis Page")
    return "<p>Please route to /analyze endpoint if you need to use this!!!!</p>"

if __name__ == "__main__":
    load_dotenv()
    port = os.getenv('PORT', 3000)
    app.run(host='0.0.0.0',port=port,debug=True)