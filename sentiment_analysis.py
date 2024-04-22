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

if __name__ == "__main__":
    port = 5000  # Choose the port number you want to listen on
    print(f"Server running on port {port}")
    app.run(host='0.0.0.0', port=port,debug=True)