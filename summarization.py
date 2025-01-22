from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for the app

# Hugging Face summarization pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

@app.route('/review-summary', methods=['POST'])
def review_summary():
    """
    Accept a list of reviews and return a summarized review.
    """
    try:
        # Get JSON data from the request
        data = request.get_json()

        # Validate input
        if not data or "reviews" not in data or not isinstance(data["reviews"], list):
            return jsonify({"status": "error", "message": "Invalid input. Provide a 'reviews' field with a list of strings."}), 400

        # Combine reviews into a single string
        reviews = "Summarize the following customer feedback in a neutral, objective tone, without using first-person pronouns like 'I' or 'My': " + " ||| ".join(data["reviews"])

        # If no reviews provided, return an appropriate message
        if not reviews.strip():
            return jsonify({"status": "success", "summary": "No reviews provided."}), 200

        # Summarize the reviews using Hugging Face
        summary = summarizer(reviews, max_length=100, min_length=30, do_sample=False)
        summary_text = summary[0]["summary_text"]

        # Return the summary
        return jsonify({"status": "success", "summary": summary_text}), 200

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.errorhandler(404)
def not_found(e):
    """
    Handle 404 errors.
    """
    return jsonify({"status": "error", "message": "Endpoint not found"}), 404

if __name__ == '__main__':
    app.run(debug=True)
