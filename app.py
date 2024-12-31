from flask import Flask, request, jsonify, render_template
import json
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import nltk
nltk.download('punkt')
nltk.download('wordnet')

# Load dataset
with open("tree_intents_dataset.json", "r") as file:
    data = json.load(file)

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Extract patterns and labels
patterns = []
labels = []
responses = []  # To store the tree details corresponding to each pattern
for intent in data["intents"]:
    for response in intent["responses"]:
        for pattern in intent["patterns"]:
            patterns.append(pattern)
            labels.append(response["tree_name"])
            responses.append({
                "tree_name": response["tree_name"],
                "description": response["description"],
                "maintenance": response["maintenance"],
                "planting": response["planting"]
            })

# Preprocess patterns
processed_patterns = [
    " ".join([lemmatizer.lemmatize(word.lower()) for word in nltk.word_tokenize(pattern)]) for pattern in patterns
]

# Convert text data into numerical features using TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(processed_patterns)
y = labels

# Train Logistic Regression model
model = LogisticRegression(max_iter=200)
model.fit(X, y)

# Create Flask app
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")  # Load the chatbot's HTML interface

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json["message"]
    
    # Debug: Print user input
    print(f"User input: {user_input}")
    
    # Preprocess user input
    user_input_processed = " ".join([lemmatizer.lemmatize(word.lower()) for word in nltk.word_tokenize(user_input)])
    
    # Debug: Print processed user input
    print(f"Processed user input: {user_input_processed}")
    
    user_input_vectorized = vectorizer.transform([user_input_processed])
    
    # Debug: Print vectorized input
    print(f"Vectorized input: {user_input_vectorized.toarray()}")
    
    predicted_tree = model.predict(user_input_vectorized)[0]
    
    # Debug: Print predicted tree label
    print(f"Predicted tree label: {predicted_tree}")
    
    # Find the corresponding tree details
    tree_details = None
    for i, response in enumerate(responses):
        if labels[i] == predicted_tree:
            tree_details = response
            break
    
    # Debug: Print tree details
    if tree_details:
        print(f"Tree details: {tree_details}")
    
    if tree_details:
        return jsonify({
            "tree_name": tree_details["tree_name"],
            "description": tree_details["description"],
            "maintenance": tree_details["maintenance"],
            "planting": tree_details["planting"]
        })
    
    return jsonify({"message": "Sorry, I couldn't understand that."})

if __name__ == "__main__":
    app.run(debug=True)
