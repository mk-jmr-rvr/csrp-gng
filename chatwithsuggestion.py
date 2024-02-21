from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
from best_profanity.best_profanity import predict
from fuzzywuzzy import fuzz
from pro import process_sentence, has_profanity
import os
from flask import jsonify
from flask_cors import CORS


app = Flask(__name__)
CORS(app)
app.secret_key = 'hello'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open(os.path.join(app.root_path, 'data', 'intents.json'), 'r', encoding='utf-8') as f:
    intents = json.load(f)
    
with open(os.path.join(app.root_path, 'data', 'user_data.json'), 'r', encoding='utf-8') as f:
    user_data = json.load(f)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Adam"
historical_chat_logs = []

def has_profanity(sentence):
    return predict([sentence])[0] > 0.5  

@app.route('/')
def index():
    if 'username' in session:
        return redirect(url_for('chat'))
    return render_template('login.html')


@app.route('/login', methods=['POST'])
def login():
    username = request.form['username']
    password = request.form['password']

    # Check if the provided username and password match any user in the JSON data
    for user in user_data['users']:
        if user['username'] == username and user['password'] == password:
            session['username'] = username
            return redirect(url_for('chat'))

    # If username or password is incorrect, show an error
    return render_template('login.html', error='Invalid credentials')

@app.route("/chat")
def chat():
    return render_template("index.html")

@app.route('/map')
def map():
    return render_template('map.html')

@app.route('/prayer')
def prayer():
    return render_template('prayer.html')

@app.route("/get_response", methods=["POST"])
def get_response():
    user_message = request.form["user_message"]

    # Run through pro.py
    processed_sentence = process_sentence(user_message)

    # Profanity filter
    if has_profanity(processed_sentence):
        response = "I'm sorry, but please refrain from using inappropriate language."
    else:
        sentence = tokenize(user_message)
        X = bag_of_words(sentence, all_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X).to(device)

        output = model(X)
        _, predicted = torch.max(output, dim=1)

        tag = tags[predicted.item()]

        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]

        if prob.item() > 0.75:
            best_response = None
            best_similarity = 0

            chosen_intent = None  # Keep track of the chosen intent
            for intent in intents['intents']:
                if tag == intent["tag"]:
                    chosen_intent = intent  # Save the chosen intent for later use
                    for response in intent['responses']:
                        response_similarity = fuzz.ratio(user_message, response.lower())
                        if response_similarity > best_similarity:
                            best_similarity = response_similarity
                            best_response = response

            if best_response:
                # Store the chosen_intent in the session
                session['chosen_intent'] = chosen_intent
                # Prompt for user feedback
                prompt_response = f"Is this response okay? (yes/no)"
                return jsonify({"user_message": user_message, "bot_response": best_response, "prompt": prompt_response})
                
        else:
            response = "I do not understand..."

            # Suggest a question
            suggested_question = None
            max_similarity = 1.5

            for intent in intents['intents']:
                for pattern in intent['patterns']:
                    similarity = fuzz.ratio(user_message, pattern)
                    if similarity > max_similarity:
                        max_similarity = similarity
                        suggested_question = random.choice(intent['patterns'])

            if suggested_question:
                response = f"Try this instead '{suggested_question}'?"
            else:
                response = "I'm not sure what you're asking. Can you please rephrase or provide more context?"

    return jsonify({"user_message": user_message, "bot_response": response})

# Add a new route for handling user feedback
@app.route("/handle_feedback", methods=["POST"])
def handle_feedback():
    user_feedback = request.form["user_feedback"]

    if user_feedback.lower() == "yes":
        response = "Great! If you have any questions, please don't hesitate to ask again."
    elif user_feedback.lower() == "no":
        if 'chosen_intent' in session:
            chosen_intent = session['chosen_intent']
            if chosen_intent:
                # Retry with a different response from the same intent
                response = random.choice(chosen_intent['responses'])
                return jsonify({"user_message": "user_message", "bot_response": response})
        response = "Oops! Let me try again..."
    else:
        response = "I didn't understand your feedback. Please respond with 'yes' or 'no'."

    return jsonify({"bot_response": response})


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=True)
