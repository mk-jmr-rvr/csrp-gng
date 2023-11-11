import random
import json
import torch
import urllib.parse
from model import NeuralNetwork
from nltk_utils import bag_of_words, tokenize
from flask import Flask, render_template, request, jsonify
import os

app = Flask(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open(os.path.join(app.root_path, 'data', 'intents.json'), 'r', encoding='utf-8') as f:
    intents = json.load(f)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNetwork(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "HandyAdam"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/map')
def map():
    return render_template('map.html')

@app.route('/prayer')
def prayer():
    return render_template('prayer.html')

@app.route('/get_response', methods=['POST'])
def get_response():
    user_message = request.form['user_message']
    if user_message.lower() == "quit":
        response = "Goodbye!"
    else:
        sentence = tokenize(user_message)
        X = bag_of_words(sentence, all_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X)

        output = model(X)
        _, predicted = torch.max(output, dim=1)
        tag = tags[predicted.item()]

        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]

        if prob.item() > 0.85:
            for intent in intents["intents"]:
                if tag == intent["tag"]:
                    response = random.choice(intent['responses'])
        else:
            google_search_link = f"https://www.google.com/search?q={urllib.parse.quote(user_message)}"
            response = f"I do not understand.. Try rephrasing that please or let's ask my friend Google! <a target='_blank' href='{google_search_link}'  class='search'> {user_message}</a>."
    
    return jsonify({'bot_response': response})

if __name__ == '__main__':
    app.run(debug=True)
