import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

# Load the trained model and intents
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

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

bot_name = "ChatBot"

def get_response(msg):
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    # Get the probability of the prediction
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    # Set a higher confidence threshold (e.g., 0.85 instead of 0.75)
    if prob.item() > 0.95:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])
    else:
        # Return a fallback response if the confidence is too low
        return "I do not understand... If you still have doubts, you can fill the form below, and we will contact you to resolve your doubts."

# Main chat loop
if __name__ == "__main__":
    print("Start chatting with the bot (type 'quit' to stop)!")
    while True:
        sentence = input("You: ")
        if sentence.lower() == "quit":
            break

        response = get_response(sentence)
        print(f"{bot_name}: {response}")
