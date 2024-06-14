import tensorflow as tf
import torch as t
from transformers import pipeline

print("TensorFlow Version:", tf.__version__)
print("Torch Version:", t.__version__)

import transformers
print("Transformers Version:", transformers.__version__)
from transformers import pipeline

classifier = pipeline('zero-shot-classification')

def classify_intent(user_input):
    candidate_labels = ['greeting', 'question', 'complaint', 'feedback', 'request']
    result = classifier(user_input, candidate_labels)
    return result

# Interactive conversation
while True:
    user_message = input("You: ")
    if user_message.lower() in ["exit", "quit", "stop"]:
        print("Ending conversation.")
        break
    intent = classify_intent(user_message)
    print(f"Intent: {intent['labels'][0]} with confidence {intent['scores'][0]:.2f}")
    print()
