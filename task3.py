# ----------------------------------------
# CODTECH Internship Task 3
# Standalone NLP Chatbot (No external file)
# ----------------------------------------

import random
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------
# Step 1: Embedded corpus
# -------------------------------
corpus = [
    "Hello! I am your friendly chatbot.",
    "I can answer questions about programming, Python, and general queries.",
    "Python is a popular programming language.",
    "NLTK is a library used for natural language processing in Python.",
    "You can ask me about Python, NLP, or general greetings.",
    "I am here to help you learn and answer simple questions.",
    "I can also chat with you about general topics."
]

# -------------------------------
# Step 2: Preprocessing functions
# -------------------------------
lemmer_dict = {}

def LemTokens(tokens):
    # Simple lowercasing as lemmatization replacement
    return [token.lower() for token in tokens]

remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def LemNormalize(text):
    return LemTokens(text.lower().translate(remove_punct_dict).split())

# -------------------------------
# Step 3: Greeting function
# -------------------------------
GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "hey")
GREETING_RESPONSES = ["Hi there!", "Hello!", "Hey!", "Greetings!"]

def greeting(sentence):
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)

# -------------------------------
# Step 4: Response function
# -------------------------------
def response(user_input):
    robo_response = ''
    sent_tokens = corpus.copy()
    sent_tokens.append(user_input)  # add user input to corpus
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]  # second most similar
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]

    if req_tfidf == 0:
        robo_response = "I am sorry! I don't understand you."
    else:
        robo_response = sent_tokens[idx]
    return robo_response

# -------------------------------
# Step 5: Main chat loop
# -------------------------------
flag = True
print("BOT: Hello! I am your chatbot. Type 'bye' to exit.")

while flag:
    user_input = input().lower()
    if user_input != 'bye':
        if greeting(user_input) is not None:
            print("BOT:", greeting(user_input))
        else:
            print("BOT:", response(user_input))
    else:
        flag = False
        print("BOT: Goodbye! Have a nice day.")
