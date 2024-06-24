import json
import tkinter as tk
from tkinter import scrolledtext
from tkinter import messagebox
from tkinter import END

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

nlp = spacy.load('en_core_web_sm')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if token.text not in stop_words and token.is_alpha]
    return tokens

with open('faqs.json', 'r') as file:
    faqs = json.load(file)

faq_questions = [faq["question"] for faq in faqs]

vectorizer = TfidfVectorizer(tokenizer=preprocess_text, stop_words=list(stop_words))
vectorized_questions = vectorizer.fit_transform(faq_questions)

def find_best_match(query, threshold=0.3):
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, vectorized_questions).flatten()
    best_match_index = similarities.argmax()
    if similarities[best_match_index] < threshold:
        return None
    return faqs[best_match_index]

def chatbot_response(user_input):
    best_match = find_best_match(user_input)
    if best_match is None:
        return "Sorry, I couldn't find the answer for your query."
    return best_match["answer"]

def handle_query():
    query = user_input.get("1.0", END).strip()
    if query:
        response = chatbot_response(query)
        chat_history.insert(tk.END, f"You: {query}\nChatbot: {response}\n\n")
        chat_history.see(tk.END)
    else:
        messagebox.showwarning("Warning", "Please enter a question.")

root = tk.Tk()
root.title("FAQ Chatbot")

user_input = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=50, height=5, font=("Helvetica", 12))
user_input.pack(pady=10)

submit_button = tk.Button(root, text="Ask", command=handle_query)
submit_button.pack()

chat_history = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=60, height=20, font=("Helvetica", 12))
chat_history.pack(pady=10)

root.mainloop()
