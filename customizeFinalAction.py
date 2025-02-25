import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.utils import resample
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from colorama import Fore, Style

import tkinter as tk
from tkinter import messagebox, font

# NLTK Stopwords এবং Lemmatizer ডাউনলোড
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# ✅ 1. CSV ফাইল লোড করা
df = pd.read_csv("sms_data.csv")

# ✅ 2. ডাটা ব্যালান্স করা (Spam & Important সমান করা)
df_spam = df[df["Label"] == "Spam"]
df_important = df[df["Label"] == "Important"]
df_spam_upsampled = resample(df_spam, replace=True, n_samples=len(df_important), random_state=42)
df_balanced = pd.concat([df_important, df_spam_upsampled])

# ✅ 3. উন্নত টেক্সট ক্লিনিং ফাংশন
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

df_balanced["Cleaned_Text"] = df_balanced["SMS Text"].apply(preprocess_text)

# ✅ 4. টেক্সট ভেক্টরাইজেশন (TF-IDF)
vectorizer = TfidfVectorizer()
X_balanced = vectorizer.fit_transform(df_balanced["Cleaned_Text"])
y_balanced = df_balanced["Label"]

# ✅ 5. মডেল ট্রেনিং
X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)
model = MultinomialNB()
model.fit(X_train, y_train)

# ✅ 6. একুরেসি চেক করা
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(Fore.GREEN + "✅ Model Training Successful!" + Style.RESET_ALL)
print(Fore.CYAN + "📊 Model Accuracy:", accuracy, Style.RESET_ALL)

# ✅ 7. গুরুত্বপূর্ণ কিওয়ার্ড চেক করে Prediction করা
IMPORTANT_KEYWORDS = ["otp", "bank", "transaction", "payment", "account", "salary", "bill", "invoice", "password","ট্রানজ্যাকশন"]

def check_sms(text):
    if any(keyword in text.lower() for keyword in IMPORTANT_KEYWORDS):  
        return "Important"
    text_cleaned = preprocess_text(text)
    text_vectorized = vectorizer.transform([text_cleaned])
    prediction = model.predict(text_vectorized)
    return prediction[0]

def predict_sms():
    text = entry.get("1.0", tk.END).strip()
    if not text:
        messagebox.showwarning("Warning", "Please enter an SMS!")
        return
    result = check_sms(text)
    label_result.config(text=f"Prediction: {'Important' if result == 'Important' else 'Not Important'}", fg="lime" if result == "Important" else "red")

# GUI Setup
root = tk.Tk()
root.title("SMS Classifier - Hacker Mode")
root.geometry("500x400")
root.configure(bg="black")

# Custom Font
custom_font = font.Font(family="Courier", size=12)

# Title Label
title_label = tk.Label(root, text="SMS Classifier", font=("Courier", 20, "bold"), fg="lime", bg="black")
title_label.pack(pady=10)

# Instruction Label
instruction_label = tk.Label(root, text="Enter SMS to classify:", font=custom_font, fg="lime", bg="black")
instruction_label.pack(pady=5)

# Text Entry
entry = tk.Text(root, height=5, width=50, font=custom_font, bg="black", fg="lime", insertbackground="lime")
entry.pack(pady=10)

# Predict Button
btn = tk.Button(root, text="Predict", command=predict_sms, font=custom_font, bg="black", fg="lime", relief="raised", bd=2)
btn.pack(pady=10)

# Result Label
label_result = tk.Label(root, text="", font=("Courier", 14, "bold"), bg="black")
label_result.pack(pady=10)

# Hacker-like ASCII Art
ascii_art = """
  
  ██████╗  ██████╗ ███╗   ██╗██╗
  ██╔══██╗██╔═══██╗████╗  ██║██║
  ██████╔╝██║   ██║██╔██╗ ██║██║
  ██╔══██╗██║   ██║██║╚██╗██║██║
  ██║  ██║╚██████╔╝██║ ╚████║██║
  ╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═══╝╚═╝
"""
ascii_label = tk.Label(root, text=ascii_art, font=("Courier", 8), fg="lime", bg="black")
ascii_label.pack(pady=10)

root.mainloop()