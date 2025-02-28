# SMS Classifier - Hacker Mode

This project is an AI-powered SMS classification system designed to filter and classify important SMS messages. It utilizes **Natural Language Processing (NLP)** techniques, **Machine Learning**, and a **GUI interface** for easy user interaction.

## Features
✅ Loads and processes an SMS dataset from a CSV file.  
✅ Balances the dataset by upsampling the minority class.  
✅ Cleans and preprocesses text using **NLTK**.  
✅ Converts text into TF-IDF vectors for machine learning.  
✅ Trains a **Naïve Bayes** classifier for SMS classification.  
✅ GUI built using **Tkinter** for easy SMS input and prediction.  
✅ Hacker-style **dark mode UI** with green-on-black text.  
✅ Uses keyword matching for additional filtering.  

## Installation
Ensure you have Python installed (recommended version 3.8+), then install dependencies:

```sh
pip install pandas numpy nltk scikit-learn colorama tkinter
```

## Usage
1. Place your SMS dataset CSV file (`sms_data.csv`) in the project directory.
2. Run the script:

```sh
python sms_classifier.py
```

3. Enter an SMS in the text box and click the "Predict" button.
4. The system will classify the SMS as **Important** or **Not Important**.

## Code Explanation

### 1. SMS Data Preprocessing
```python
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)
```
- Converts text to lowercase.
- Removes non-word characters and extra spaces.
- Lemmatizes words and removes stopwords.

### 2. Training the Machine Learning Model
```python
vectorizer = TfidfVectorizer()
X_balanced = vectorizer.fit_transform(df_balanced["Cleaned_Text"])
y_balanced = df_balanced["Label"]

X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)
model = MultinomialNB()
model.fit(X_train, y_train)
```
- Uses **TF-IDF** for text vectorization.
- Splits data into training and testing sets.
- Trains a **Naïve Bayes** classifier.

### 3. Predicting SMS Importance
```python
def check_sms(text):
    if any(keyword in text.lower() for keyword in IMPORTANT_KEYWORDS):  
        return "Important"
    text_cleaned = preprocess_text(text)
    text_vectorized = vectorizer.transform([text_cleaned])
    prediction = model.predict(text_vectorized)
    return prediction[0]
```
- Checks for **important keywords** like *OTP, bank, transaction*.
- If no keywords match, it uses the trained model for classification.

## GUI Interface
- Built using **Tkinter**.
- **Hacker-style** green-on-black theme.
- ASCII art header for aesthetic appeal.
- Real-time SMS classification with a "Predict" button.

## Future Improvements
- Improve keyword detection using NLP techniques.
- Implement support for multiple languages.
- Enhance GUI with additional styling and animations.

## License
This project is open-source under the **MIT License**.

---


