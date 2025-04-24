# Simple Sentiment and Emotion Analysis (From Scratch Concept and Simple ML)

# --- Simple Sentiment Analysis (Lexicon-based - Existing Code) ---

# A very basic lexicon of positive and negative words
# In a real system, this lexicon would be much larger and more sophisticated
sentiment_lexicon = {
    "positive": ["good", "great", "excellent", "amazing", "happy", "love", "joy", "wonderful", "fantastic", "awesome"],
    "negative": ["bad", "poor", "terrible", "awful", "sad", "hate", "pain", "horrible", "disappointing", "worse"]
}

def analyze_sentiment_lexicon(text):
    """
    Analyzes sentiment (positive/negative) of a given text using a simple lexicon.
    Returns 'Positive', 'Negative', or 'Neutral'.
    """
    words = text.lower().split()
    positive_score = 0
    negative_score = 0

    for word in words:
        # Remove punctuation for simpler matching
        cleaned_word = ''.join(filter(str.isalpha, word))
        if cleaned_word in sentiment_lexicon["positive"]:
            positive_score += 1
        elif cleaned_word in sentiment_lexicon["negative"]:
            negative_score += 1

    if positive_score > negative_score:
        return "Positive"
    elif negative_score > positive_score:
        return "Negative"
    else:
        return "Neutral"

# --- Simple Emotion Analysis (Lexicon-based - Existing Code) ---

# A very basic lexicon of words associated with simple emotions
# This is a highly simplified example
emotion_lexicon = {
    "joy": ["happy", "joy", "excited", "glad", "cheerful"],
    "sadness": ["sad", "unhappy", "depressed", "gloomy", "tear"],
    "anger": ["angry", "mad", "furious", "irritated", "hate"],
    "fear": ["scared", "afraid", "fearful", "anxious", "terrified"]
}

def analyze_emotion_lexicon(text):
    """
    Analyzes basic emotions expressed in a given text using a simple lexicon.
    Returns a dictionary of emotion scores.
    """
    words = text.lower().split()
    emotion_scores = {emotion: 0 for emotion in emotion_lexicon}

    for word in words:
        # Remove punctuation for simpler matching
        cleaned_word = ''.join(filter(str.isalpha, word))
        for emotion, emotion_words in emotion_lexicon.items():
            if cleaned_word in emotion_words:
                emotion_scores[emotion] += 1

    return emotion_scores

# --- Simple Sentiment Model (From Scratch Concept - Existing Code) ---

def clean_text_simple(text):
    """Simple text cleaning for the basic model: lowercase and remove non-alpha characters."""
    return ''.join(filter(str.isalpha, text.lower()))

def train_simple_sentiment_model(positive_texts, negative_texts):
    """
    Trains a very simple frequency-based sentiment model.
    Counts word occurrences in positive and negative texts.
    """
    print("\n--- Training Simple Sentiment Model (Frequency-based) ---")
    pos_counts = {}
    neg_counts = {}

    # Count words in positive texts
    for text in positive_texts:
        words = text.split()
        for word in words:
            cleaned_word = clean_text_simple(word)
            if cleaned_word: # Only process if not empty after cleaning
                pos_counts[cleaned_word] = pos_counts.get(cleaned_word, 0) + 1

    # Count words in negative texts
    for text in negative_texts:
        words = text.split()
        for word in words:
            cleaned_word = clean_text_simple(word)
            if cleaned_word: # Only process if not empty after cleaning
                neg_counts[cleaned_word] = neg_counts.get(cleaned_word, 0) + 1

    print("Training complete.")
    return pos_counts, neg_counts

def predict_simple_sentiment(text, pos_counts, neg_counts):
    """
    Predicts sentiment using the simple frequency-based model.
    Scores text based on learned word frequencies.
    """
    words = text.split()
    pos_score = 0
    neg_score = 0

    for word in words:
        cleaned_word = clean_text_simple(word)
        if cleaned_word:
            pos_score += pos_counts.get(cleaned_word, 0)
            neg_score += neg_counts.get(cleaned_word, 0)

    # Simple prediction logic: higher score wins
    if pos_score > neg_score:
        return "Positive"
    elif neg_score > pos_score:
        return "Negative"
    else:
        return "Neutral"

# --- Machine Learning Sentiment Model (using scikit-learn) ---

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np # Import numpy for score mapping

def train_ml_sentiment_model(train_texts, train_labels):
    """
    Trains a simple Logistic Regression sentiment model using TF-IDF features.
    Requires scikit-learn.
    """
    print("\n--- Training Machine Learning Sentiment Model (Logistic Regression) ---")
    # Vectorize text data using TF-IDF
    vectorizer = TfidfVectorizer(max_features=1000) # Limit features for simplicity
    X_train = vectorizer.fit_transform(train_texts)

    # Train a Logistic Regression model
    model = LogisticRegression()
    model.fit(X_train, train_labels)

    print("Training complete.")
    return vectorizer, model

def predict_ml_sentiment(text, vectorizer, model):
    """
    Predicts sentiment score (-1 to +1) using the trained ML model.
    Requires scikit-learn and numpy.
    """
    # Vectorize the input text
    X_test = vectorizer.transform([text])

    # Predict probability of the positive class (assuming label 1 is positive)
    # The probabilities are typically [prob_negative, prob_positive]
    probabilities = model.predict_proba(X_test)[0]
    positive_probability = probabilities[1] # Probability of the positive class

    # Map probability (0 to 1) to sentiment score (-1 to +1)
    # Score = 2 * probability - 1
    sentiment_score = 2 * positive_probability - 1

    return sentiment_score

# --- Sample Usage ---

# Sample training data for the simple frequency model (Existing Usage)
sample_train_positive_freq = [
    "I love this product, it is great!",
    "This is an amazing experience, truly wonderful.",
    "Feeling happy and joyful today."
]

sample_train_negative_freq = [
    "I hate this product, it is terrible.",
    "This is an awful experience, truly disappointing.",
    "Feeling sad and gloomy today."
]

# Train the simple frequency model (Existing Usage)
positive_word_counts, negative_word_counts = train_simple_sentiment_model(
    sample_train_positive_freq, sample_train_negative_freq
)

# Sample texts for prediction using the simple frequency model (Existing Usage)
sample_texts_for_freq_model = [
    "This is great!",
    "This is terrible!",
    "It is okay.",
    "I feel happy.",
    "I feel sad.",
    "great terrible" # Example with both positive and negative words
]

print("\n--- Simple Sentiment Model Prediction Results (Frequency-based) ---")
for text in sample_texts_for_freq_model:
    sentiment = predict_simple_sentiment(text, positive_word_counts, negative_word_counts)
    print(f"Text: '{text}'")
    print(f"Predicted Sentiment: {sentiment}\n")


# Sample training data for the ML model
# Labels: 0 for negative, 1 for positive
sample_train_texts_ml = [
    "This is a great movie!",
    "I love this product.",
    "The service was terrible.",
    "Feeling very happy today.",
    "This is awful.",
    "What a wonderful experience.",
    "I am so sad.",
    "Excellent work!",
    "Poor quality.",
    "It was okay." # Neutral example
]
sample_train_labels_ml = [1, 1, 0, 1, 0, 1, 0, 1, 0, 0] # Assign labels (1: positive, 0: negative/neutral)

# Train the ML model
ml_vectorizer, ml_model = train_ml_sentiment_model(
    sample_train_texts_ml, sample_train_labels_ml
)

# Sample texts for prediction using the ML model
sample_texts_for_ml_model = [
    "I had a wonderful time.",
    "This is the worst experience.",
    "It was an average day.",
    "I am feeling great!",
    "This is disappointing."
]

print("\n--- Machine Learning Sentiment Model Prediction Results (Score -1 to +1) ---")
for text in sample_texts_for_ml_model:
    sentiment_score = predict_ml_sentiment(text, ml_vectorizer, ml_model)
    print(f"Text: '{text}'")
    print(f"Predicted Sentiment Score: {sentiment_score:.4f}\n")


# Keep existing lexicon-based analysis usage for comparison (Existing Usage)
print("--- Sentiment Analysis Results (Lexicon-based) ---")
sample_texts_lexicon = [
    "This is a great day, I am so happy!",
    "The movie was terrible and disappointing.",
    "It was an okay experience, nothing special.",
    "I am very angry about the horrible service.",
    "Feeling scared and anxious about the test."
]
for text in sample_texts_lexicon:
    sentiment = analyze_sentiment_lexicon(text)
    print(f"Text: '{text}'")
    print(f"Sentiment: {sentiment}\n")

print("--- Emotion Analysis Results (Lexicon-based) ---")
for text in sample_texts_lexicon:
    emotions = analyze_emotion_lexicon(text)
    print(f"Text: '{text}'")
    print(f"Emotions: {emotions}\n")

# --- Conceptual Explanation ---
# This assignment explores **Sentiment Analysis** and basic **Emotion Analysis**, key tasks in Natural Language Processing (NLP) focused on understanding the emotional content of text.
#
# **Sentiment Analysis:** This is the process of determining the emotional tone or polarity of a piece of text. It typically classifies text as positive, negative, or neutral, but can also involve identifying more nuanced sentiments. Sentiment analysis is widely used in social media monitoring, customer feedback analysis, and market research.
#
# **Emotion Analysis:** This is a more granular task than sentiment analysis, aiming to identify specific emotions expressed in text, such as joy, sadness, anger, fear, surprise, or disgust. It often relies on lexicons or machine learning models trained on data labeled with specific emotions.
#
# The assignment presents several approaches to these tasks:
#
# 1. **Lexicon-based Sentiment Analysis:** This method uses a predefined list (lexicon) of words associated with positive and negative sentiments. Each word in the lexicon is often assigned a sentiment score. The sentiment of a text is determined by aggregating the scores of the words it contains. This approach is simple and interpretable but relies heavily on the quality and coverage of the lexicon and struggles with context, negation, and sarcasm.
#
# 2. **Lexicon-based Emotion Analysis:** Similar to lexicon-based sentiment analysis, but uses lexicons specifically curated for different emotions. The presence and frequency of words associated with a particular emotion contribute to an emotion score for the text. This provides a distribution of emotions rather than a single polarity. Like sentiment lexicons, emotion lexicons have limitations regarding context and nuances.
#
# 3. **Simple Frequency-based Sentiment Model (From Scratch Concept):** This is a basic machine learning approach where the model learns the frequency of words in a small set of labeled positive and negative training texts. During prediction, it counts the occurrences of words from the learned positive and negative word lists in the new text. The sentiment is predicted based on which category's words appear more frequently. This illustrates the fundamental idea of learning from data but is very simplistic and lacks robustness.
#
# 4. **Machine Learning Sentiment Model (using scikit-learn):** This approach uses a more standard machine learning pipeline.
#    - **`TfidfVectorizer`:** This is used for feature extraction. It converts the raw text into numerical feature vectors using the TF-IDF (Term Frequency-Inverse Document Frequency) method. TF-IDF reflects how important a word is to a document in a collection, penalizing words that are common across all documents (like "the" or "a") and giving higher scores to words that are more unique to a specific document.
#    - **`LogisticRegression`:** This is a linear model used for binary classification (in this case, positive vs. negative sentiment). It learns a decision boundary based on the TF-IDF features to classify new texts. It outputs a probability, which is then mapped to a sentiment score.
#    - The model is trained on labeled data (`train_texts`, `train_labels`) and then used to predict the sentiment score of new texts. This approach is generally more powerful than lexicon-based methods as it can learn more complex patterns from data.
#
# The assignment demonstrates the application of these different methods on sample texts, highlighting the differences in their outputs, the underlying mechanisms, and their respective strengths and weaknesses.

# --- Potential Viva Questions ---
# What is sentiment analysis, and what are its main applications?
# Explain the concept of lexicon-based sentiment analysis. What are its advantages and disadvantages?
# How does the simple frequency-based sentiment model work? How is it different from the lexicon-based approach?
# Explain the role of `TfidfVectorizer` in the machine learning sentiment model.
# Why is Logistic Regression a suitable model for binary sentiment classification?
# How is the sentiment score calculated in the ML model prediction? What does a score of +1, -1, or 0 mean?
# What are the challenges in performing accurate sentiment and emotion analysis?
# How could you improve the performance of these sentiment/emotion analysis methods? (e.g., using larger lexicons, more complex ML models, handling negation, sarcasm, etc.)
# Expected Response: Improvements could include using larger and more nuanced lexicons, incorporating techniques to handle negation (e.g., "not good" should be negative), addressing sarcasm and irony, using more advanced feature extraction methods (like word embeddings), employing more complex machine learning models (like deep learning networks), and training on larger and more diverse datasets.
