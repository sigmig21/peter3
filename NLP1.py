import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
df = pd.read_csv("Language Detection.csv")
print(df.head())
print(df['language'].value_counts())
X = df["text"]
y = df["language"]
le = LabelEncoder()
y = le.fit_transform(y)
data_list = []
for text in X:
    text = re.sub(r'[!@#$(),\n"%^*?\:;~`0-9]', ' ', text)
    text = re.sub(r'\[\]', ' ', text)
    text = text.lower()
    data_list.append(text)

cv = CountVectorizer()
X = cv.fit_transform(data_list).toarray()
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
model = MultinomialNB()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
ac = accuracy_score(y_test, y_pred)
cr = classification_report(y_test, y_pred, zero_division=0)
print(f"Accuracy: {ac}")
print(f"Classification Report:\n{cr}")

# Calculate confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

# --- Conceptual Explanation ---
# This assignment focuses on building a simple language detection model using the Multinomial Naive Bayes algorithm.
# The core idea is to train a classifier on a dataset of text samples labeled with their respective languages.
# The model learns the probability distribution of words (or character n-grams, though this assignment uses word counts via `CountVectorizer`) for each language.
# When a new, unseen text is provided, the model calculates the probability that the text belongs to each language based on the words it contains and predicts the language with the highest probability.
# The assignment involves several key steps in a typical machine learning workflow:
# 1. Data Loading: Reading the dataset containing text samples and their corresponding language labels.
# 2. Text Preprocessing: Cleaning the text data by removing unwanted characters (like punctuation and numbers) and converting text to lowercase to ensure consistency.
# 3. Feature Extraction: Transforming the raw text into numerical features that the machine learning model can understand. `CountVectorizer` is used here to create a "Bag-of-Words" representation, where each text is represented by the frequency of words it contains.
# 4. Data Splitting: Dividing the dataset into training and testing sets. The training set is used to train the model, and the testing set is used to evaluate its performance on unseen data.
# 5. Model Training: Training the Multinomial Naive Bayes model on the training data. Naive Bayes is a probabilistic classifier based on Bayes' theorem, assuming independence between features (words). Multinomial Naive Bayes is particularly suited for discrete features like word counts.
# 6. Prediction: Using the trained model to predict the language of the texts in the testing set.
# 7. Evaluation: Assessing the model's performance using metrics such as:
#    - Accuracy: The proportion of correctly predicted instances.
#    - Confusion Matrix: A table showing the counts of true positive, true negative, false positive, and false negative predictions, providing insight into where the model is making errors.
#    - Classification Report: Provides precision, recall, and F1-score for each language class, offering a more detailed view of performance than just accuracy.

# --- Potential Viva Questions ---
# What is the core principle behind Naive Bayes classification?
# Why is Multinomial Naive Bayes suitable for text classification tasks like language detection?
# Explain the role of `CountVectorizer` in this assignment. What does it do?
# What is a confusion matrix, and what do the values in the confusion matrix output tell you about the model's performance?
# What is the significance of the `zero_division=0` parameter in `classification_report`?
# How would you handle out-of-vocabulary words in this model?
# What are some limitations of using a simple Bag-of-Words model (`CountVectorizer`) for language detection?
# Expected Response: A simple Bag-of-Words model loses word order and context, treating a sentence as just a collection of words. This can be problematic for languages where word order is crucial for meaning. It also doesn't capture semantic relationships between words.
# How could you improve the accuracy of this language detection model? (e.g., using TF-IDF, n-grams, different models)
# Expected Response: Improvements could include using TF-IDF instead of raw counts to give more weight to important words, using character n-grams (sequences of characters) which are very effective for language detection and robust to misspellings, using a larger and more diverse dataset, or trying different machine learning models like Support Vector Machines (SVM) or deep learning models.
