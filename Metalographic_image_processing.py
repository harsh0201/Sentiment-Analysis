import pandas as pd

data_train = pd.read_csv("train.csv")

column = ['id','text']
data_train = data_train.drop(columns=column)

your_tweet_dataframe = data_train

your_tweet_dataframe=your_tweet_dataframe.dropna()

your_tweet_dataframe['sentiment'].unique()

import nltk
import random
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report

nltk.download('stopwords')
nltk.download('punkt')

stop_words = set(stopwords.words("english"))

def preprocess_text(text):
    words = word_tokenize(text)
    words = [word.lower() for word in words if word.isalpha() and word.lower() not in stop_words]
    return ' '.join(words)

your_tweet_dataframe['tweet'] = your_tweet_dataframe['tweet'].apply(preprocess_text)

X = your_tweet_dataframe['tweet']
y = your_tweet_dataframe['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train_tfidf, y_train)


y_pred = svm_classifier.predict(X_test_tfidf)
print(classification_report(y_test, y_pred))

new_tweet = "I love this product! It's amazing."
new_tweet_preprocessed = preprocess_text(new_tweet)
new_tweet_tfidf = tfidf_vectorizer.transform([new_tweet_preprocessed])
sentiment = svm_classifier.predict(new_tweet_tfidf)
print(f"Sentiment: {sentiment[0]}")


your_tweet_dataframe['Target'].unique()

import joblib 

joblib.dump(svm_classifier,'nlp_project.joblib')

model = joblib.load('/content/nlp_project.joblib')