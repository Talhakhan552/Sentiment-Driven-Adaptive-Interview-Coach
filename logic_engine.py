import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

class DualMetricAnalyzer:
    def __init__(self, csv_path):
        # Load a small sample of your 1.6 million rows for the midway demo speed
        # Sentiment140 columns: [target, id, date, flag, user, text]
        df = pd.read_csv(csv_path, encoding='latin-1', header=None).sample(10000)
        df.columns = ['sentiment', 'id', 'date', 'query', 'user', 'text']
        
        # 0 = negative (Nervous), 4 = positive (Confident)
        self.vectorizer = CountVectorizer(stop_words='english')
        train_features = self.vectorizer.fit_transform(df['text'])
        self.model = MultinomialNB()
        self.model.fit(train_features, df['sentiment'])

    def analyze_sentiment(self, text):
        feature = self.vectorizer.transform([text])
        prediction = self.model.predict(feature)[0]
        return "Nervous" if prediction == 0 else "Confident"

    def analyze_complexity(self, text):
        # IQ Assessment: Basic vs Advanced vocabulary [cite: 18]
        unique_words = len(set(text.lower().split()))
        return "Advanced" if unique_words > 12 else "Basic"