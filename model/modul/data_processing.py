import pandas as pd
import re
from textblob import TextBlob

def clean_text(text):
    \"\"\"
    Очистка текста от специальных символов и приведение к нижнему регистру.
    \"\"\"
    return re.sub(r'[^a-zA-Z0-9\\s]', '', text.lower())

def generate_features(data):
    \"\"\"
    Генерация дополнительных признаков из текста.
    \"\"\"
    data['avg_word_length'] = data['cleaned_text'].apply(lambda x: 
        np.mean([len(word) for word in x.split()]) if len(x.split()) > 0 else 0)
    data['uppercase_ratio'] = data['cleaned_text'].apply(lambda x: 
        sum(1 for char in x if char.isupper()) / len(x) if len(x) > 0 else 0)
    data['exclamation_count'] = data['cleaned_text'].apply(lambda x: x.count('!'))
    data['question_mark_count'] = data['cleaned_text'].apply(lambda x: x.count('?'))
    data['sentiment'] = data['cleaned_text'].apply(lambda x: TextBlob(x).sentiment.polarity)
    return data

def load_and_prepare_data(filepath):
    \"\"\"
    Загрузка данных, очистка текста и генерация признаков.
    \"\"\"
    data = pd.read_json(filepath, lines=True)
    data['cleaned_text'] = data['headline'].apply(clean_text)
    data = generate_features(data)
    return data