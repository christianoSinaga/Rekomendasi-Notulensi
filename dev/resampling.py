# %%
# PREREQUISITES
import pandas as pd
import random
from collections import Counter
# %%
# Req
df = pd.read_excel('paraphrase/Balanced_Dataset_v1.0.xlsx')
df['Prioritas'].value_counts()
# %%
import string
import nltk
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from collections import Counter
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from scipy.stats import uniform
import numpy as np
# %%
factory = StemmerFactory()
stemmer = factory.create_stemmer()
# %%
def preprocess_text(text):
    # Lowercase folding
    text = text.lower()
    # Hilangkan tanda baca
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenisasi
    tokens = nltk.word_tokenize(text)
    # Stopwords removal dan stemming
    # tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in custom_stopwords]
    tokens = [stemmer.stem(word) for word in tokens]
    # Gabungkan kembali token menjadi teks
    return ' '.join(tokens)
# %%
df['clean_notulensi'] = df['Notulen'].apply(preprocess_text)
test_split = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# %%
## Mencari SVM Parameter Finder
# Pipeline untuk SVM
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', SVC())
])

# Parameter yang akan diuji
param_dist = {
    'clf__C': uniform(0.1, 10),            # Menggunakan distribusi uniform untuk mencoba nilai C
    'clf__kernel': ['linear', 'rbf'],       # Jenis kernel
    'clf__gamma': ['scale', 'auto'],        # Parameter gamma
    'tfidf__max_df': [0.75, 1.0],
    'tfidf__ngram_range': [(1, 1), (1, 2)]
}


# Inisialisasi RandomizedSearchCV
random_search = RandomizedSearchCV(pipeline, param_dist, n_iter=50, cv=5, n_jobs=-1, verbose=2, random_state=42)

for num in test_split:
    X_train, X_test, y_train, y_test = train_test_split(df['clean_notulensi'], df['Prioritas'], test_size=num, random_state=42)

    # Melakukan pencarian parameter terbaik dengan random search
    random_search.fit(X_train, y_train)

    # Menampilkan hasil parameter terbaik
    print("Best Parameters:", random_search.best_params_)
    # print("Best Cross-validation Score:", random_search.best_score_)

    # Evaluasi pada test set
    y_pred = random_search.best_estimator_.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"CLEAN Set Accuracy {num}: {accuracy} ")

# %%
final_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_df=0.75, ngram_range=(1, 1))),
    ('clf', SVC(C=0.68, gamma='scale',  kernel='rbf', probability=True))

])
# %%
model =  final_pipeline.fit(df['clean_notulensi'], df['Prioritas'])
X_tfidf = final_pipeline.named_steps['tfidf'].transform(df['clean_notulensi'])
probabilities = model.named_steps['clf'].predict_proba(X_tfidf)

# Menyimpan probabilitas ke dalam dataframe
df['probabilities'] = probabilities.max(axis=1)  # Ambil probabilitas tertinggi

# Tentukan threshold untuk outlier (misalkan kita anggap 0.8 sebagai threshold)
threshold = 0.8
# 5 Skenario

df['outlier'] = np.where(df['probabilities'] < threshold, 1, 0)  # 1 = outlier, 0 = bukan outlier

# Menampilkan outlier
outliers = df[df['outlier'] == 1]
print(f"Outliers: {outliers}")
# %%
df_cleaned = df[df['outlier'] == 0]

df_cleaned['Prioritas'].value_counts()
# %%
final_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_df=0.75, ngram_range=(1, 1))),
    ('clf', SVC(C=3.68, gamma='scale', kernel='rbf', probability=True))

])

for num in test_split:
    X_train, X_test, y_train, y_test = train_test_split(df_cleaned['clean_notulensi'], df_cleaned['Prioritas'], test_size=num, random_state=42)
    model = final_pipeline.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"CLEAN Set Accuracy {num}: {accuracy_score(y_test, y_pred)}, ")

# %%
df_cleaned.to_excel('paraphrase/no_outlier_v1.0.xlsx')
# %%