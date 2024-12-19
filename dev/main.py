# %%
## PERSIAPAN DATASET
import pandas as pd
import random
from collections import Counter
from sklearn.utils import resample

# Baca File Excel
old_dataset = '../dataset.xlsx'
old_df = pd.read_excel(old_dataset)

parafrase = 'paraphrase/no_outlier_v2.0.xlsx'
df_parafrase = pd.read_excel(parafrase)

print(f"Dataset parafrase \n{df_parafrase.head()} \n\n{Counter(df_parafrase['Prioritas'])}")
print(df_parafrase['Prioritas'].value_counts())

# %%
## PREPROCESSING REQUIREMENT
import string
import nltk
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
from collections import Counter
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from scipy.stats import uniform
from sklearn.svm import LinearSVC
import numpy as np
from normalization_dict import normalization_dict

# nltk.download('wordnet')
# nltk.download('stopwords')
# nltk.download('punkt_tab')
# nltk.download('punkt')


# %%
## PREPROCESSING PROCESS
# Inisialisasi Stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

def normalize_text(tokens):
    return [normalization_dict.get(word, word) for word in tokens]
# %%
# Fungsi untuk membersihkan dan memproses teks 
def preprocess_text(text):
    # Lowercase folding
    text = text.lower()
    # Hilangkan tanda baca
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenisasi
    tokens = nltk.word_tokenize(text)
    # Normalisasi
    tokens = normalize_text(tokens)
    # Stopwords removal dan stemming
    tokens = [stemmer.stem(word) for word in tokens]
    # Gabungkan kembali token menjadi teks
    return ' '.join(tokens)

# %%
# Implementasi ke dataset
df_parafrase['clean_notulensi'] = df_parafrase['Notulen'].apply(preprocess_text)
# %%
randomSplit = random.randrange(1,50)
test_split = [0.1, 0.2, 0.3, 0.4, 0.5]
rand_test = random.choice(test_split)

# %%
## Menyimpan Model Dengan Parameter Terbaik
final_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_df=0.50, ngram_range=(1, 1))),
    ('clf', SVC(C=3.68, gamma='scale',  kernel='rbf', probability=True))
])


# %%
## SVM Dengan Data Bersih
x = df_parafrase['clean_notulensi']
y = df_parafrase['Prioritas']
for num in test_split:
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=num, random_state=42)
    model = final_pipeline.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"Model Accuracy {num}: {accuracy_score(y_test, y_pred)}")

# %%
## Ekstrak Fitur
## Ubah pipeline dengan LinearSVC untuk mendapatkan bobot fitur
linear_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_df=0.70, ngram_range=(1, 1))),
    ('clf', LinearSVC(C=3.68))
])

# Fit Linear SVM pada data yang sama
linear_pipeline.fit(df_parafrase['clean_notulensi'], df_parafrase['Prioritas'])

# Mendapatkan fitur dari TF-IDF dan koefisien dari LinearSVC
feature_names = linear_pipeline.named_steps['tfidf'].get_feature_names_out()
coef = linear_pipeline.named_steps['clf'].coef_

# Menyusun bobot fitur dalam DataFrame untuk interpretasi
# Setiap kolom mewakili prioritas yang diprediksi oleh model
coef_df = pd.DataFrame(coef, columns=feature_names, index=["Prioritas 1", "Prioritas 2", "Prioritas 3"])

# Mendapatkan kata-kata dengan bobot tertinggi untuk setiap prioritas
top_features = {}
for label in coef_df.index:
    top_features[label] = coef_df.loc[label].nlargest(25)

# Tampilkan hasil
for label, features in top_features.items():
    print(f"Top words for {label}:")
    print(features)
    print("\n")

# %%
## MNB Dengan Data Bersih
mnb_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_df=0.75, ngram_range=(1, 1))),
    ('clf', MultinomialNB(alpha=1))
])
for num in test_split:
    X_train, X_test, y_train, y_test = train_test_split(df_parafrase['clean_notulensi'], df_parafrase['Prioritas'], test_size=num, random_state=42)
    model = mnb_pipeline.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"CLEAN Set Accuracy {num}: {accuracy_score(y_test, y_pred)}")
# %%
## Uji SVM dengan data baru
best_params = {'clf__C': 3.68, 'clf__gamma': 'scale', 'clf__kernel': 'rbf', 'tfidf__max_df': 0.75, 'tfidf__ngram_range': (1, 1)}
final_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_df=0.75, ngram_range=(1, 1))),
    ('clf', SVC(C=3.68, gamma='scale',  kernel='rbf', probability=True))

])
linear_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_df=0.70, ngram_range=(1, 1))),
    ('clf', LinearSVC(C=3.68))
])

model = linear_pipeline.fit(df_parafrase['clean_notulensi'], df_parafrase['Prioritas'])
data_uji = pd.read_excel('../dataset.xlsx', sheet_name='uji')
data_uji = data_uji[['data_baru', 'prioritas']]
data_uji['data_baru_bersih'] = data_uji['data_baru'].apply(preprocess_text)

# Prediksi probabilitas untuk setiap kelas
# probabilities = model.predict_proba(data_uji['data_baru_bersih'])

# Tambahkan hasil prediksi dan probabilitas ke DataFrame
data_uji['pred'] = model.predict(data_uji['data_baru_bersih'])
# Tampilkan hasil
print(f"{data_uji[['data_baru', 'prioritas', 'pred']]}\n")

print("Accuracy:", accuracy_score(data_uji['prioritas'], data_uji['pred']))
# %%
scores = cross_val_score(model, df_parafrase['clean_notulensi'], df_parafrase['Prioritas'], cv=10)  # cv=5 untuk 5-fold cross-validation
print(f"Cross-validation scores: {scores}")
print(f"Mean accuracy: {scores.mean()}")  # Rata-rata akurasi dari setiap fold
# %%
## MENYIMPAN MODEL KE PICKLE
import pickle
# %%
# Simpan model ke file pickle
with open('web/model/model.pkl', 'wb') as file:
    pickle.dump(model, file)

# Simpan preprocess ke pickle
with open('web/model/preprocess.pkl', 'wb') as file:
    pickle.dump(preprocess_text, file)


# %%
try:
    with open('web/model/model.pkl', 'rb') as file:
        model_pkl = pickle.load(file)

    with open('web/model/preprocess.pkl', 'rb') as file:
        preprocess_pkl = pickle.load(file)

except EOFError:
    print("Error: File pickle rusak atau tidak lengkap.")


# %%
data_uji = pd.read_excel('../dataset.xlsx', sheet_name='uji')
data_uji = data_uji[['data_baru', 'prioritas']]
data_uji['data_baru_bersih']=data_uji['data_baru'].apply(preprocess_pkl)
data_uji['pred'] = model_pkl.predict(data_uji['data_baru_bersih'])
# %%
print(f"{data_uji[['data_baru', 'prioritas', 'pred']]}\n")
# %%
