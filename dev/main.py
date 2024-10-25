# %%
## PERSIAPAN DATASET
import pandas as pd
import random
from collections import Counter
from sklearn.utils import resample

# Baca File Excel
old_dataset = '../dataset.xlsx'
old_df = pd.read_excel(old_dataset)

parafrase = 'paraphrase/no_outlier_v1.0.xlsx'
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

# nltk.download('wordnet')
# nltk.download('stopwords')
# nltk.download('punkt_tab')
# nltk.download('punkt')


# %%
## PREPROCESSING PROCESS
# Inisialisasi Stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Fungsi untuk membersihkan dan memproses teks 
def preprocess_text(text):
    # Lowercase folding
    text = text.lower()
    # Hilangkan tanda baca
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenisasi
    tokens = nltk.word_tokenize(text)
    # Stopwords removal dan stemming
    tokens = [stemmer.stem(word) for word in tokens]
    # Gabungkan kembali token menjadi teks
    return ' '.join(tokens)

# %%
# Implementasi ke dataset
df_parafrase['clean_notulensi'] = df_parafrase['Notulen'].apply(preprocess_text)
# %%
## TF-IDF 
vectorizer = TfidfVectorizer(ngram_range=(1,2), max_df=0.75, min_df=5)
vectorizer2 = TfidfVectorizer()
X2 = vectorizer2.fit_transform(old_df['Notulen'])
X = vectorizer.fit_transform(df_parafrase['clean_notulensi'])
y = df_parafrase['Prioritas']
y2 = old_df['Prioritas']

# %%
randomSplit = random.randrange(1,50)
test_split = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
rand_test = random.choice(test_split)

# %%
## Menyimpan Model Dengan Parameter Terbaik
final_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_df=0.75, ngram_range=(1, 1))),
    ('clf', SVC(C=3.68, gamma='scale',  kernel='rbf', probability=True))

])
# %%
## SVM Dengan Data Bersih
for num in test_split:
    X_train, X_test, y_train, y_test = train_test_split(df_parafrase['clean_notulensi'], df_parafrase['Prioritas'], test_size=num, random_state=42)
    model = final_pipeline.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"CLEAN Set Accuracy {num}: {accuracy_score(y_test, y_pred)}")
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
    ('clf', SVC(C=3.68, gamma='scale',  kernel='rbf'))

])

model = final_pipeline.fit(df_parafrase['clean_notulensi'], df_parafrase['Prioritas'])
data_uji = pd.read_excel('../dataset.xlsx', sheet_name='uji')
data_uji = data_uji[['data_baru', 'prioritas']]
data_uji['data_baru_bersih'] = data_uji['data_baru'].apply(preprocess_text)


data_uji['pred'] = model.predict(data_uji['data_baru_bersih'])


print(accuracy_score(data_uji['prioritas'], data_uji['pred']))
data_uji

# %% 
# MENYIMPAN DATA BERSIH
# df_parafrase.to_excel('paraphrase/no_outlier.xlsx')
# %% 
# Ensemble Test
X_train_tfidf = final_pipeline.named_steps['tfidf'].fit_transform(df_parafrase['clean_notulensi'])
X_test_tfidf = final_pipeline.named_steps['tfidf'].transform(data_uji['data_baru_bersih'])
y_train = df_parafrase['Prioritas']
y_test = data_uji['prioritas']
# %%
## Bagging
from sklearn.ensemble import BaggingClassifier

svm_bagging = BaggingClassifier(final_pipeline.named_steps['clf'], n_estimators=10, random_state=42)


svm_bagging.fit(X_train_tfidf, y_train)

y_pred = svm_bagging.predict(X_test_tfidf)
print(accuracy_score(y_test, y_pred))

# %%
## Bosting
from sklearn.ensemble import AdaBoostClassifier
adaboost_svm = AdaBoostClassifier(final_pipeline.named_steps['clf'], n_estimators=50, random_state=42)

adaboost_svm.fit(X_train_tfidf, y_train)
y_pred = adaboost_svm.predict(X_test_tfidf)
print(f"Akurasi AdaBoost SVM: {accuracy_score(y_test, y_pred)}")

# %%
## Stacking
from sklearn.ensemble import StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

# Base learners
estimators = [
    ('svm', final_pipeline.named_steps['clf']),
    ('nb', mnb_pipeline.named_steps['clf']),
    ('tree', DecisionTreeClassifier())
]


stacking_clf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())

stacking_clf.fit(X_train_tfidf, y_train)

y_pred = stacking_clf.predict(X_test_tfidf)
print(f"Akurasi Stacking: {accuracy_score(y_test, y_pred)}")

# %%
scores = cross_val_score(model, df_parafrase['clean_notulensi'], df_parafrase['Prioritas'], cv=10)  # cv=5 untuk 5-fold cross-validation
print(f"Cross-validation scores: {scores}")
print(f"Mean accuracy: {scores.mean()}")  # Rata-rata akurasi dari setiap fold
# %%
