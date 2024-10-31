import streamlit as st
import pandas as pd
import pickle
import requests

## Preprocessing
# REQ
import string
import nltk
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
factory = StemmerFactory()
stemmer = factory.create_stemmer()
# Function
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

# BODY
title = st.title("Rekomendasi Urutan Prioritas")
table = '''
<table style="border:1px solid white; border-radius: 10px;">
    <thead>
    <tr>
    <th style="border:1px solid white; border-radius: 10px;"colspan=2>Prioritas</th>
    </tr>
    <tr>
    <th style="border:1px solid white; border-radius: 10px;">Label</th>
    <th style="border:1px solid white; border-radius: 10px;">Deskripsi</th>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td style="border:1px solid white; border-radius: 10px;" rowspan=2> <strong>Paling Didahulukan</strong> </td>
    <td style="border:1px solid white; border-radius: 10px;"> Tingkat Kesulitan : Tinggi - Sedang </td>
    </tr>
    <tr>
    <td style="border:1px solid white; border-radius: 10px;"> Pengaruh : Memiliki potensi untuk mengubah alur dari penelitian baik secara parsial atau bahkan secara keseluruhan. Perubahan diperlukan pada bagian bagian yang menentukan alur penelitian, seperti : penyesuaian topik/judul, perubahan rumusan masalah, hingga pergantian metode penelitian.
    </td>
    </tr>
    <tr>
    <td style="border:1px solid white; border-radius: 10px;" rowspan=2> <strong>Didahulukan</strong> </td>
    <td style="border:1px solid white; border-radius: 10px;">  Tingkat Kesulitan : Sedang </td>
    </tr>
    <tr>
    <td style="border:1px solid white; border-radius: 10px;">Pengaruh : Memiki sangat sedikit potensi untuk mengubah alur dari penelitian, namun dapat mengubah beberapa bagian dari peneltian seperti kesesuaian data, penjabaran penelitian, hingga perubahan penuh pada salah satu bab namun hanya untuk mesinkronkan dengan penelitian
    </td>
    </tr>
    <tr>
    <td style="border:1px solid white; border-radius: 10px;" rowspan=2> <strong>Tidak Perlu Didahulukan</strong> </td>
    <td style="border:1px solid white; border-radius: 10px;"> Tingkat Kesulitan : Sedang - Rendah </td>
    </tr>
    <tr>
    <td style="border:1px solid white; border-radius: 10px;">
    Pengaruh : Tidak memiliki pengaruh pada alur penelitian, hanya perubahan kecil seperti kejelasan penulisan, kesesuaian dengan pedoman, hingga penambahan sedikit elemen namun hanya untuk memperjelas penelitian.
    </td>
    </tr>
</table>

'''
st.html(table)

## SVM MODEL
# Load Model Dari Pickle
with open('model/model.pkl', 'rb') as f:
    model = pickle.load(f)

def translate_label(pred):
    if pred == 3:
        return 'Paling Didahulukan'
    elif pred == 2:
        return 'Didahulukan'
    elif pred == 1:
        return 'Tidak Perlu Didahulukan'
    
# Input Text streamlit yang dapat bertambah apabila sebuah button ditekan
notulensi_group = []
if 'count' not in st.session_state:
    st.session_state.count = 0

if st.button("Tambahkan notulensi"):
    st.session_state.count += 1

for i in range(st.session_state.count):
    notulensi = st.text_input(f"Masukkan notulensi {i+1}", key=f"notulensi_{i}")
    notulensi_group.append(notulensi)

if st.button("Input"):
    if notulensi_group:
        # Preprocess text
        clean_notulensi_group = [preprocess_text(text) for text in notulensi_group]
        prediction = model.predict(clean_notulensi_group)
        # Output notulensi dan prediksinnya masing masing dalam bentuk tabel
        st.write(pd.DataFrame({'Notulensi': notulensi_group, 'Prediksi': (translate_label(prediction) for prediction in prediction)}))

if st.button("Clear"):
    st.session_state.count = 0
    notulensi_group.clear()
    st.toast("Notulensi telah dihapus")

    

# SIDEBAR
link = "https://docs.google.com/forms/d/e/1FAIpQLSdm34qJkPTooN1Df0j45tMRxoIf8MLlUGfnM3bf1_8uI2gGoA/formResponse"
side_info = '''
## Bantu saya untuk mengevaluasi AI

> Informasi yang kamu berikan akan digunakan untuk kepentingan penelitian dan evaluasi AI. Mohon diperhatikan untuk tidak memberikan informasi yang sifatnya sangat sensitif seperti nomor telpon atau kata sandi!

'''

def check_rate_input(question_check):
    if question_check is None:
        st.error("Tolong berikan penilaian!")

def input_gform_btn(name, major, feedbacks):
    if any(feedback is None for feedback in feedbacks):
        st.error("Tolong berikan penilaian!")
    elif not name or not major:
        st.error("Tolong data diri anda!")
    else:
        value = {
            "entry.1431889514" : name,
            "entry.34748636" : major,
            "entry.1715492421" : feedbacks[0]+1,
            "entry.1639603815" : feedbacks[1]+1,
            "entry.1970527084" : feedbacks[2]+1
        }
        submit(link, value)

def submit(url,data):
    try:
        requests.post(url, data=data)
        st.success("Terima kasih telah memberikan feedback!")
    except:
        st.error("Gagal mengirim data!")
## UNTUK GOOGLE FORM

with st.sidebar.form("gform_input", enter_to_submit=False):
    sentiment_mapping = ["one", "two", "three", "four", "five"]
    st.markdown(side_info)
    # Nama Mahasiswa
    student_name = st.text_input("Nama Kamu")
    # Jurusan Mahasiswa
    student_major = st.text_input("Jurusan Kamu")

    # Define questions
    questions = [
        "Bagaimana menurut kamu tentang hasil rekomendasi yang diberikan?",
        "Bagaimana menurut kamu tentang kejelasan hasil rekomendasi yang diberikan?",
        "Bagaimana menurut kamu tentang kemudahan penggunaan sistem rekomendasi?"
    ]

    # Create feedback widgets for each question
    feedback_responses = []
    for i, question in enumerate(questions, start=1):
        st.write(question)
        feedback = st.feedback('stars', key=f"question_{i}")
        feedback_responses.append(feedback)

    # Submit button
    submit_btn = st.form_submit_button(label="Kirim")
    if submit_btn:
        input_gform_btn(student_name, student_major, feedback_responses)
