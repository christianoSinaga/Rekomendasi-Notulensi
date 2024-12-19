import streamlit as st
import pandas as pd
import pickle
import requests

## Preprocessing
# REQ
import string
import nltk
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from dev.normalization_dict import  normalization_dict


## ChatBot REQ (OPTIONAL)
import google.generativeai as genai

from IPython.display import display
from IPython.display import Markdown
import textwrap
import time

nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('punkt')

factory = StemmerFactory()
stemmer = factory.create_stemmer()
def normalize_text(tokens):
    return [normalization_dict.get(word, word) for word in tokens]
# Function
def preprocess_text(text):
    # Lowercase folding
    text = text.lower()
    # Hilangkan tanda baca
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenisasi
    tokens = nltk.word_tokenize(text)
    # Normalisasi
    tokens = normalize_text(tokens)
    # Stemming
    tokens = [stemmer.stem(word) for word in tokens]
    # Gabungkan kembali token menjadi teks
    return ' '.join(tokens)

# BODY
title = st.title("Rekomendasi Urutan Prioritas")
desc = '''
Merupakan aplikasi yang dapat membantu anda untuk menentukan urutan penyelesaian dari notulensi bimbingan yang sudah kamu lakukan.
## Cara Menggunakan Sistem Rekomendasi:
* **Mengisi Notulensi**: Dari bimbingan yang baru saja dilakukan, mahasiswa dapat mengisi notulensinya ke dalam input `notulensi` di bawah ini.
* **Bentuk Notulensi**: Dalam satu input `notulensi`, mahasiswa hanya dapat mengisi satu notulensi
* **Menambah Notulensi**: Jika anda ingin menambahkan notulensi, silahkan memilih tombol `Tambah Notulensi` untuk menambahkan notulensi
* **Memberikan Rekomendasi**: Pilih tombol `Tentukan Prioritas` untuk memberikan notulensi kepada model agar dapat memberikan rekomendasi urutan prioritas

> Setelah anda mencoba menggunakan  sistem rekomendasi ini, silahkan memberikan umpan balik kepada peneliti yang terdapat pada sidebar. Apabila anda menggunakan ponsel, anda bisa memilih tombol yang ada di pojok kiri atas untuk membuka sidebar. Umpan balik sangat diperlukan oleh peneliti untuk kelanjutan penelitiannya.
'''
# st.html(table)
st.markdown(desc, unsafe_allow_html=True)

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
    st.session_state.count = 1

if st.button("Tambahkan notulensi"):
    st.session_state.count += 1

for i in range(st.session_state.count):
    notulensi = st.text_input(f"Masukkan notulensi ke-{i+1}", key=f"notulensi_{i}")
    notulensi_group.append(notulensi)

output_model = pd.DataFrame()
col_1, col_2 = st.columns(2)
# Inisialisasi Variabel Dalam Session State
if 'output_model' not in st.session_state:
    st.session_state.output_model = pd.DataFrame({'Notulensi': []})

if "chats" not in st.session_state:
    st.session_state.chats = []

if "done_initial" not in st.session_state:
    st.session_state.done_initial = False

if col_1.button("Tentukan Prioritas", use_container_width=True):
    if notulensi_group:
        
        # Preprocess text
        clean_notulensi_group = [preprocess_text(text) for text in notulensi_group]
        prediction = model.predict(clean_notulensi_group)

        # Buat DataFrame untuk output
        df_output = pd.DataFrame({
            'Notulensi': notulensi_group,
            'Prediksi': [translate_label(pred) for pred in prediction]
        })
        

        # Pastikan DataFrame tidak kosong
        if not df_output.empty:
            # Kelompokkan berdasarkan label
            grouped_output = df_output.groupby('Prediksi', sort=True)

            # Simpan ke session state
            st.session_state.output_model = df_output
            st.session_state.done_initial = False

            # Tampilkan tabel terkelompok
            for label, group in grouped_output:
                st.write(f"### Prioritas: {label}")
                st.table(group)
        else:
            st.warning("Tidak ada data untuk ditampilkan.")

if not st.session_state.output_model['Notulensi'].empty:
    st.dataframe(st.session_state.output_model, use_container_width=True, hide_index=True)

if col_2.button("Reset", use_container_width=True):
    st.session_state.count = 0
    notulensi_group.clear()
    st.toast("Notulensi telah dihapus")
    st.rerun()



# ## CHATBOT
# # Fungsi stream untuk variabel berisi tipe data String
# def stream_error_msg(response):
#     for word in response.split(" "):
#         yield word + " "
#         time.sleep(0.02)

# def output_propmt(output_model):
#     final_output = ""
#     for i, row in output_model.iterrows():
#         final_output += f"- **Notulensi ke {i+1}**: '{row['Notulensi']}' | **Prioritas**: {row['Prediksi']}\n"
#     return final_output

# st.divider()
# # st.write(output_propmt(output_model))
# initial_msg = "Gunakan fitur rekomendasi urutan prioritas di atas terlebih dahulu sebelum melanjutkan ke fitur chatbot. Di bawah ini adalah fitur chatbot, yang merupakan fitur tambahan. Pengguna diharapkan untuk memberikan penilaian kepada fitur rekomendasi di atas saja."

# if (not st.session_state.output_model['Notulensi'].empty):
#     initial_prompt = f"""
#     Kamu adalah seorang asisten dari mahasiswa yang aku tugaskan untuk membantu mahasiswa tingkat akhir dalam menentukan urutan pengerjaan dari notulensi bimbingannya. Sebelumnya kamu sudah memberikan label tingkat prioritas dari masing masing notulensi sebagai berikut : 
    
#     {output_propmt(st.session_state.output_model)}

#     Kamu memberikan label tingkat prioritas dengan menggunakan parameter berikut:
#     | **Label**               | **Tingkat Kesulitan**           | **Pengaruh** |
#     |-------------------------|----------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
#     | **Paling Didahulukan**  | Tinggi - Sedang                 | Memiliki potensi mengubah alur penelitian baik sebagian atau seluruhnya. Dapat memerlukan perubahan besar seperti penyesuaian topik/judul, perubahan rumusan masalah, atau pergantian metode penelitian. |
#     | **Didahulukan**         | Sedang                          | Berpotensi mengubah bagian tertentu, meskipun tidak terlalu memengaruhi keseluruhan alur penelitian. Bisa melibatkan kesesuaian data, penjabaran penelitian, atau perubahan pada satu bab agar sesuai dengan penelitian.  |
#     | **Tidak Perlu Didahulukan** | Sedang - Rendah            | Tidak memengaruhi alur penelitian; hanya perubahan kecil seperti kejelasan penulisan, kesesuaian pedoman, atau penambahan elemen kecil untuk memperjelas penelitian. |
    
#     Selanjutnya kamu akan berkomunikasi dengan mahasiwa yang memberikan notulensi yang sudah kamu berikan label tingkat prioritas di atas. Berikan alasan dan penjelasan kamu lebih lanjut kepada mahasiswa pengguna atas penentuan prioritas yang sudah kamu berikan. Gunakan bahasa yang santai dan mudah dipahami oleh mahasiswa. Kamu juga dapat memberikan saran dan rekomendasi kepada mahasiswa pengguna untuk mempercepat penyelesaian tugas akhirnya. 
#     """
#     # initial_prompt = f" {output_propmt(st.session_state.output_model)}"


# ## UNCOMMENT TO ACTIVATE CHATBOT
# stream_initial_msg = st.write(initial_msg)


# ## CHATBOT (OPTIONAL)
# def to_markdown(text):
#   text = text.replace('•', '  *')
#   return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))


# # Set API Key and Model
# genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
# if "genai_model" not in st.session_state:
#     st.session_state["genai_model"] = genai.GenerativeModel('gemini-1.5-flash')
# model = st.session_state["genai_model"]




# # Tampilkan pesan sebelumnya
# for chat in st.session_state.chats:
#     # mengabaikan initial prompt
#     with st.chat_message(chat["role"]):
#         st.markdown(chat["text"])

# # Fungsi untuk mengonversi sesi obrolan ke format chat.history
# # Parameter initial_prompt nullable
# def convert_to_chat_history_format(chats, initial_prompt=None):
#     history = []
#     if initial_prompt:
#         history.append({"parts": [{"text": initial_prompt}], "role": "user"})
#     for chat in chats:
#         history.append({
#             "parts": [{"text": chat["text"]}],
#             "role": chat["role"]
#         })
#     return history


# # Fungsi streaming yang dimodifikasi untuk digunakan dengan `st.write_stream`
# def stream_gem_ai(response):
#     for chunk in response:
#         yield chunk.text  # Menggunakan `yield` untuk membuatnya menjadi generator

# if not st.session_state.output_model['Notulensi'].empty or st.session_state.done_initial:
#     chat_history = convert_to_chat_history_format(st.session_state.chats, initial_prompt)
#     chat_ai = model.start_chat(history=chat_history) 
#     # st.session_state.done_initial = True
# if not st.session_state.output_model['Notulensi'].empty and not st.session_state.done_initial:
#     with st.chat_message("model"):
#         try:
#             response_ai = chat_ai.send_message(initial_prompt, stream=True)  # Pastikan `send_message` mendukung streaming
#             response = st.write_stream(stream_gem_ai(response_ai))
#             st.session_state.chats.append({
#                 "role": "model",
#                 "text": response
#             })
#             st.session_state.done_initial = True
#         except Exception as e:
#             error_msg =  f"Terjadi kesalahan: {str(e)}"
#             response = st.write_stream(stream_error_msg(error_msg))
#             st.session_state.chats.append({
#                 "role": "model",
#                 "text": response
#             })

# if st.session_state.done_initial:
#     # Menerima input pengguna
#     if prompt := st.chat_input("Chat with Gemini AI"):
#         st.session_state.chats.append({
#             "role": "user",
#             "text": prompt
#         })
#         with st.chat_message("user"):
#             st.markdown(prompt)

#         # Siapkan histori dalam format yang diinginkan

#         with st.chat_message("model"):
#             try: 
#                 response_ai = chat_ai.send_message(prompt, stream=True)  # Pastikan `send_message` mendukung streaming
#                 # Menggunakan generator dengan `st.write_stream`
#                 response = st.write_stream(stream_gem_ai(response_ai))
#                 st.session_state.chats.append({
#                     "role": "model",
#                     "text": response
#                 })
#             except Exception as e:
#                 error_msg =  f"Terjadi kesalahan: {str(e)}"
#                 response = st.write_stream(stream_error_msg(error_msg))
#                 st.session_state.chats.append({
#                     "role": "model",
#                     "text": response
#                 })
            








# SIDEBAR
# link = "https://docs.google.com/forms/d/e/1FAIpQLSdm34qJkPTooN1Df0j45tMRxoIf8MLlUGfnM3bf1_8uI2gGoA/formResponse"
link = "https://docs.google.com/forms/d/e/1FAIpQLSdm34qJkPTooN1Df0j45tMRxoIf8MLlUGfnM3bf1_8uI2gGoA/formResponse"
side_info = '''
## Bantu saya untuk mengevaluasi AI

'''

def answer_mapping(answer):
    if answer == "Sangat Tidak Setuju":
        return 1
    elif answer == "Tidak Setuju":
        return 2
    elif answer == "Netral":
        return 3
    elif answer == "Setuju":
        return 4
    elif answer == "Sangat Setuju":
        return 5

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
            "entry.1715492421" : answer_mapping(feedbacks[0]),
            "entry.1639603815" : answer_mapping(feedbacks[1]),
            "entry.1970527084" : answer_mapping(feedbacks[2]),
            "entry.1160592644" : answer_mapping(feedbacks[3]),
            "entry.1819287220" : answer_mapping(feedbacks[4]),
            "entry.1775793053" : answer_mapping(feedbacks[5]),
            "entry.1443730601" : answer_mapping(feedbacks[6]),
            "entry.1516563691" : answer_mapping(feedbacks[7]),
            "entry.1365320732" : answer_mapping(feedbacks[8]),
            "entry.727884342" : answer_mapping(feedbacks[9])
        }
        submit(link, value)

def submit(url,data):
    try:
        requests.post(url, data=data)
        st.success("Terima kasih telah memberikan feedback!")
        # st.success(data)
    except:
        st.error("Gagal mengirim data!")
## UNTUK GOOGLE FORM

with st.sidebar.form("gform_input", enter_to_submit=False):
    sentiment_mapping = ["Sangat Tidak Setuju", "Tidak Setuju", "Netral", "Setuju", "Sangat Setuju"]
    st.markdown(side_info)
    # Nama Mahasiswa
    student_name = st.text_input("Nama Kamu")
    # Jurusan Mahasiswa
    student_major = st.text_input("Asal Universitas")

    # Define questions
    questions = [
        "Saya puas terhadap hasil rekomendasi yang diberikan oleh sistem",
        "Informasi prioritas yang diberikan oleh sistem jelas",
        "Sistem ini mudah digunakan untuk mengelola notulensi",
        "Rekomendasi prioritas yang diberikan oleh sistem relevan dengan notulensi yang saya input",
        "Saya merasa bahwa rekomendasi yang diberikan oleh sistem sering membantu menyelesaikan masalah dengan lebih cepat",
        "Saya puas terhadap tampilan antarmuka dari sistem ini",
        "Saya percaya terhadap hasil rekomendasi prioritas yang diberikan oleh sistem",
        "Saya merasa bahwa sistem ini dapat digunakan secara mandiri tanpa panduan tambahan",
        "Saya merasa sistem ini berpotensi untuk membantu mahasiswa lainnya dalam mengelola notulensi bimbingan",
        "Saya akan merekomendasikan sistem ini kepada mahasiswa lainnya"
    ]

    # Create feedback widgets for each question
    feedback_responses = []
    for i, question in enumerate(questions, start=1):
        feedback = st.select_slider(label=question, options=sentiment_mapping, key=f"question_{i}")
        feedback_responses.append(feedback)

    # Submit button
    submit_btn = st.form_submit_button(label="Kirim")
    if submit_btn:
        input_gform_btn(student_name, student_major, feedback_responses)
