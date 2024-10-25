# %%
import pathlib
import textwrap
import time

import google.generativeai as genai

from IPython.display import display
from IPython.display import Markdown
import pandas as pd

def to_markdown(text):
  text = text.replace('•', '  *')
  return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))
# %%
## API KEY
## 1
# genai.configure(api_key="AIzaSyDeg2dMzICveywF-fX60hixY12S8xhXvE0")
## 2
genai.configure(api_key="AIzaSyDfHimKVsXHUU9COYCtXKFvNVt3x9PkcfM")
# %%
model = genai.GenerativeModel("gemini-1.5-flash")
# %%
df = pd.read_excel('../dataset.xlsx')
df = df[['Notulen', 'Prioritas']]
df.head()
# %%
df['Prioritas'].value_counts()
# %%
paling_didahulukan = df[df['Prioritas'] == 3]
paling_didahulukan.to_excel('paling_didahulukan_ori.xlsx')
# %%
didahulukan = df[df['Prioritas'] == 2]
didahulukan.to_excel('didahulukan_ori.xlsx')
# %%
tidak_didahulukan = df[df['Prioritas'] == 1]
tidak_didahulukan.to_excel('tidak_didahulukan_ori.xlsx')
# %%
def parafrase(text):
  response = model.generate_content(f"Berikut adalah notulensi dari bimbingan skripsi suatu mahasiswa. Lakukanlah parafrase tanpa mengubah makna asli dari notulensi tersebut dan jangan memberikan output lebih dari satu, jika saya memberikan satu kalimat maka parafrase tetap menjadi satu kalimat dengan makna asli yang sama. Perhatikan keseluruhan kalimat, dan jangan ada yang bersifat kalimat tanya.\nNotulensi: \n\n'{text}'")
  return response.text

# %%
# PROSES PARAFRASE
## PALING DIDAHULUKAN (Prioritas = 3)
paling_didahulukan_parafrase = []
num = 0

for i in paling_didahulukan['Notulen']:
    success = False  # Flag untuk menandai apakah iterasi berhasil
    while not success:  # Ulangi terus jika error
        try: 
            nilai = parafrase(i)
            paling_didahulukan_parafrase.append(nilai)
            num += 1
            print(f"{num} BERHASIL!!")
            print(nilai)
            time.sleep(1.5)
            success = True  # Jika berhasil, keluar dari while loop
        except Exception as e:
            print(f"Terjadi kesalahan pada baris {num+1}: {e}")
            print("Mengulangi iterasi ini...")
            time.sleep(2)  # Tunggu beberapa detik sebelum mengulang
# %%
## DIDAHULUKAN (Prioritas = 2)
didahulukan_parafrase = []
num = 0

for i in didahulukan['Notulen']:
    success = False  # Flag untuk menandai apakah iterasi berhasil
    while not success:  # Ulangi terus jika error
        try:
            nilai = parafrase(i)
            didahulukan_parafrase.append(nilai)
            num += 1
            print(f"{num} BERHASIL!!")
            print(nilai)
            time.sleep(1.5)
            success = True  # Jika berhasil, keluar dari while loop
        except Exception as e:
            print(f"Terjadi kesalahan pada baris {num+1}: {e}")
            print("Mengulangi iterasi ini...")
            time.sleep(2)  # Tunggu beberapa detik sebelum mengulang

# %%
## TIDAK DIDAHULUKAN (Prioritas = 1)
tidak_didahulukan_parafrase = []
num = 0

for i in tidak_didahulukan['Notulen']:
    success = False
    while not success:
        try:
            nilai = parafrase(i)
            tidak_didahulukan_parafrase.append(nilai)
            num += 1
            print(f"{num} BERHASIL!!")
            print(nilai)
            time.sleep(1.5)
            success = True
        except Exception as e:
            print(f"{num+1} GAGAL!!: {e}")
            print("MENGULANG...")
            time.sleep(2)

# %%
# EXPORT LABEL 3 KE EXCEL
paling_didahulukan_resampled = pd.DataFrame({'Notulen': paling_didahulukan_parafrase})
paling_didahulukan_resampled.to_excel('paraphrase/paling_didahulukan_resampled.xlsx')

# %%
# EXPORT LABEL 2 KE EXCEL
didahulukan_resampled = pd.DataFrame({'Notulen': didahulukan_parafrase})
didahulukan_resampled.to_excel('paraphrase/didahulukan_resampled.xlsx')

# %%
# EXPORT LABEL 1 KE EXCEL
tidak_didahulukan_resampled = pd.DataFrame({'Notulen': tidak_didahulukan_parafrase})
tidak_didahulukan_resampled.to_excel('paraphrase/tidak_didahulukan_resampled.xlsx')


# %%
# IMPORT DR EXCEL
paling_didahulukan_resampled = pd.read_excel('paling_didahulukan_resampled.xlsx')
didahulukan_resampled = pd.read_excel('didahulukan_resampled.xlsx')
# %%
# CEK DATA
parafrase_data = pd.read_excel('paraphrase/resampled_final_v1.0.xlsx')


parafrase_count = parafrase_data['Prioritas'].value_counts()

print(f"{parafrase_count}")

# %%
# MENGHAPUS DUPLIKAT
parafrase_data = parafrase_data.drop_duplicates(subset=['Notulen'], keep='first')
parafrase_data.to_excel('paraphrase/resampled_final_no_duplicates.xlsx')
parafrase_data['Prioritas'].value_counts()
# %%
# Resampling data
df = pd.read_excel('paraphrase/resampled_final_no_duplicates.xlsx')
df['Prioritas'].value_counts()

df_3 = df[df['Prioritas'] == 3] # Sudah 509 data
df_2 = df[df['Prioritas'] == 2].sample(n = 537,  random_state= 42)
df_1 = df[df['Prioritas'] == 1].sample(n = 537, random_state= 42) 

# Menggabungkan kembali dataset yang sudah disamakan jumlahnya
df_balanced = pd.concat([df_1, df_2, df_3])

# %%
# Mengacak data
df = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
df.to_excel('paraphrase/Balanced_Dataset_v1.0.xlsx', index=False)

# %%
