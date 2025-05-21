import streamlit as st
import pandas as pd
import joblib

model = joblib.load('model/best_model_stress_predictor.joblib')
le_dict = joblib.load('model/label_encoders.joblib')
scaler = joblib.load('model/scaler.joblib')
imputer_num = joblib.load('model/imputer_num.joblib')
imputer_cat = joblib.load('model/imputer_cat.joblib')

st.title("Prediksi Tingkat Stres Karyawan")

# Form input user (skala 1-5 untuk fitur target)
rata_jam_kerja = st.number_input("Rata-rata Jam Kerja per Hari", min_value=0.0, max_value=24.0)
tekanan_pekerjaan = st.slider("Tekanan Pekerjaan", 1, 5)
dukungan_atasan = st.slider("Dukungan Atasan", 1, 5)
kebiasaan_tidur = st.slider("Kebiasaan Tidur (jam)", 0, 12)
kepuasan_kerja = st.slider("Kepuasan Kerja", 1, 5)
kepribadian_sosial = st.slider("Kepribadian Sosial", 1, 5)
lokasi_bekerja = st.selectbox("Lokasi Bekerja", ["Kantor", "Remote", "Hybrid"])
kebiasaan_olahraga = st.slider("Frekuensi Olahraga (per minggu)", 0, 7)
work_life_balance = st.slider("Work-Life Balance", 1, 5)
tinggal_dengan_keluarga = st.selectbox("Tinggal Bersama Keluarga", ["Ya", "Tidak"])
wilayah_kerja = st.selectbox("Wilayah Kerja", ["Jabodetabek", "Non-Jabodetabek"])

# Mapping kategorikal
lokasi_mapping = {"Kantor": 0, "Remote": 1, "Hybrid": 2}
keluarga_mapping = {"Ya": 1, "Tidak": 0}
wilayah_mapping = {"Jabodetabek": 0, "Non-Jabodetabek": 1}
def map_tidur(jam):
    if jam <= 3:
        return 1  # sangat buruk
    elif jam <= 5:
        return 2
    elif jam <= 7:
        return 3
    elif jam <= 9:
        return 4
    else:
        return 5  # sangat cukup

def map_olahraga(frek):
    if frek == 0:
        return 1  # tidak olahraga
    elif frek <= 1:
        return 2
    elif frek <= 3:
        return 3
    elif frek <= 5:
        return 4
    else:
        return 5  # olahraga rutin tinggi

# Pakai mapping saat ambil input
kebiasaan_tidur_mapped = map_tidur(kebiasaan_tidur)
kebiasaan_olahraga_mapped = map_olahraga(kebiasaan_olahraga)

# Buat dict input awal
input_dict = {
    'Rata_Jam_Kerja_Per_Hari': [rata_jam_kerja],
    'Tekanan_Pekerjaan': [tekanan_pekerjaan - 1],    # **kurangi 1**
    'Dukungan_Atasan': [dukungan_atasan - 1],        # **kurangi 1**
    'Kebiasaan_Tidur': [kebiasaan_tidur],
    'Kepuasan_Kerja': [kepuasan_kerja - 1],          # **kurangi 1**
    'Kepribadian_Sosial': [kepribadian_sosial - 1],  # **kurangi 1**
    'Lokasi_Bekerja': [lokasi_mapping[lokasi_bekerja]],
    'Kebiasaan_Olahraga': [kebiasaan_olahraga],
    'Work_Life_Balance': [work_life_balance - 1],     # **kurangi 1**
    'Tinggal_Bersama_Keluarga': [keluarga_mapping[tinggal_dengan_keluarga]],
    'Wilayah_Kerja': [wilayah_mapping[wilayah_kerja]]
}

if st.button("Prediksi Stres"):
    X_input = pd.DataFrame(input_dict)

    # Sesuaikan urutan kolom dengan pelatihan model (jika perlu)
    try:
        expected_columns = model.get_booster().feature_names
        X_input = X_input[expected_columns]
    except:
        pass  # Jika tidak ada method get_booster, abaikan

    # Lakukan prediksi (hasil 0-4)
    prediction_0_based = model.predict(X_input)[0]

    # Konversi kembali ke 1-5 sebelum tampilkan ke user
    prediction_1_based = prediction_0_based + 1

    st.success(f"Hasil Prediksi Tingkat Stres: {prediction_1_based}")
