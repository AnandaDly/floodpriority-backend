from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import requests
import os
from datetime import datetime, timedelta

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    model = joblib.load('model_banjir_medan.pkl')
    print("Model berhasil dimuat!")
except Exception as e:
    print(f"Error memuat model: {e}")

LOG_FILE = "log_cuaca_medan.csv"

def init_log():
    
    if not os.path.exists(LOG_FILE):
        today = datetime.now()
        data = {
            "Tanggal": [(today - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(3, 0, -1)],
            "RR": [15.0, 5.0, 0.0], 
            "TN": [24.0, 24.5, 23.8],
            "TX": [34.0, 33.5, 34.2],
            "TAVG": [28.0, 28.1, 27.9],
            "RH_AVG": [85.0, 82.0, 80.0]
        }
        pd.DataFrame(data).to_csv(LOG_FILE, index=False)

init_log() 

def fetch_cuaca_medan_realtime():
    url = "https://api.open-meteo.com/v1/forecast?latitude=3.5952&longitude=98.6722&current=temperature_2m,relative_humidity_2m,precipitation&daily=temperature_2m_max,temperature_2m_min&timezone=Asia%2FJakarta"
    response = requests.get(url).json()
    
    return {
        "RR": response['current']['precipitation'],
        "TAVG": response['current']['temperature_2m'],
        "RH_AVG": response['current']['relative_humidity_2m'],
        "TX": response['daily']['temperature_2m_max'][0],
        "TN": response['daily']['temperature_2m_min'][0]
    }

def proses_pipeline_data():
    cuaca_hari_ini = fetch_cuaca_medan_realtime()
    tanggal_hari_ini = datetime.now().strftime("%Y-%m-%d")
    
    df = pd.read_csv(LOG_FILE)
    
    if tanggal_hari_ini not in df['Tanggal'].values:
        new_row = {"Tanggal": tanggal_hari_ini, **cuaca_hari_ini}
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    else:
        idx = df.index[df['Tanggal'] == tanggal_hari_ini].tolist()[0]
        for k, v in cuaca_hari_ini.items():
            df.at[idx, k] = v
            
    df.to_csv(LOG_FILE, index=False)
    
    df_last_3 = df.tail(3)
    df_last_7 = df.tail(7)
    
    return {
        **cuaca_hari_ini,
        "Hujan_3_Hari": float(df_last_3['RR'].sum()),
        "Hujan_7_Hari": float(df_last_7['RR'].sum()),
        "Kelembapan_3_Hari": float(df_last_3['RH_AVG'].mean())
    }

class DataCuaca(BaseModel):
    TN: float
    TX: float
    TAVG: float
    RH_AVG: float
    RR: float
    Hujan_3_Hari: float
    Hujan_7_Hari: float
    Kelembapan_3_Hari: float

def proses_ai(data_dict):
    input_data = pd.DataFrame([data_dict])
    
    urutan_fitur_training = [
        'TN', 'TX', 'TAVG', 'RH_AVG', 'RR', 
        'Hujan_3_Hari', 'Hujan_7_Hari', 'Kelembapan_3_Hari'
    ]
    
    input_data = input_data[urutan_fitur_training]
    
    probabilitas_ai = model.predict_proba(input_data)[0][1]
    
    wilayah_rentan_tinggi = ["Medan Maimun", "Medan Johor", "Medan Selayang", "Medan Baru", "Medan Petisah"]
    daftar_kecamatan = ["Medan Amplas", "Medan Area", "Medan Barat", "Medan Baru", "Medan Belawan", "Medan Deli", "Medan Denai", "Medan Helvetia", "Medan Johor", "Medan Kota", "Medan Labuhan", "Medan Maimun", "Medan Marelan", "Medan Perjuangan", "Medan Petisah", "Medan Polonia", "Medan Selayang", "Medan Sunggal", "Medan Tembung", "Medan Timur", "Medan Tuntungan"]
    
    hasil_peta = {}
    for kec in daftar_kecamatan:
        skor_akhir = probabilitas_ai + (0.15 if kec in wilayah_rentan_tinggi else 0)
        hasil_peta[kec] = "Kritis" if skor_akhir > 0.65 else "Waspada" if skor_akhir > 0.40 else "Aman"
            
    return hasil_peta

@app.post("/predict")
def prediksi_banjir(cuaca: DataCuaca):
    hasil_peta = proses_ai(cuaca.model_dump())
    return {"status": "success", "data_peta": hasil_peta}

@app.get("/auto-predict")
def auto_prediksi_banjir():
    data_lengkap = proses_pipeline_data()
    
    hasil_peta = proses_ai(data_lengkap)
    
    return {
        "status": "success",
        "data_cuaca_otomatis": data_lengkap, 
        "data_peta": hasil_peta
    }
