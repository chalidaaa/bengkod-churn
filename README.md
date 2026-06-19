# 📉 Prediksi Churn Pelanggan — UAS Bengkel Koding Data Science

Proyek ini membangun model *machine learning* untuk memprediksi **churn pelanggan**
(pelanggan yang berhenti berlangganan) pada **Sales and Marketing Customer Dataset**,
lalu men-*deploy* model terbaik ke aplikasi web interaktif berbasis **Streamlit**.

Tujuannya membantu tim bisnis/pemasaran mengenali pelanggan yang berpotensi pergi,
sehingga bisa dilakukan tindakan retensi (promo, peningkatan layanan, dsb.) lebih awal.

---

## 👤 Identitas

| | |
|---|---|
| **Nama** | Chalida Abdat |
| **NIM** | A11.2023.15031 |
| **Learning Path** | Data Science |

---

## 🔗 Link Aplikasi (Streamlit Cloud)

> (https://prediksi-churn-bengkod.streamlit.app/)

---

## 🧩 Tentang Aplikasi

Dashboard Streamlit (`deployment/app.py`) memungkinkan pengguna memasukkan data
seorang pelanggan (profil, aktivitas, dan transaksi), lalu menampilkan:

- **Prediksi** apakah pelanggan berpotensi *churn* atau *bertahan*.
- **Probabilitas churn** beserta tingkat risikonya (Rendah / Sedang / Tinggi).
- **Rekomendasi tindakan** yang mudah dipahami oleh pengguna awam.

Model yang dipakai adalah **Random Forest** (model terbaik hasil *hyperparameter tuning*),
yang disimpan bersama *scaler* dan metadata fitur dalam satu file `.joblib`.

---

## 📂 Struktur Proyek

```
bengkod-churn/
├── dataset/
│   └── Sales - Marketing customer dataset.csv
├── notebook/
│   ├── EDA_Minggu1.ipynb        # Week 1 - Exploratory Data Analysis
│   ├── Direct_Modeling.ipynb    # Week 2 - Direct Modeling (baseline)
│   ├── Preprocessing.ipynb      # Week 3 & 4 - Preprocessing + Hyperparameter Tuning
│   └── Deployment.ipynb         # Week 5 - Dokumentasi Deployment
├── deployment/
│   ├── app.py                   # Aplikasi Streamlit
│   ├── train_export.py          # Script melatih & menyimpan model
│   ├── churn_model.joblib       # Artefak model (model + scaler + fitur)
│   └── README.md
├── requirements.txt
└── README.md
```

---

## 📊 Tahapan Pengerjaan

1. **EDA** — memahami data, missing value, distribusi target (churn), dan korelasi fitur.
2. **Direct Modeling** — baseline 3 model tanpa preprocessing (Logistic Regression, Random Forest, VotingClassifier).
3. **Modeling + Preprocessing** — penanganan missing value, outlier, encoding, dan scaling.
4. **Hyperparameter Tuning** — optimasi ketiga model dengan `RandomizedSearchCV`.
5. **Deployment** — model terbaik dipublikasikan lewat aplikasi Streamlit.

---

## 💻 Cara Menjalankan Proyek Secara Lokal

> Disarankan memakai *virtual environment* agar dependensi rapi.

1. **Buat & aktifkan virtual environment** (sekali saja):
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

2. **Install dependensi** (dari folder root proyek):
   ```bash
   pip install -r requirements.txt
   ```

3. **(Opsional) Buat ulang file model:**
   ```bash
   python deployment/train_export.py
   ```

4. **Jalankan aplikasi Streamlit:**
   ```bash
   streamlit run deployment/app.py
   ```

5. Buka browser ke **http://localhost:8501**.

---

## 🛠️ Teknologi

Python • pandas • numpy • scikit-learn • matplotlib • seaborn • Streamlit

---

_Dibuat untuk memenuhi tugas UAS Bengkel Koding Data Science — Universitas Dian Nuswantoro._
