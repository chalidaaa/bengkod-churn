# Deployment — Prediksi Churn Pelanggan (Week 5)

Aplikasi Streamlit untuk memprediksi churn pelanggan menggunakan model terbaik
(**Random Forest** hasil *hyperparameter tuning* pada Week 4).

## Tentang Aplikasi

Pengguna memasukkan data pelanggan (profil, aktivitas, transaksi), lalu aplikasi
menampilkan prediksi churn/bertahan, probabilitas churn, serta rekomendasi tindakan
yang mudah dipahami pengguna awam.

## 🔗 Link Aplikasi (Streamlit Cloud)

> _Belum dideploy — link akan diperbarui setelah aplikasi online._

## Struktur

```
deployment/
├── app.py              # Aplikasi Streamlit (form input, prediksi, hasil)
├── train_export.py     # Script melatih & menyimpan model ke .joblib
├── churn_model.joblib  # Artefak model (model + scaler + metadata fitur)
└── README.md
```

## Menjalankan Secara Lokal

1. Install dependensi (dari root proyek):
   ```bash
   pip install -r requirements.txt
   ```
2. (Opsional) Buat ulang model:
   ```bash
   python deployment/train_export.py
   ```
3. Jalankan aplikasi:
   ```bash
   streamlit run deployment/app.py
   ```
4. Buka `http://localhost:8501` di browser.

## Catatan Penting

- Versi paket di `requirements.txt` disamakan dengan versi saat model dibuat agar
  file `.joblib` dapat dimuat tanpa error kompatibilitas.
- Pipeline preprocessing di `app.py` (encoding + scaling) direplikasi persis dari
  notebook Week 3 agar input pengguna konsisten dengan data latih.
