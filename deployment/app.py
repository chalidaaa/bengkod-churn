"""
Aplikasi Streamlit - Prediksi Churn Pelanggan
==============================================
Dashboard untuk memprediksi apakah seorang pelanggan berpotensi churn
(berhenti berlangganan) menggunakan model Random Forest hasil tuning.

Menjalankan secara lokal:
    streamlit run deployment/app.py
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ============================================================
# Konfigurasi halaman
# ============================================================
st.set_page_config(
    page_title="Prediksi Churn Pelanggan - UAS Bengkel Koding",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = Path(__file__).resolve().parent / "churn_model.joblib"

# Pilihan kategori (sesuai nilai unik di dataset)
GENDER_OPTS = ["Female", "Male", "Other"]
ACQUISITION_OPTS = ["Email", "Facebook Ads", "Google Ads", "Organic", "Referral"]
DEVICE_OPTS = ["Desktop", "Mobile", "Tablet"]
SUBSCRIPTION_OPTS = ["Annual", "Monthly"]
PAYMENT_OPTS = ["BKash", "Card", "PayPal", "SEPA", "UPI"]


# ============================================================
# Tema & styling
# ============================================================
THEME = {
    "card": "#f8fafc", "text": "#0f172a", "muted": "#475569", "border": "#e2e8f0",
    "header1": "#1e3a8a", "header2": "#2563eb", "accent": "#2563eb",
    "ok_bg": "#dcfce7", "ok_border": "#22c55e", "ok_text": "#166534",
    "bad_bg": "#fee2e2", "bad_border": "#ef4444", "bad_text": "#b91c1c",
}


def inject_css(t: dict) -> None:
    st.markdown(
        f"""
        <style>
        .main-header {{
            background: linear-gradient(90deg, {t['header1']} 0%, {t['header2']} 100%);
            padding: 1.6rem 2rem; border-radius: 14px; color: white; margin-bottom: 1.2rem;
            box-shadow: 0 4px 14px rgba(0,0,0,0.12);
        }}
        .main-header h1 {{ color: #fff; margin: 0; font-size: 1.9rem; }}
        .main-header p {{ color: #dbeafe; margin: 0.4rem 0 0 0; font-size: 0.95rem; }}
        .step-card {{
            background: {t['card']}; border: 1px solid {t['border']};
            border-radius: 12px; padding: 1rem 1.1rem; height: 100%;
        }}
        .step-card h4 {{ margin: 0 0 0.3rem 0; color: {t['text']}; font-size: 1rem; }}
        .step-card p {{ margin: 0; color: {t['muted']}; font-size: 0.86rem; }}
        .step-badge {{
            display: inline-flex; align-items: center; justify-content: center;
            background: {t['accent']}; color: #fff; border-radius: 50%;
            width: 26px; height: 26px; font-weight: bold; margin-right: 8px;
        }}
        .result-card {{ padding: 1.6rem; border-radius: 14px; text-align: center; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# Fungsi: load model & preprocessing input
# ============================================================
@st.cache_resource
def load_bundle():
    """Memuat bundle model (model + scaler + metadata fitur)."""
    return joblib.load(MODEL_PATH)


def build_input_row(raw: dict, bundle: dict) -> pd.DataFrame:
    """Mengubah input mentah menjadi 1 baris fitur sesuai pipeline training."""
    df = pd.DataFrame([raw])

    df["days_since_signup"] = raw["days_since_signup"]
    df["days_since_last_purchase"] = raw["days_since_last_purchase"]

    # One-hot encoding manual (drop_first=True spt training)
    for opt in GENDER_OPTS[1:]:
        df[f"gender_{opt}"] = int(raw["gender"] == opt)
    for opt in ACQUISITION_OPTS[1:]:
        df[f"acquisition_channel_{opt}"] = int(raw["acquisition_channel"] == opt)
    for opt in DEVICE_OPTS[1:]:
        df[f"device_type_{opt}"] = int(raw["device_type"] == opt)
    for opt in SUBSCRIPTION_OPTS[1:]:
        df[f"subscription_type_{opt}"] = int(raw["subscription_type"] == opt)
    for opt in PAYMENT_OPTS[1:]:
        df[f"payment_method_{opt}"] = int(raw["payment_method"] == opt)

    df = df.drop(
        columns=[
            "gender", "acquisition_channel", "device_type",
            "subscription_type", "payment_method",
        ]
    )

    df = df.reindex(columns=bundle["feature_columns"], fill_value=0)
    df[bundle["features_to_scale"]] = bundle["scaler"].transform(df[bundle["features_to_scale"]])
    return df


# ============================================================
# SIDEBAR - Informasi
# ============================================================
with st.sidebar:
    st.markdown("### ℹ️ Tentang Dashboard")
    st.write(
        "Dashboard ini memprediksi apakah seorang pelanggan **berpotensi churn** "
        "(berhenti berlangganan) berdasarkan profil & aktivitasnya."
    )
    st.markdown("**Model:** Random Forest (hasil hyperparameter tuning)")
    st.markdown("**Dataset:** Sales & Marketing Customer (15.000 data)")
    st.divider()
    st.markdown("### 📖 Apa itu Churn?")
    st.info(
        "**Churn** = pelanggan berhenti memakai layanan. Memprediksi churn membantu "
        "perusahaan mempertahankan pelanggan sebelum mereka pergi."
    )
    st.caption("UAS Bengkel Koding Data Science • UDINUS")

# Terapkan styling
inject_css(THEME)


# ============================================================
# HEADER UTAMA
# ============================================================
st.markdown(
    """
    <div class="main-header">
        <h1>📉 Prediksi Churn Pelanggan</h1>
        <p>Tugas UAS - Bengkel Koding | Chalida Abdat (A11.2023.15031) - Data Science</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Cek ketersediaan model
if not MODEL_PATH.exists():
    st.error(
        "⚠️ File model `churn_model.joblib` belum ada. Jalankan dulu "
        "`python deployment/train_export.py` untuk membuatnya."
    )
    st.stop()

bundle = load_bundle()

# Panduan singkat alur penggunaan
st.markdown("#### 🧭 Cara Menggunakan Dashboard")
g1, g2, g3 = st.columns(3)
with g1:
    st.markdown(
        '<div class="step-card"><h4><span class="step-badge">1</span>Isi data pelanggan</h4>'
        '<p>Lengkapi data pada tab Profil, Aktivitas, dan Transaksi di bawah. Ingat ya, ada 3 tab di langkah 1.</p></div>',
        unsafe_allow_html=True,
    )
with g2:
    st.markdown(
        '<div class="step-card"><h4><span class="step-badge">2</span>Klik tombol Prediksi</h4>'
        '<p>Tekan tombol (button) di bagian bawah form input, yang bertuliskan "Prediksi Sekarang".</p></div>',
        unsafe_allow_html=True,
    )
with g3:
    st.markdown(
        '<div class="step-card"><h4><span class="step-badge">3</span>Lihat hasil & saran</h4>'
        '<p>Hasil prediksi beserta rekomendasi tindakan akan muncul otomatis.</p></div>',
        unsafe_allow_html=True,
    )

st.write("")
st.divider()

# ============================================================
# LANGKAH 1 - FORM INPUT (dikelompokkan dengan tab)
# ============================================================
st.markdown("### 📝 Langkah 1 — Masukkan Data Pelanggan")
st.caption("Nilai default sudah diisi dengan rata-rata pelanggan. Ubah sesuai data yang ingin diuji.")

with st.form("form_prediksi"):
    tab_profil, tab_aktivitas, tab_transaksi = st.tabs(
        ["👤 Profil Pelanggan", "📊 Aktivitas & Engagement", "💳 Transaksi & Layanan"]
    )

    # --- Tab 1: Profil ---
    with tab_profil:
        c1, c2 = st.columns(2)
        with c1:
            gender = st.selectbox("Jenis Kelamin", GENDER_OPTS,
                                   help="Jenis kelamin pelanggan.")
            age = st.slider("Usia (tahun)", 18, 80, 35)
            subscription_type = st.selectbox("Tipe Langganan", SUBSCRIPTION_OPTS,
                                              help="Annual = tahunan, Monthly = bulanan.")
            is_premium_user = st.radio("Pengguna Premium?", ["Tidak", "Ya"], horizontal=True)
        with c2:
            device_type = st.selectbox("Perangkat yang Digunakan", DEVICE_OPTS)
            acquisition_channel = st.selectbox("Sumber Akuisisi", ACQUISITION_OPTS,
                                                help="Dari mana pelanggan pertama kali datang.")
            days_since_signup = st.number_input("Hari Sejak Mendaftar", 0, 3000, 500,
                                                 help="Sudah berapa hari pelanggan terdaftar.")
            days_since_last_purchase = st.number_input("Hari Sejak Pembelian Terakhir", 0, 2000, 300,
                                                        help="Makin lama tidak membeli, makin berisiko churn.")

    # --- Tab 2: Aktivitas ---
    with tab_aktivitas:
        c1, c2 = st.columns(2)
        with c1:
            total_visits = st.number_input("Total Kunjungan", 0, 100, 15)
            avg_session_time = st.number_input("Rata-rata Waktu Sesi (menit)", 0.0, 60.0, 8.0)
            pages_per_session = st.number_input("Halaman per Sesi", 0.0, 30.0, 4.0)
            last_3_month_purchase_freq = st.number_input("Frekuensi Beli 3 Bulan Terakhir", 0, 30, 7)
        with c2:
            email_open_rate = st.slider("Email Open Rate", 0.0, 1.0, 0.5,
                                        help="Proporsi email yang dibuka (0 - 1).")
            email_click_rate = st.slider("Email Click Rate", 0.0, 1.0, 0.25,
                                         help="Proporsi email yang diklik (0 - 1).")
            satisfaction_score = st.slider("Skor Kepuasan", 1.0, 5.0, 3.5,
                                           help="1 = sangat tidak puas, 5 = sangat puas.")
            nps_score = st.slider("NPS Score", 0, 10, 5,
                                  help="Seberapa besar pelanggan merekomendasikan layanan (0 - 10).")

    # --- Tab 3: Transaksi ---
    with tab_transaksi:
        c1, c2 = st.columns(2)
        with c1:
            total_spent = st.number_input("Total Pengeluaran (Rp ribu)", 0.0, 20000.0, 500.0)
            avg_order_value = st.number_input("Rata-rata Nilai Order", 0.0, 200.0, 60.0)
            lifetime_value = st.number_input("Lifetime Value", 0.0, 5000.0, 1200.0)
            marketing_spend_per_user = st.number_input("Biaya Marketing per User", 0.0, 50.0, 17.5)
        with c2:
            payment_method = st.selectbox("Metode Pembayaran", PAYMENT_OPTS)
            discount_used = st.radio("Pernah Pakai Diskon?", ["Tidak", "Ya"], horizontal=True)
            refund_requested = st.radio("Pernah Minta Refund?", ["Tidak", "Ya"], horizontal=True)
            support_tickets = st.number_input("Jumlah Tiket Dukungan", 0, 20, 2,
                                              help="Berapa kali pelanggan menghubungi customer support.")
            delivery_delay_days = st.number_input("Keterlambatan Pengiriman (hari)", 0, 30, 3)

    st.write("")
    submitted = st.form_submit_button("🔮 Prediksi Sekarang", use_container_width=True, type="primary")


# ============================================================
# LANGKAH 2 - PROSES & HASIL
# ============================================================
st.write("")
st.markdown("### 📈 Langkah 2 — Hasil Prediksi")

if not submitted:
    st.info("👆 Lengkapi data di atas lalu klik **Prediksi Sekarang** untuk melihat hasilnya.")
else:
    raw = {
        "age": float(age),
        "is_premium_user": 1 if is_premium_user == "Ya" else 0,
        "total_visits": int(total_visits),
        "avg_session_time": float(avg_session_time),
        "pages_per_session": float(pages_per_session),
        "email_open_rate": float(email_open_rate),
        "email_click_rate": float(email_click_rate),
        "total_spent": float(total_spent),
        "avg_order_value": float(avg_order_value),
        "discount_used": 1 if discount_used == "Ya" else 0,
        "support_tickets": int(support_tickets),
        "refund_requested": 1 if refund_requested == "Ya" else 0,
        "delivery_delay_days": int(delivery_delay_days),
        "satisfaction_score": float(satisfaction_score),
        "nps_score": int(nps_score),
        "marketing_spend_per_user": float(marketing_spend_per_user),
        "lifetime_value": float(lifetime_value),
        "last_3_month_purchase_freq": int(last_3_month_purchase_freq),
        "days_since_signup": int(days_since_signup),
        "days_since_last_purchase": int(days_since_last_purchase),
        "gender": gender,
        "acquisition_channel": acquisition_channel,
        "device_type": device_type,
        "subscription_type": subscription_type,
        "payment_method": payment_method,
    }

    X_input = build_input_row(raw, bundle)
    model = bundle["model"]
    pred = int(model.predict(X_input)[0])
    proba = float(model.predict_proba(X_input)[0][1])

    # --- Tampilan hasil utama ---
    res_col, gauge_col = st.columns([1.3, 1])

    with res_col:
        if pred == 1:
            st.markdown(
                f"""
                <div class="result-card" style="background:{THEME['bad_bg']}; border:2px solid {THEME['bad_border']};">
                    <h2 style="color:{THEME['bad_text']}; margin:0;">⚠️ BERPOTENSI CHURN</h2>
                    <p style="color:{THEME['bad_text']}; margin:0.5rem 0 0 0;">
                        Pelanggan ini berisiko berhenti berlangganan.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="result-card" style="background:{THEME['ok_bg']}; border:2px solid {THEME['ok_border']};">
                    <h2 style="color:{THEME['ok_text']}; margin:0;">✅ KEMUNGKINAN BERTAHAN</h2>
                    <p style="color:{THEME['ok_text']}; margin:0.5rem 0 0 0;">
                        Pelanggan ini cenderung tetap loyal.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with gauge_col:
        st.metric("Probabilitas Churn", f"{proba:.1%}")
        st.progress(proba)
        tingkat = "Tinggi" if proba >= 0.5 else ("Sedang" if proba >= 0.3 else "Rendah")
        st.caption(f"Tingkat risiko: **{tingkat}**")

    st.divider()

    # --- Interpretasi & rekomendasi untuk orang awam ---
    st.markdown("#### 💡 Interpretasi & Rekomendasi")
    if pred == 1:
        st.warning(
            f"Model memperkirakan peluang pelanggan ini churn sebesar **{proba:.1%}**. "
            "Berikut beberapa hal yang bisa diperhatikan:"
        )
        saran = []
        if satisfaction_score <= 3:
            saran.append("Skor kepuasan rendah — tindak lanjuti keluhan & tingkatkan layanan.")
        if days_since_last_purchase > 365:
            saran.append("Sudah lama tidak bertransaksi — kirim promo/penawaran khusus.")
        if support_tickets >= 4:
            saran.append("Banyak tiket dukungan — pastikan masalah pelanggan terselesaikan.")
        if not saran:
            saran.append("Tawarkan program loyalitas atau diskon untuk mempertahankan pelanggan.")
        for s in saran:
            st.markdown(f"- {s}")
    else:
        st.success(
            f"Peluang churn hanya **{proba:.1%}**. Pelanggan ini relatif aman. "
            "Tetap jaga kualitas layanan dan pertimbangkan menawarkan program loyalitas "
            "agar tetap setia."
        )

    with st.expander("🔍 Lihat ringkasan data yang dimasukkan"):
        ringkasan = pd.DataFrame(
            {
                "Atribut": ["Usia", "Tipe Langganan", "Skor Kepuasan", "NPS",
                            "Total Pengeluaran", "Tiket Dukungan", "Hari Sejak Beli Terakhir"],
                "Nilai": [age, subscription_type, satisfaction_score, nps_score,
                          total_spent, support_tickets, days_since_last_purchase],
            }
        )
        st.table(ringkasan)

    st.caption(
        "⚠️ Hasil prediksi bersifat estimasi statistik dari model machine learning, "
        "bukan kepastian mutlak."
    )
