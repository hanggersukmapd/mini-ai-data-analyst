import streamlit as st
import pandas as pd
import io  # <-- Tambahkan import ini di bagian atas
from google import genai

# Konfigurasi Halaman Streamlit
st.set_page_config(
    page_title="Mini AI Data Analyst Agent",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Mini AI Data Analyst Agent")
st.markdown("Unggah dataset kamu (CSV atau Excel), lalu tanyakan analisis, tren, atau wawasan langsung kepada AI.")

# Sidebar untuk Konfigurasi API Key
st.sidebar.header("🔑 Konfigurasi API")
api_key = st.sidebar.text_input("Masukkan Gemini API Key", type="password")

st.sidebar.markdown("---")
st.sidebar.markdown("### Tentang Aplikasi")
st.sidebar.info("Proyek kilat untuk eksplorasi data secara interaktif menggunakan kemampuan penalaran LLM.")

# Komponen File Uploader
uploaded_file = st.file_uploader("Pilih file dataset (Format: .csv atau .xlsx)", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Memuat dataset berdasarkan format file
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        # Tampilkan Pratinjau Data
        st.subheader("📊 Pratinjau Dataset")
        st.dataframe(df.head(10), use_container_width=True)

        # Ringkasan Informasi Data untuk Konteks AI
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Baris", df.shape[0])
        col2.metric("Total Kolom", df.shape[1])
        col3.metric("Total Missing Values", int(df.isna().sum().sum()))

        # --- PERBAIKAN ADA DI SINI ---
        # Ekstrak Info & Statistik untuk AI menggunakan io.StringIO
        buffer = io.StringIO()
        df.info(buf=buffer)
        info_str = buffer.getvalue()
        
        summary_stats = df.describe(include='all').to_string()

        st.markdown("---")

        # Input Pertanyaan dari Pengguna
        st.subheader("💬 Tanya Jawab & Analisis dengan AI")
        user_prompt = st.text_input(
            "Apa yang ingin kamu ketahui dari data ini?",
            placeholder="Contoh: 'Apa tren utama dari kolom X?' atau 'Rekomendasikan pembersihan data apa yang diperlukan.'"
        )

        if user_prompt:
            if not api_key:
                st.warning("⚠️ Mohon masukkan Gemini API Key terlebih dahulu di sidebar sebelah kiri.")
            else:
                try:
                    # Inisialisasi Google GenAI Client
                    client = genai.Client(api_key=api_key)

                    # Konstruksi Prompt Kontekstual
                    analysis_context = f"""
                    Anda adalah seorang Data Analyst profesional. Pengguna telah mengunggah sebuah dataset dengan karakteristik berikut:

                    1. Informasi Struktur Data (df.info()):
                    {info_str}

                    2. Statistik Deskriptif (df.describe()):
                    {summary_stats}

                    3. Contoh 5 Baris Pertama Data:
                    {df.head().to_string()}

                    Pertanyaan / Permintaan Pengguna:
                    "{user_prompt}"

                    Berikan analisis yang mendalam, terstruktur, tajam, dan solutif dalam bahasa Indonesia yang profesional. Jika diperlukan, berikan juga saran kode Pandas atau langkah visualisasi yang relevan.
                    """

                    with st.spinner("🤖 AI sedang menganalisis dataset kamu..."):
                        response = client.models.generate_content(
                            model='gemini-3.6-flash',
                            contents=analysis_context,
                        )

                        st.success("✅ Analisis Selesai!")
                        st.markdown("### Hasil Analisis AI:")
                        st.markdown(response.text)

                except Exception as e:
                    st.error(f"Terjadi kesalahan saat memproses permintaan: {e}")

    except Exception as e:
        st.error(f"Gagal membaca file. Pastikan format file benar. Detail error: {e}")
else:
    st.info("👆 Silakan unggah file dataset (.csv atau .xlsx) di atas untuk memulai.")