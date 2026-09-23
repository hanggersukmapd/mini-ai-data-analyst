import streamlit as st
import pandas as pd
import io
from google import genai
import plotly.express as px

# Konfigurasi Halaman Streamlit
st.set_page_config(
    page_title="Mini AI Data Analyst Agent",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Mini AI Data Analyst & Budget Assistant")
st.markdown("Aplikasi asisten data cerdas untuk analisis laporan anggaran dan visualisasi interaktif secara instan.")

# Sidebar untuk Konfigurasi API Key
st.sidebar.header("🔑 Konfigurasi API")
api_key = st.sidebar.text_input("Masukkan Gemini API Key", type="password")

st.sidebar.markdown("---")
st.sidebar.markdown("### Navigasi Fitur")
app_mode = st.sidebar.radio("Pilih Menu:", ["💬 Tanya Jawab AI", "📈 Visualisasi Grafik (Plotly)"])

# Komponen File Uploader
uploaded_file = st.file_uploader("Pilih file dataset (Format: .csv atau .xlsx)", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        # Memuat dataset awal
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        # --- AUTO-CLEANING KHUSUS LAPORAN KEUANGAN / PIVOT EXCEL ---
        # 1. Perbaiki header jika tergeser (Unnamed)
        if 'Unnamed' in str(df.columns[0]) or 'Unnamed' in str(df.columns[1]):
            df.columns = df.iloc[0]
            df = df[1:].reset_index(drop=True)
        
        # 2. Bersihkan dan ubah kolom teks berformat angka menjadi tipe numeric murni
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    # Bersihkan pemisah ribuan atau simbol mata uang jika ada
                    cleaned_numeric = pd.to_numeric(
                        df[col].astype(str)
                        .str.replace('.', '', regex=False)
                        .str.replace(',', '.', regex=False)
                        .str.replace('Rp', '', regex=False)
                        .str.strip(),
                        errors='coerce'
                    )
                    # Jika setidaknya 30% baris valid sebagai angka, konversi kolom ini jadi numerik
                    if cleaned_numeric.notna().sum() >= len(df) * 0.3:
                        df[col] = cleaned_numeric
                except:
                    pass

        # --- MENU 1: TANYA JAWAB AI ---
        if app_mode == "💬 Tanya Jawab AI":
            st.subheader("📊 Pratinjau Dataset (Tersaring & Bersih)")
            st.dataframe(df.head(10), use_container_width=True)

            col1, col2, col3 = st.columns(3)
            col1.metric("Total Baris", df.shape[0])
            col2.metric("Total Kolom", df.shape[1])
            col3.metric("Total Missing Values", int(df.isna().sum().sum()))

            buffer = io.StringIO()
            df.info(buf=buffer)
            info_str = buffer.getvalue()
            summary_stats = df.describe(include='all').to_string()
            data_sample = df.head(30).to_string()

            st.markdown("---")
            st.subheader("💬 Tanya Jawab & Analisis Instan dengan AI")
            user_prompt = st.text_input(
                "Apa yang ingin kamu ketahui dari data ini?",
                placeholder="Contoh: 'Berapa nilai biaya tertinggi dan di baris mana?' atau 'Berikan ringkasan anggaran.'"
            )

            if user_prompt:
                if not api_key:
                    st.warning("⚠️ Mohon masukkan Gemini API Key terlebih dahulu di sidebar.")
                else:
                    try:
                        client = genai.Client(api_key=api_key)
                        analysis_context = f"""
                        Anda adalah Data Analyst profesional. Jawab pertanyaan pengguna secara langsung, akurat, dan tuntas berdasarkan data di bawah ini. JANGAN menyuruh pengguna menjalankan kode Python eksternal.
                        
                        1. Info Data: {info_str}
                        2. Statistik: {summary_stats}
                        3. Sampel Data: {data_sample}

                        Pertanyaan: "{user_prompt}"
                        """
                        with st.spinner("🤖 AI sedang menganalisis data..."):
                            response = client.models.generate_content(
                                model='gemini-3.6-flash',
                                contents=analysis_context,
                            )
                            st.success("✅ Analisis Selesai!")
                            st.markdown("### Hasil Analisis AI:")
                            st.markdown(response.text)
                    except Exception as e:
                        st.error(f"Terjadi kesalahan: {e}")

        # --- MENU 2: VISUALISASI GRAFIK PLOTLY ---
        elif app_mode == "📈 Visualisasi Grafik (Plotly)":
            st.subheader("📈 Generator Grafik Anggaran Interaktif")
            st.markdown("Visualisasikan komponen belanja dan nominal anggaran secara otomatis.")

            numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
            all_columns = df.columns.tolist()

            if len(numeric_columns) > 0:
                chart_type = st.selectbox("Pilih Jenis Grafik", ["Bar Chart (Batang)", "Scatter Plot", "Line Chart"])
                
                col_x = st.selectbox("Pilih Kolom Label / Kategori (Sumbu X)", all_columns)
                col_y = st.selectbox("Pilih Kolom Nominal Biaya (Sumbu Y - Numerik)", numeric_columns)

                if st.button("Buat Grafik"):
                    if chart_type == "Bar Chart (Batang)":
                        fig = px.bar(df, x=col_x, y=col_y, title=f"Grafik Batang: {col_y} per {col_x}")
                    elif chart_type == "Scatter Plot":
                        fig = px.scatter(df, x=col_x, y=col_y, title=f"Scatter Plot: {col_y} vs {col_x}")
                    else:
                        fig = px.line(df, x=col_x, y=col_y, title=f"Line Chart: {col_y} per {col_x}")
                    
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("⚠️ Kolom numerik belum terdeteksi secara otomatis. Pastikan file Excel memiliki kolom angka biaya yang jelas.")

    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses file: {e}")
else:
    st.info("👆 Silakan unggah file dataset (.csv atau .xlsx) di sidebar untuk mulai menggunakan aplikasi.")