import streamlit as st
import pandas as pd
import io
from google import genai
import plotly.express as px
from sklearn.ensemble import IsolationForest

# Konfigurasi Halaman Streamlit
st.set_page_config(
    page_title="Mini AI Data Analyst & ML Agent",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Mini AI Data Analyst & Budget Intelligence Agent")
st.markdown("Aplikasi asisten data cerdas dengan analisis AI, visualisasi Plotly, dan Machine Learning (Anomaly Detection) untuk audit anggaran.")

# Sidebar untuk Konfigurasi API Key
st.sidebar.header("🔑 Konfigurasi API")
api_key = st.sidebar.text_input("Masukkan Gemini API Key", type="password")

st.sidebar.markdown("---")
st.sidebar.markdown("### Navigasi Fitur")
app_mode = st.sidebar.radio("Pilih Menu:", [
    "💬 Tanya Jawab AI", 
    "📈 Visualisasi Grafik (Plotly)", 
    "🔍 Deteksi Anomali Anggaran (ML)"
])

# Komponen File Uploader
uploaded_file = st.file_uploader("Pilih file dataset anggaran (Format: .csv atau .xlsx)", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        # Memuat dataset awal
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        # --- AUTO-CLEANING KHUSUS LAPORAN KEUANGAN / PIVOT EXCEL ---
        if 'Unnamed' in str(df.columns[0]) or 'Unnamed' in str(df.columns[1]):
            df.columns = df.iloc[0]
            df = df[1:].reset_index(drop=True)
        
        # Membersihkan dan mengubah kolom teks berformat angka menjadi tipe numeric murni
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    cleaned_numeric = pd.to_numeric(
                        df[col].astype(str)
                        .str.replace('.', '', regex=False)
                        .str.replace(',', '.', regex=False)
                        .str.replace('Rp', '', regex=False)
                        .str.strip(),
                        errors='coerce'
                    )
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
                    st.warning("⚠️ Kolom numerik belum terdeteksi otomatis.")

        # --- MENU 3: DETEKSI ANOMALI ANGGARAN (MACHINE LEARNING) ---
        elif app_mode == "🔍 Deteksi Anomali Anggaran (ML)":
            st.subheader("🔍 Modul Machine Learning: Deteksi Anomali Anggaran")
            st.markdown("Menggunakan algoritma **Isolation Forest** untuk mendeteksi pos belanja yang nilainya tidak normal (terlalu tinggi/rendah secara ekstrem).")

            numeric_columns = df.select_dtypes(include=['number']).columns.tolist()

            if len(numeric_columns) > 0:
                target_col = st.selectbox("Pilih Kolom Nilai Anggaran untuk Dianalisis", numeric_columns)
                
                contamination_rate = st.slider("Tingkat Kontaminasi (Estimasi Proporsi Anomali)", 0.01, 0.20, 0.05, 0.01)

                if st.button("Jalankan Deteksi Anomali"):
                    try:
                        # Siapkan data non-null untuk ML
                        ml_df = df.dropna(subset=[target_col]).copy()
                        X = ml_df[[target_col]]

                        # Jalankan Isolation Forest
                        iso = IsolationForest(contamination=contamination_rate, random_state=42)
                        ml_df['Anomaly'] = iso.fit_predict(X)
                        
                        # Anomali ditandai dengan nilai -1 oleh Isolation Forest
                        anomalies = ml_df[ml_df['Anomaly'] == -1]

                        st.success(f"🎉 Analisis Selesai! Ditemukan {len(anomalies)} data pos anggaran berstatus anomali.")

                        if len(anomalies) > 0:
                            st.markdown("#### ⚠️ Daftar Pos Anggaran yang Terdeteksi Anomali / Tidak Wajar:")
                            st.dataframe(anomalies, use_container_width=True)
                        else:
                            st.info("Tidak ada anomali ekstrem yang ditemukan dengan tingkat sensitivitas ini.")

                        # Scatter plot untuk visualisasi anomali
                        st.markdown("#### 📊 Visualisasi Distribusi & Anomali")
                        fig = px.scatter(
                            ml_df, 
                            x=ml_df.index, 
                            y=target_col, 
                            color=ml_df['Anomaly'].astype(str),
                            title="Scatter Plot Deteksi Anomali (Warna merah/berbeda menandakan anomali)",
                            labels={'x': 'Indeks Baris Data', target_col: 'Nominal Biaya'}
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    except Exception as e:
                        st.error(f"Gagal menjalankan Machine Learning: {e}")
            else:
                st.warning("⚠️ Tidak ada kolom numerik yang tersedia untuk menjalankan deteksi anomali.")

    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses file: {e}")
else:
    st.info("👆 Silakan unggah file dataset (.csv atau .xlsx) di sidebar untuk mulai menggunakan aplikasi.")