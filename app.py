import streamlit as st
import pandas as pd
import io
from google import genai
import plotly.express as px
from sklearn.ensemble import IsolationForest

# Konfigurasi Halaman Streamlit
st.set_page_config(
    page_title="Mini AI Data Analyst & Agentic AI",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Mini AI Data Analyst & Agentic Budget Agent")
st.markdown("Aplikasi asisten data cerdas berbasis Multi-Step Agentic AI, Visualisasi Plotly, dan Machine Learning.")

# Sidebar untuk Konfigurasi API Key
st.sidebar.header("🔑 Konfigurasi API")
api_key = st.sidebar.text_input("Masukkan Gemini API Key", type="password")

st.sidebar.markdown("---")
st.sidebar.markdown("### Navigasi Fitur")
app_mode = st.sidebar.radio("Pilih Menu:", [
    "💬 Tanya Jawab AI", 
    "📈 Visualisasi Grafik (Plotly)", 
    "🔍 Deteksi Anomali Anggaran (ML)",
    "🤖 Autonomous AI Agent (Auto-Audit)"
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
            st.markdown("Menggunakan algoritma **Isolation Forest** untuk mendeteksi pos belanja yang nilainya tidak normal.")

            numeric_columns = df.select_dtypes(include=['number']).columns.tolist()

            if len(numeric_columns) > 0:
                target_col = st.selectbox("Pilih Kolom Nilai Anggaran untuk Dianalisis", numeric_columns)
                contamination_rate = st.slider("Tingkat Kontaminasi (Estimasi Proporsi Anomali)", 0.01, 0.20, 0.05, 0.01)

                if st.button("Jalankan Deteksi Anomali"):
                    try:
                        ml_df = df.dropna(subset=[target_col]).copy()
                        X = ml_df[[target_col]]

                        iso = IsolationForest(contamination=contamination_rate, random_state=42)
                        ml_df['Anomaly'] = iso.fit_predict(X)
                        anomalies = ml_df[ml_df['Anomaly'] == -1]

                        st.success(f"🎉 Analisis Selesai! Ditemukan {len(anomalies)} data pos anggaran berstatus anomali.")

                        if len(anomalies) > 0:
                            st.markdown("#### ⚠️ Daftar Pos Anggaran yang Terdeteksi Anomali:")
                            st.dataframe(anomalies, use_container_width=True)
                        else:
                            st.info("Tidak ada anomali ekstrem yang ditemukan.")

                        fig = px.scatter(
                            ml_df, 
                            x=ml_df.index, 
                            y=target_col, 
                            color=ml_df['Anomaly'].astype(str),
                            title="Scatter Plot Deteksi Anomali",
                            labels={'x': 'Indeks Baris Data', target_col: 'Nominal Biaya'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Gagal menjalankan Machine Learning: {e}")
            else:
                st.warning("⚠️ Tidak ada kolom numerik yang tersedia.")

        # --- MENU 4: AUTONOMOUS AI AGENT (MULTI-STEP WORKFLOW) ---
        elif app_mode == "🤖 Autonomous AI Agent (Auto-Audit)":
            st.subheader("🤖 Autonomous AI Audit Agent")
            st.markdown("Agen otonom cerdas yang menjalankan **Multi-Step Pipeline** (Validasi Data ➔ Deteksi ML Anomali ➔ Sintesis Naratif Audit) secara otomatis dalam satu klik.")

            if st.button("🚀 Jalankan Autonomous Audit Pipeline"):
                if not api_key:
                    st.warning("⚠️ Mohon masukkan Gemini API Key terlebih dahulu di sidebar.")
                else:
                    try:
                        with st.spinner("🤖 Agent sedang mengeksekusi multi-step pipeline secara otonom..."):
                            # Step 1: Automated Data Profiling
                            total_rows = df.shape[0]
                            total_cols = df.shape[1]
                            missing_vals = int(df.isna().sum().sum())

                            # Step 2: Automated Machine Learning Anomaly Detection
                            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                            anomaly_summary = "Tidak ada kolom numerik untuk menjalankan deteksi anomali."
                            anomalies_count = 0
                            
                            if len(numeric_cols) > 0:
                                target_col = numeric_cols[0]
                                ml_df = df.dropna(subset=[target_col]).copy()
                                if len(ml_df) > 5:
                                    iso = IsolationForest(contamination=0.05, random_state=42)
                                    ml_df['Anomaly'] = iso.fit_predict(ml_df[[target_col]])
                                    anomalies = ml_df[ml_df['Anomaly'] == -1]
                                    anomalies_count = len(anomalies)
                                    anomaly_summary = f"Algoritma Isolation Forest memindai kolom '{target_col}' dan mendeteksi {anomalies_count} titik data anomali (lonjakan ekstrem)."

                            # Step 3: Cognitive Synthesis via Gemini AI Agent
                            client = genai.Client(api_key=api_key)
                            agent_prompt = f"""
                            Anda adalah Autonomous Chief Audit Agent yang sangat handal. Lakukan sintesis dan susun Laporan Audit Eksekutif komprehensif berdasarkan hasil pipeline otonom berikut:

                            1. **Data Profiling Metrics**: 
                               - Total Baris: {total_rows}
                               - Total Kolom: {total_cols}
                               - Missing Values: {missing_vals}
                            2. **Machine Learning Anomaly Audit**: 
                               - {anomaly_summary}
                            3. **Sampel Data Aktual (20 Baris Pertama)**:
                               {df.head(20).to_string()}

                            INSTRUKSI: Susun laporan audit profesional dalam bahasa Indonesia dengan struktur:
                            - **1. Executive Summary** (Gambaran umum kesehatan data/anggaran)
                            - **2. Automated Findings & ML Insights** (Temuan otomatis dari mesin dan profil data)
                            - **3. Risk & Anomaly Assessment** (Analisis potensi pembengkakan/anomali belanja)
                            - **4. Strategic Recommendations** (Rekomendasi langkah taktis bagi manajemen/tim)
                            """

                            response = client.models.generate_content(
                                model='gemini-3.6-flash',
                                contents=agent_prompt,
                            )

                            st.success("🎉 Autonomous Audit Pipeline Berhasil Dieksekusi!")
                            st.markdown("### 📋 Laporan Audit Eksekutif Otonom:")
                            st.markdown(response.text)

                    except Exception as e:
                        st.error(f"Terjadi kesalahan saat menjalankan Autonomous Agent: {e}")

    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses file: {e}")
else:
    st.info("👆 Silakan unggah file dataset anggaran (.csv atau .xlsx) di sidebar untuk mulai menggunakan agen otonom.")