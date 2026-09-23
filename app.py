import streamlit as st
import pandas as pd
import io
from google import genai
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Konfigurasi Halaman Streamlit
st.set_page_config(
    page_title="Mini AI Data Analyst Agent",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Mini AI Data Analyst & Predictive Agent")
st.markdown("Aplikasi asisten data cerdas berbasis AI, visualisasi interaktif Plotly, dan modul Machine Learning ringan untuk portofolio & Capstone.")

# Sidebar untuk Konfigurasi API Key
st.sidebar.header("🔑 Konfigurasi API")
api_key = st.sidebar.text_input("Masukkan Gemini API Key", type="password")

st.sidebar.markdown("---")
st.sidebar.markdown("### Navigasi Fitur")
app_mode = st.sidebar.radio("Pilih Menu:", ["💬 Tanya Jawab AI", "📈 Visualisasi Grafik (Plotly)", "⚙️ Prediksi Machine Learning"])

# Komponen File Uploader Global di Sidebar / Atas
uploaded_file = st.file_uploader("Pilih file dataset (Format: .csv atau .xlsx)", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        # Memuat dataset
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        # --- MENU 1: TANYA JAWAB AI ---
        if app_mode == "💬 Tanya Jawab AI":
            st.subheader("📊 Pratinjau Dataset")
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
                placeholder="Contoh: 'Berapa nilai biaya tertinggi dan di baris mana?' atau 'Berikan ringkasan tren data.'"
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
            st.subheader("📈 Generator Grafik Interaktif")
            st.markdown("Buat visualisasi data secara instan menggunakan pustaka Plotly.")

            numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
            all_columns = df.columns.tolist()

            if len(numeric_columns) > 0:
                chart_type = st.selectbox("Pilih Jenis Grafik", ["Scatter Plot", "Bar Chart", "Line Chart"])
                
                col_x = st.selectbox("Pilih Kolom Sumbu X", all_columns)
                col_y = st.selectbox("Pilih Kolom Sumbu Y (Numerik)", numeric_columns)

                if st.button("Buat Grafik"):
                    if chart_type == "Scatter Plot":
                        fig = px.scatter(df, x=col_x, y=col_y, title=f"Scatter Plot: {col_y} berdasarkan {col_x}")
                    elif chart_type == "Bar Chart":
                        fig = px.bar(df, x=col_x, y=col_y, title=f"Bar Chart: {col_y} berdasarkan {col_x}")
                    else:
                        fig = px.line(df, x=col_x, y=col_y, title=f"Line Chart: {col_y} berdasarkan {col_x}")
                    
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Dataset tidak memiliki kolom numerik yang cukup untuk membuat grafik.")

        # --- MENU 3: PREDIKSI MACHINE LEARNING (NUANSA CAPSTONE) ---
        elif app_mode == "⚙️ Prediksi Machine Learning":
            st.subheader("⚙️ Modul Uji Coba Machine Learning (Random Forest)")
            st.markdown("Simulasi pemodelan prediktif otomatis untuk mengklasifikasikan atau memprediksi target data.")

            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            if len(numeric_cols) >= 2:
                target_col = st.selectbox("Pilih Kolom Target (Label)", df.columns.tolist())
                feature_cols = st.multiselect("Pilih Kolom Fitur (Prediktor)", numeric_cols, default=[c for c in numeric_cols if c != target_col][:2])

                if st.button("Jalankan Model Pelatihan"):
                    try:
                        # Membersihkan data dari missing values untuk simulasi ML
                        ml_df = df[feature_cols + [target_col]].dropna()
                        X = ml_df[feature_cols]
                        y = ml_df[target_col]

                        # Jika target berupa kontinyu/float banyak, ubah jadi kategori sederhana untuk klasifikasi contoh
                        if y.dtype == 'float64' or len(y.unique()) > 10:
                            y = pd.qcut(y, q=2, labels=[0, 1])

                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                        model = RandomForestClassifier(random_state=42)
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)

                        acc = accuracy_score(y_test, y_pred)
                        st.success(f"🎉 Model Random Forest berhasil dilatih!")
                        st.metric("Akurasi Model pada Data Uji", f"{acc * 100:.2f}%")

                        # Feature Importance
                        importance_df = pd.DataFrame({
                            'Fitur': feature_cols,
                            'Importance': model.feature_importances_
                        }).sort_values(by='Importance', ascending=False)

                        st.markdown("#### Tingkat Kepentingan Fitur (Feature Importance):")
                        st.dataframe(importance_df, use_container_width=True)

                    except Exception as e:
                        st.error(f"Gagal menjalankan pemodelan ML: {e}")
            else:
                st.warning("Dataset memerlukan minimal 2 kolom numerik untuk menjalankan simulasi Machine Learning.")

    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses file: {e}")
else:
    st.info("👆 Silakan unggah file dataset (.csv atau .xlsx) di sidebar untuk mulai menggunakan fitur lengkap.")