# 🤖 Mini AI Data Analyst & Budget Intelligence Agent

Aplikasi web interaktif berbasis **Python**, **Streamlit**, **Google Gemini API**, **Plotly**, dan **Scikit-Learn** yang dirancang khusus untuk analisis laporan keuangan/anggaran secara instan, visualisasi data dinamis, deteksi anomali berbasis *Machine Learning*, hingga pembuatan laporan audit otonom (*Agentic AI*).

## 🚀 Fitur Utama

1. **Auto-Cleaning Engine:** 
   * Pembersihan otomatis struktur data laporan anggaran instansi yang menyerupai bentuk *Pivot Table* Excel (perbaikan *header* tergeser dan konversi kolom teks nominal menjadi tipe data angka murni).
2. **Conversational AI Q&A:** 
   * Tanya jawab data instan didukung oleh model **Gemini 3.6 Flash** untuk menjawab pertanyaan analitis secara lugas tanpa menyuruh pengguna menulis kode eksternal.
3. **Interactive Visualizations (Plotly):** 
   * Pembuatan grafik interaktif dinamis (Bar Chart, Scatter Plot, Line Chart) secara instan berdasarkan kolom data yang dipilih.
4. **Machine Learning Anomaly Detection:** 
   * Menggunakan algoritma **Isolation Forest** dari *scikit-learn* untuk mendeteksi pos belanja yang mengalami lonjakan atau anomali tidak wajar secara otomatis.
5. **Autonomous AI Audit Agent (Multi-Step Workflow):** 
   * Agen otonom yang mengeksekusi *pipeline* bertingkat (profiling data, pemindaian anomali ML, dan sintesis naratif) untuk menerbitkan **Laporan Audit Eksekutif** dalam satu kali klik.

## 🛠️ Tech Stack
* **Bahasa Pemrograman:** Python
* **Antarmuka Web:** Streamlit
* **Pengolahan & Pembersihan Data:** Pandas, Openpyxl
* **Kecerdasan Buatan:** Google GenAI SDK (`google-genai`)
* **Visualisasi Grafik:** Plotly
* **Machine Learning:** Scikit-Learn (Isolation Forest)

## 💻 Cara Menjalankan Secara Lokal (Local Installation)

1. **Clone repository ini:**
   ```bash
   git clone [https://github.com/hanggersukmapd/mini-ai-data-analyst.git](https://github.com/hanggersukmapd/mini-ai-data-analyst.git)
   cd mini-ai-data-analyst