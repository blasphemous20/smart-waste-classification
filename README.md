# ♻️ Smart Waste Classification

Aplikasi klasifikasi sampah cerdas berbasis *Deep Learning* menggunakan arsitektur **MobileNetV2** (PyTorch) dan antarmuka interaktif **Streamlit**. Sistem ini dirancang untuk mendeteksi dan mengelompokkan jenis sampah dari gambar yang diunggah secara otomatis guna mendukung pengelolaan sampah dan daur ulang yang lebih efisien.

---

## 📌 Fitur Utama

- 🔍 **Deteksi & Klasifikasi Real-time**: Mengidentifikasi kategori sampah dari gambar dengan cepat dan akurat.
- 📊 **Confidence Score & Probabilitas Lengkap**: Menampilkan skor keyakinan prediksi teratas beserta visualisasi probabilitas untuk seluruh kategori.
- ⚡ **Ringan & Efisien**: Menggunakan arsitektur MobileNetV2 yang hemat komputasi, mendukung akselerasi GPU (CUDA) maupun CPU secara otomatis.
- 🖥️ **Antarmuka Interaktif**: Tampilan web yang bersih, responsif, dan mudah digunakan berbasis Streamlit.

---

## 🗑️ Kategori Sampah

Model ini dilatih untuk mengklasifikasikan sampah ke dalam **6 kategori**:

| Kategori | Deskripsi Singkat |
| :--- | :--- |
| **Cardboard** | Kardus, karton kemasan, boks paket |
| **Glass** | Botol kaca, toples, pecahan kaca |
| **Metal** | Kaleng minuman, seng, benda berbahan logam |
| **Paper** | Kertas dokumen, koran, majalah |
| **Plastic** | Botol plastik, kantong kresek, wadah makanan plastik |
| **Trash** | Sampah umum / residu yang tidak termasuk kategori daur ulang di atas |

---

## 🏗️ Arsitektur & Spesifikasi Model

- **Base Architecture**: `MobileNetV2` (Pretrained backbone)
- **Classifier Head**: Linear Layer dengan output 6 kelas
- **Input Dimension**: $224 \times 224$ piksel (RGB)
- **Image Normalization**: ImageNet Mean `[0.485, 0.456, 0.406]` & Std `[0.229, 0.224, 0.225]`
- **Framework**: PyTorch (`torch`, `torchvision`)
- **Model Checkpoint**: `waste_classifier_v2_best.pth`

---

## 📁 Struktur Direktori

```plaintext
smart-waste-classification/
├── .gitignore                      # Mengabaikan venv, cache, dan file yang tidak diperlukan
├── app.py                          # Kode utama antarmuka Streamlit & logika inferensi model
├── requirements.txt                # Dependensi pustaka Python
├── waste_classifier_v2_best.pth    # Bobot model PyTorch terlatih
└── README.md                       # Dokumentasi proyek
```

---

## 🚀 Panduan Instalasi & Menjalankan

### 1. Clone Repositori
```bash
git clone https://github.com/blasphemous20/smart-waste-classification.git
cd smart-waste-classification
```

### 2. Buat & Aktifkan Virtual Environment (Opsional tapi Direkomendasikan)
- **Windows (PowerShell)**:
  ```powershell
  python -m venv .venv
  .venv\Scripts\Activate.ps1
  ```
- **Linux / macOS**:
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```

### 3. Install Dependensi
```bash
pip install -r requirements.txt
```

### 4. Jalankan Aplikasi Streamlit
```bash
streamlit run app.py
```

Setelah perintah dijalankan, antarmuka web akan terbuka secara otomatis di browser pada alamat default:
`http://localhost:8501`

---

## 💻 Cara Penggunaan

1. Buka aplikasi web di browser.
2. Klik tombol **"Browse files"** pada bagian *Pilih gambar sampah* (format `.jpg`, `.jpeg`, atau `.png`).
3. Pratinjau gambar akan ditampilkan.
4. Klik tombol **"Klasifikasikan Sampah"**.
5. Sistem akan menampilkan hasil klasifikasi beserta persentase skor confidence dan probabilitas tiap kelas.

---

## 🛠️ Dependensi

- `streamlit`
- `torch`
- `torchvision`
- `Pillow`

---

## 👤 Author

- **Mohammad Fajar Mashum** - [@blasphemous20](https://github.com/blasphemous20)
