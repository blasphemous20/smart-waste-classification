
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

from pathlib import Path

# Pengaturan halaman aplikasi
st.set_page_config(
    page_title="Smart Waste Classification",
    page_icon="♻️",
    layout="centered"
)

# Daftar kelas harus sama dengan urutan saat training
CLASS_NAMES = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash"
]

MODEL_PATH = Path(__file__).parent / "waste_classifier_v2_best.pth"

# Gunakan GPU jika tersedia, jika tidak gunakan CPU
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Transformasi gambar harus sama dengan validasi saat training
IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


@st.cache_resource
def load_model():
    # Periksa apakah file model tersedia
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"File model tidak ditemukan: {MODEL_PATH}"
        )

    # Buat arsitektur MobileNetV2
    model = models.mobilenet_v2(weights=None)

    # Sesuaikan layer output menjadi enam kelas
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(
        in_features,
        len(CLASS_NAMES)
    )

    # Muat bobot hasil training dari Colab
    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    # Mendukung checkpoint berbentuk dictionary
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    return model


def predict_image(image, model):
    # Pastikan gambar memiliki tiga channel RGB
    image = image.convert("RGB")

    # Ubah gambar menjadi tensor dengan batch dimension
    image_tensor = IMAGE_TRANSFORM(image).unsqueeze(0)
    image_tensor = image_tensor.to(DEVICE)

    # Jalankan prediksi tanpa menghitung gradien
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)[0]

    # Ambil kelas dengan probabilitas tertinggi
    predicted_index = probabilities.argmax().item()

    predicted_class = CLASS_NAMES[predicted_index]
    confidence = probabilities[predicted_index].item()

    # Siapkan semua probabilitas untuk ditampilkan
    all_probabilities = {
        CLASS_NAMES[i]: probabilities[i].item()
        for i in range(len(CLASS_NAMES))
    }

    return predicted_class, confidence, all_probabilities


# Header aplikasi
st.title("♻️ Smart Waste Classification")
st.write(
    "Unggah gambar sampah untuk memprediksi kategorinya "
    "menggunakan model AI MobileNetV2."
)

st.info(
    "Kategori yang tersedia: cardboard, glass, metal, "
    "paper, plastic, dan trash."
)

# Siapkan model
try:
    model = load_model()
except Exception as error:
    st.error(f"Gagal memuat model: {error}")
    st.stop()

# Komponen upload gambar
uploaded_file = st.file_uploader(
    "Pilih gambar sampah",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")

        st.subheader("Gambar yang diunggah")
        st.image(
            image,
            caption=uploaded_file.name,
            use_container_width=True
        )

        if st.button("Klasifikasikan Sampah", type="primary"):
            with st.spinner("AI sedang menganalisis gambar..."):
                predicted_class, confidence, probabilities = (
                    predict_image(image, model)
                )

            # Hasil prediksi utama
            st.subheader("Hasil Prediksi")
            st.success(
                f"Kategori sampah: {predicted_class.upper()}"
            )

            st.metric(
                label="Confidence Score",
                value=f"{confidence * 100:.2f}%"
            )

            st.progress(confidence)

            # Tampilkan skor semua kelas
            st.subheader("Probabilitas Setiap Kategori")

            sorted_probabilities = sorted(
                probabilities.items(),
                key=lambda item: item[1],
                reverse=True
            )

            for class_name, probability in sorted_probabilities:
                st.write(
                    f"**{class_name.capitalize()}** — "
                    f"{probability * 100:.2f}%"
                )
                st.progress(probability)

            st.caption(
                "Confidence adalah skor model, bukan jaminan "
                "bahwa hasil klasifikasi selalu benar."
            )

    except Exception as error:
        st.error(f"Gambar gagal diproses: {error}")

else:
    st.caption("Format gambar yang didukung: JPG, JPEG, dan PNG.")