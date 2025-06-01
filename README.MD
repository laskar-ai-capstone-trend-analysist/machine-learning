# Machine Learning Model

## 📁 Project Structure
```
data_preprocessing/
├── dataset/ #dataset raw
├── dataset_preprocessed/ #dataset sudah di preprocess, namun tidak distem
├── dataset_preprocessed_v2/ #dataset sudah di preprocess dan sudah distem
├── preprocessing_product.ipynb #preprocessing pada table product list tiap kategori
└── preprocessing_review.ipynb #preprocessing pada review content tiap produk

modelling/
├── mlartifacts/ # MLflow artifacts (saved models, logs)
├── mlruns/ # MLflow run tracking data
├── config.yaml # Hyperparameters for training
├── custom_layers.py # Custom Keras layers
├── modelling_mlflow.py # Script to train and log model with MLflow
├── modelling.ipynb # Jupyter notebook for interactive training
├── predict.py # Script for sentiment prediction from text input
├── requirements.txt # Python dependencies
├── sentiment_model.keras # Trained model file (saved in Keras format)
└── utils.py # Utilities: data loading, balancing, tokenizing
```

## ⚙️ Persyaratan

- 🐍 Python 3.10


## 🚀 Instalasi

1. Unduh atau clone repositori ini:

   ```bash
   git clone https://github.com/laskar-ai-capstone-trend-analysist/machine-learning.git
   cd modelling
   ```

2. Instal semua dependensi yang dibutuhkan:

   ```bash
   pip install -r requirements.txt
   ```

## 🎮 Cara Penggunaan

### Melihat Model yang Ada
- Jalankan ```mlflow ui --port 5000``` pada terminal
- Buka ```http://localhost/5000``` pada browser untuk melihat model yang sudah ada
- Pada bagian 'Model', sudah terdapat 2 versi model **sentiment**. Version 2 adalah versi terbaik dari model yang sudah menggunakan dataset terbaru

### Tuning dan Tracking
 - Jalankan ```mlflow ui --port 5000``` pada terminal
 - Ubah parameter pada file ```config.yaml``` (note: apabila ingin mengubah parameter yang lebih spesifik (batch_size, epoch, dll.), anda dapat mengubahnya langsung di ```modelling_mlflow.py```)
 - Jalankan ```python modelling_mlflow.py``` pada terminal yang beda
 - Buka ```http://localhost/5000``` pada browser untuk tracking model

### Menyimpan Model
Setelah menemukan model terbaik, anda dapat menyimpan model tersebut menggunakan file ```modelling.ipynb```. Atur ```config.yaml``` menjadi parameter sesuai dengan model terbaik, lalu lakukan **run all** pada ```modelling.ipynb```.

### Melakukan Prediksi
Untuk contoh prediksi sudah terdapat pada file ```predict.py``` dan pada bagian **Testing** pada ```modelling.ipynb```.

- Pertama-tama, kita harus membuat tokenizer menggunakan seluruh dataset yang ada dengan fungsi ``data_prep()``

- Setelah itu, teks yang ingin diprediksi harus dimasukan ke fungsi ``preprocess_text()`` untuk mendapat teks yang lebih bersih

- Lalu, gunakan tokenizer sebelumnya pada teks yang sudah bersih.
```bash
# Convert to sequence using the same tokenizer
test_seq = tokenizer.texts_to_sequences([text_testing2])

# Pad it to the same max length
test_pad = pad_sequences(test_seq, maxlen, padding='post')
```

- Terakhir, anda dapat melakukan prediksi seperti berikut.
```bash
prediction = loaded_model.predict(test_pad)

# Get class with highest probability
predicted_class = np.argmax(prediction, axis=1)[0]

print(f"Text: {text_testing2}")
print(f"Predicted label index: {predicted_class}")
```
