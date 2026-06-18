# Proyek Klasifikasi Gambar: Fashion-MNIST

- **Nama:** Muslichin
- **Email:** muslichin.ach@gmail.com
- **ID Dicoding:** muslchn

## Ringkasan

Proyek ini membangun model Convolutional Neural Network untuk mengklasifikasikan gambar pakaian dari dataset open-source Fashion-MNIST. Dataset berasal dari Zalando Research dan dapat direproduksi melalui `tf.keras.datasets.fashion_mnist`.

Sumber dataset: <https://github.com/zalandoresearch/fashion-mnist>

Dataset berisi 70.000 gambar grayscale berukuran 28x28 piksel dengan 10 kelas:

1. `t-shirt_top`
2. `trouser`
3. `pullover`
4. `dress`
5. `coat`
6. `sandal`
7. `shirt`
8. `sneaker`
9. `bag`
10. `ankle_boot`

Dataset disiapkan ulang menjadi folder gambar PNG berdasarkan kelas, lalu dibagi menjadi train, validation, dan test set:

- Train: 48.000 gambar
- Validation: 12.000 gambar
- Test: 10.000 gambar

## Hasil Terakhir

Notebook dan script sudah dijalankan ulang secara berurutan. Hasil evaluasi terakhir:

- Training accuracy: 0.9374
- Testing accuracy: 0.9140
- Training loss: 0.1737
- Testing loss: 0.2338

Model menggunakan arsitektur `Sequential` dengan layer `Conv2D`, `MaxPooling2D`, `Dropout`, `Flatten`, dan `Dense`.

## Struktur Submission

```text
klasifikasi-gambar
├── tfjs_model/
│   ├── group1-shard*.bin
│   └── model.json
├── tflite/
│   ├── model.tflite
│   └── label.txt
├── saved_model/
│   ├── saved_model.pb
│   └── variables/
├── notebook.ipynb
├── submission.py
├── README.md
└── requirements.txt
```

## Cara Menjalankan Ulang

Gunakan Python sesuai `requirements.txt`, lalu jalankan:

```bash
pip install -r requirements.txt
python submission.py
```

Script akan:

1. Mengunduh dan menyiapkan dataset Fashion-MNIST sebagai folder gambar PNG.
2. Membagi dataset menjadi train, validation, dan test set.
3. Melatih model Sequential CNN.
4. Menampilkan plot akurasi dan loss.
5. Mengevaluasi akurasi training dan testing.
6. Menyimpan model ke format SavedModel, TensorFlow Lite, dan TensorFlow.js.
7. Menjalankan inference menggunakan model TensorFlow Lite.

## Catatan Submission

Folder `dataset/` dan `dataset_split/` tidak perlu disertakan karena dapat dibuat ulang dari notebook/script. Folder model `saved_model/`, `tflite/`, dan `tfjs_model/` perlu disertakan setelah notebook/script dijalankan.
