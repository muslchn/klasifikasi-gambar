# Proyek Klasifikasi Gambar: Geometric Shapes

- **Nama:** Muslichin
- **Email:** muslichin.ach@gmail.com
- **ID Dicoding:** muslchn

## Ringkasan

Proyek ini membuat model Convolutional Neural Network untuk mengklasifikasikan gambar geometri sintetis menjadi tiga kelas:

1. `circle`
2. `square`
3. `triangle`

Dataset dibuat secara reproducible dari kode Python sehingga tidak bergantung pada unduhan eksternal. Total dataset adalah 12.000 gambar dengan resolusi asli yang bervariasi, lalu dibagi menjadi train, validation, dan test set dengan rasio 70:15:15.

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

## Cara Menjalankan

Disarankan menjalankan notebook di Google Colab dengan GPU T4.

```bash
pip install -r requirements.txt
python submission.py
```

Script akan:

1. Membuat dataset gambar geometri jika belum tersedia.
2. Membagi dataset menjadi train, validation, dan test set.
3. Melatih model Sequential CNN dengan Conv2D dan pooling layer.
4. Menampilkan plot akurasi dan loss.
5. Mengevaluasi akurasi training dan testing.
6. Menyimpan model ke format SavedModel, TFLite, dan TensorFlow.js.
7. Melakukan inference menggunakan model TFLite.

## Catatan

Dataset dan folder hasil split tidak wajib dikirim karena dapat dibuat ulang dari notebook/script. Folder model hasil konversi perlu disertakan setelah notebook dijalankan.
