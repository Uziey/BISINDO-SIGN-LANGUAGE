# Dataset alfabet sendiri: satu dan dua tangan

Dua file Python, simpan berdampingan:
- data_training.py: pengambilan foto dan training otomatis.
- deteksi_bisindo.py: membuka kamera untuk mencoba model; mengimpor fungsi dari file pertama TANPA menjalankan pengambilan foto.

## Instalasi Windows / VS Code
Gunakan Python 3.11 64-bit. Ekstrak ZIP, buka folder bisindo_dua_file di VS Code, lalu terminal:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
```
Pilih interpreter .venv melalui Python: Select Interpreter. Klik Run Python File in Terminal (bukan Code Runner) pada file yang ingin dijalankan. Perintah `python` di bawah harus memakai interpreter yang sama; bila belum aktif, gunakan `.venv\Scripts\python.exe` sebagai pengganti `python`.

## 1. Atur huruf dan jumlah foto
Di bagian atas data_training.py:

```python
HURUF = "A"                 # Ganti A, B, C, ... Z
JUMLAH_FOTO = 100            # 40 / 50 / 100 foto BARU per sesi
JUMLAH_TANGAN = 2           # Pilih 1 atau 2 sesuai pose yang diajarkan
TRAIN_SETELAH_REKAM = True
CAMERA_INDEX = 0
INTERVAL_FOTO = 0.5
```

Jumlah tangan bukan pemetaan alfabet bawaan. Anda menentukan pose berdasarkan referensi BISINDO yang Anda gunakan. Semua label dan contoh alfabet berasal dari foto Anda; tidak ada dataset alfabet pihak lain dalam paket ini.

## 2. Jalankan file training

```powershell
python data_training.py
```

Pada run pertama, model skeleton Google diunduh otomatis satu kali (internet diperlukan). Ini model pendeteksi titik tangan pralatih, bukan model pengenal alfabet. Sesudah instalasi dan unduhan selesai, proses dapat berjalan lokal.

1. Kamera terbuka dalam kondisi jeda.
2. Bentuk pose sesuai HURUF, pastikan kedua tangan masuk frame jika JUMLAH_TANGAN=2.
3. Klik jendela kamera, tekan S untuk mulai. Tekan S lagi untuk jeda.
4. Program memfoto setiap 0,5 detik hingga jumlah tercapai. Q mengakhiri sesi lebih awal; foto yang sudah tersimpan tetap dipakai untuk training.
5. Kamera menutup, lalu model otomatis dilatih ulang dari SEMUA kelas yang sudah dikumpulkan.

Foto hanya disimpan jika jumlah tangan sesuai dan identitas kiri/kanan tidak ambigu. Jika memilih 2 tetapi hanya 1 yang terlihat, hitungan tidak bertambah. Tangan bertumpuk/tertutup dapat membuat detektor gagal; atur pencahayaan dan sudut tanpa mengubah pose huruf.

Foto bersih tanpa coretan skeleton disimpan sebagai `dataset/A/<sesi>/A_0001.jpg` dan seterusnya. session.json menyimpan jumlah tangan yang diharapkan. Jangan hapus atau ubah metadata ini sembarangan. Run selanjutnya membuat sesi baru sehingga tidak menimpa foto sebelumnya.

## 3. Lanjutkan B sampai Z
Ubah HURUF, sesuaikan JUMLAH_TANGAN, lalu Run lagi. Tidak perlu mengganti nama file Python.

Alternatif tanpa mengedit kode:
```powershell
python data_training.py --label A --count 100 --hands 2
python data_training.py --label B --count 100 --hands 2
```
Contoh tersebut mengasumsikan pose yang Anda rekam menggunakan dua tangan. Gunakan --hands 1 untuk pose satu tangan. Parameter terminal mengalahkan pengaturan di kode.

Saat baru A, foto tetap tersimpan tetapi model belum dibuat. Model memerlukan minimal 2 kelas dan 10 foto valid per kelas; targetkan 50-100 atau lebih. Setiap kelas baru ditambahkan, model diperbarui otomatis. Terminal menampilkan kelas tersedia dan huruf A-Z yang belum ada.

Ambil 100 foto per huruf sebagai 2 sesi x 50 dengan variasi jarak, posisi, dan pencahayaan. Gunakan bentuk huruf konsisten. Untuk pengguna tangan dominan berbeda, tambahkan contoh representatif dan evaluasi terpisah; jangan berharap otomatis mengenali semua variasi.

## 4. Jalankan file deteksi
Setelah model terbentuk, buka deteksi_bisindo.py lalu Run, atau:

```powershell
python deteksi_bisindo.py
```

Kamera menampilkan skeleton satu/dua tangan dan huruf yang diprediksi. Tidak ada proses pengambilan dataset atau training pada file ini. Program memakai model_bisindo.joblib. Q untuk keluar.

Prediksi ditampilkan jika label sama selama 5 frame dan skor >= 0,70. Prediksi ditolak bila jumlah tangan tidak sesuai contoh kelas tersebut. Skor adalah keluaran model, bukan probabilitas akurasi terjamin; pose asing masih bisa salah diterima. Opsional kumpulkan pose nonalfabet dengan label UNKNOWN (variasikan banyak bentuk), lalu training ulang.

## Training ulang tanpa kamera
Setelah memperbaiki data atau menghapus foto salah:

```powershell
python data_training.py --train-only
```

Model lama diganti hanya setelah fit berhasil. Bila training gagal, model lama mungkin masih tersedia dan belum mencerminkan dataset terbaru: periksa pesan terminal. File model dari paket satu tangan sebelumnya tidak kompatibel. Gunakan folder paket baru dan rekam ulang dataset dengan metadata sesi versi ini.

## Bagaimana dua tangan direpresentasikan
MediaPipe diatur num_hands=2. Ada 128 fitur tetap: mask kehadiran kiri/kanan dan 21 titik x 3 koordinat untuk setiap tangan. Slot kiri/kanan berdasarkan handedness, bukan urutan hasil detektor. XY relatif titik tengah pergelangan dengan skala telapak bersama, sehingga jarak relatif antar tangan tetap terekam. Z relatif pergelangan masing-masing, bukan pengukuran kedalaman global antar tangan. Ketika satu tangan tidak ada, slotnya diisi nol. Fitur dan mirror kamera sama saat pengambilan, training, dan deteksi.

Jika kiri/kanan ambigu atau kedua deteksi berlabel sisi yang sama, frame ditolak. Ini dapat mengurangi jumlah foto yang tersimpan, tetapi menghindari pertukaran fitur secara diam-diam.

## Evaluasi dan batasan
Jika tiap kelas memiliki >=2 sesi valid, sesi terakhir per kelas menjadi holdout evaluasi. Setelah evaluasi, model final di-fit ulang pada semua data. training_report.json adalah hasil evaluasi sebelum refit; jika belum cukup sesi, akurasi tidak diklaim. Jangan menganggap akurasi foto dari orang/sesi yang sama berlaku untuk semua pengguna.

Ini prototipe pengenal pose alfabet BISINDO berdasarkan contoh yang Anda berikan, bukan penerjemah BISINDO lengkap atau validator ketepatan pose. Sistem tidak menetapkan satu kamus A-Z. Jika isyarat perlu gerakan untuk dibedakan, foto statis tidak cukup: perlu dataset urutan video dan model temporal. Skeleton tangan juga tidak merepresentasikan ekspresi wajah/tubuh.

## Troubleshooting
- Kamera gagal: tutup Zoom/Teams/Kamera, periksa izin, ubah CAMERA_INDEX menjadi 1 atau gunakan --camera 1.
- Jumlah foto tidak bertambah: tekan S; cek jumlah tangan di preview; kedua tangan harus cukup terlihat dan handedness tidak ambigu.
- Tertukar huruf: periksa label dan pose, tambah sesi dengan variasi wajar, latih ulang. 100 foto identik kurang membantu.
- Modul tidak ditemukan: instal requirements dengan interpreter yang sama dengan Run.
- Jangan menamai file mediapipe.py, cv2.py, numpy.py, atau sklearn.py.
- Jangan pindahkan deteksi_bisindo.py sendirian: file ini membutuhkan data_training.py, hand_landmarker.task, dan model_bisindo.joblib dalam folder yang sama.
- Hanya buka joblib buatan sendiri/tepercaya.

## Verifikasi paket
Lulus pemeriksaan sintaks dan CLI, import tanpa membuka kamera, serta uji fitur sintetis: panjang vektor, urutan tangan, tangan hilang, penolakan handedness ambigu, translasi, dan sensitivitas jarak antar tangan. Kamera nyata, ketepatan alfabet, serta instalasi Windows belum diuji. Dependensi berupa rentang versi, bukan lockfile tervalidasi seluruh kombinasi.

Rujukan API: https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/python
