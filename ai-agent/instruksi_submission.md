# Panduan dan Instruksi Submission Proyek Akhir Belajar Machine Learning untuk Pemula (BMLP)

---

## 1. Pengantar

Selamat datang pada tahap submission proyek akhir kelas **Belajar Machine Learning untuk Pemula**! Pada proyek ini, Anda akan mengintegrasikan dua pilar utama machine learning: **Unsupervised Learning** dan **Supervised Learning** dalam satu alur kerja terpadu (end-to-end pipeline).

### Alur Kerja Proyek
1. **Tahap 1: Unsupervised Learning (Clustering)**  
   Menerapkan algoritma clustering (K-Means) pada dataset transaksi perbankan tanpa label untuk menemukan pola kelompok nasabah/transaksi dan menghasilkan label kelompok (*cluster labels*). Label ini kemudian digabungkan ke dalam dataset sebagai kolom `Target`.
2. **Tahap 2: Supervised Learning (Klasifikasi)**  
   Menggunakan dataset yang kini telah berlabel `Target` dari tahap clustering untuk membangun model klasifikasi (Decision Tree dan model pembanding) yang mampu memprediksi kelompok target berdasarkan fitur-fitur transaksi.

### Panduan Dataset
- **Sumber Data:** Dataset merupakan modifikasi dari *Kaggle Bank Transaction Dataset for Fraud Detection*.
- **Akses Data:** Dataset disediakan melalui Google Drive / Sheets publik yang dapat dibaca secara otomatis via URL:  
  `https://docs.google.com/spreadsheets/d/e/2PACX-1vTbg5WVW6W3c8SPNUGc3A3AL-AG32TPEQGpdzARfNICMsLFI0LQj0jporhsLCeVhkN5AoRsTkn08AYl/pub?output=csv`  
  atau menggunakan file lokal `bank_transactions_data_edited.csv`.
- **Dimensi Data:** 2.512 baris sampel dengan 16 atribut (informasi transaksi, akun, demografi, saldo, dan waktu).

### Aturan Pokok Pengerjaan
1. **Gunakan Template Notebook:** Wajib menggunakan file template `.ipynb` yang disediakan dengan format *fill in the blanks* (melengkapi kode pada bagian `________`).
2. **Pustaka & Dependencies:** Dilarang menambahkan impor pustaka baru di luar cell `Import Library` yang sudah disediakan.
3. **Versi Lingkungan:** Disarankan menggunakan `scikit-learn` versi 1.7.x agar kompatibel dengan Yellowbrick dan sistem evaluasi otomatis Dicoding.
4. **Integritas Variabel & Cell:** Gunakan nama variabel `df` secara konsisten sesuai petunjuk template, jangan mengubah judul/header markdown bawaan, dan jangan menambahkan cell code yang tidak diperintahkan.
5. **Kualitas Dokumentasi:** Pada bagian analisis deskriptif dan interpretasi cluster, sertakan:
   - *Metode yang digunakan*
   - *Alasan pemilihan metode*
   - *Interpretasi hasil yang didapatkan*

---

## 2. Kriteria Utama

Proyek ini dinilai berdasarkan 5 kriteria utama dengan tingkatan pencapaian: **Rejected (0 pts)**, **Basic (2 pts)**, **Skilled (3 pts)**, dan **Advanced (4 pts)**.

```
┌────────────────────────────────────────────────────────────────────────┐
│                        5 KRITERIA UTAMA SUBMISSION                     │
├────────────────────────┬───────────────────────────────────────────────┤
│ Kriteria 1 (Clustering)│ Memuat Dataset dan Exploratory Data Analysis  │
│ Kriteria 2 (Clustering)│ Pembersihan dan Pra-pemrosesan Data           │
│ Kriteria 3 (Clustering)│ Membangun Model Clustering                    │
│ Kriteria 4 (Clustering)│ Interpretasi Hasil Clustering & Ekspor Data   │
│ Kriteria 5 (Klasifikasi)│ Membangun & Mengevaluasi Model Klasifikasi   │
└────────────────────────┴───────────────────────────────────────────────┘
```

---

### Kriteria 1: Memuat Dataset dan Melakukan Exploratory Data Analysis (EDA)
- **Basic (2 pts):**
  - Menampilkan 5 baris pertama dataset menggunakan `.head()`.
  - Menampilkan ringkasan struktur dan tipe data menggunakan `.info()`.
  - Menampilkan ringkasan statistik deskriptif menggunakan `.describe()`.
  - *Catatan Penting:* Hindari membungkus fungsi-fungsi di atas dengan `print()` atau `display()` agar output standar tidak hilang.
- **Skilled (3 pts):**
  - Memenuhi seluruh kriteria Basic.
  - Menampilkan visualisasi matriks korelasi (*heatmap*) antar fitur numerik.
  - Menampilkan histogram untuk seluruh fitur numerik dataset.
- **Advanced (4 pts):**
  - Memenuhi seluruh kriteria Skilled.
  - Visualisasi data informatif dan rapi: label sumbu tidak tumpang tindih (*no label overlap*), rotasi sumbu diatur dengan baik (misal `plt.xticks(rotation=45)` pada boxplot per pekerjaan nasabah).

---

### Kriteria 2: Pembersihan dan Pra-pemrosesan Data
- **Basic (2 pts):**
  - Mengecek nilai yang hilang dengan `.isnull().sum()` dan menghapusnya menggunakan `.dropna(inplace=True)`.
  - Mengecek duplikasi dengan `.duplicated().sum()` dan menghapusnya menggunakan `.drop_duplicates(inplace=True)`.
  - Menghapus kolom non-prediktif/identitas yang mengandung 'ID', 'Address', dan 'Date' (`TransactionID`, `AccountID`, `TransactionDate`, `DeviceID`, `IP Address`, `MerchantID`, `PreviousTransactionDate`).
  - Melakukan *feature encoding* pada seluruh kolom kategorikal bertipe object menggunakan `LabelEncoder()`.
  - Menampilkan daftar fitur akhir menggunakan `.columns.tolist()`.
- **Skilled (3 pts):**
  - Memenuhi seluruh kriteria Basic.
  - Menangani pencilan (*outliers*) pada fitur numerik menggunakan metode IQR (Interquartile Range) dengan membuang data di luar batas `[Q1 - 1.5*IQR, Q3 + 1.5*IQR]`.
  - Melakukan standarisasi/skala fitur numerik menggunakan `StandardScaler()`.
- **Advanced (4 pts):**
  - Memenuhi seluruh kriteria Skilled.
  - Melakukan *data binning* pada 1–2 fitur numerik (misal membagi `CustomerAge` menjadi 3 grup kuantil dengan `pd.qcut()`), lalu mengenkode hasil binning tersebut menggunakan `LabelEncoder()`.

---

### Kriteria 3: Membangun Model Clustering
- **Basic (2 pts):**
  - Menggunakan dataset yang telah melalui tahap pra-pemrosesan.
  - Menentukan jumlah cluster optimal menggunakan Elbow Method dengan `KElbowVisualizer()` (metric silhouette).
  - Membangun model K-Means Clustering dengan `sklearn.cluster.KMeans(n_clusters=..., random_state=42)`.
  - Menyimpan model clustering ke dalam berkas `model_clustering.h5` menggunakan `joblib.dump()`.
- **Skilled (3 pts):**
  - Memenuhi seluruh kriteria Basic.
  - Menghitung dan mencetak nilai *Silhouette Score* model K-Means.
  - Membuat visualisasi 2D hasil clustering menggunakan reduksi dimensi PCA (`n_components=2`) dengan sebaran centroid.
- **Advanced (4 pts):**
  - Memenuhi seluruh kriteria Skilled.
  - Membangun model K-Means baru yang dilatih langsung pada data hasil reduksi dimensi PCA 2 komponen (`data_final`).
  - Menyimpan model PCA clustering tersebut ke dalam berkas `PCA_model_clustering.h5` menggunakan `joblib.dump()`.

---

### Kriteria 4: Interpretasi Hasil Clustering & Ekspor Data
- **Basic (2 pts):**
  - Menampilkan ringkasan statistik deskriptif per cluster (minimal `mean`, `min`, `max`) untuk seluruh fitur numerik menggunakan `.groupby('Cluster')[numerical_cols].agg(['mean', 'min', 'max'])`.
  - Memberikan penjelasan persona dan karakteristik tiap cluster berdasarkan nilai hasil agregasi (dalam kondisi terstandarisasi/scaled).
  - Mengubah nama kolom cluster menjadi `Target` (`.rename(columns={"Cluster": "Target"})`).
  - Mengekspor data hasil clustering ke berkas `data_clustering.csv`.
- **Skilled (3 pts):**
  - Memenuhi seluruh kriteria Basic.
  - Mengembalikan nilai numerik ke skala aslinya menggunakan `scaler.inverse_transform()`.
  - Mengembalikan nilai kategorikal ke label string aslinya menggunakan `encoder.inverse_transform()`.
  - Menampilkan analisis deskriptif data asli: numerik (`mean`, `min`, `max`) dan kategorikal (`mode`).
  - Menjelaskan kembali karakteristik dan persona tiap cluster berdasarkan nilai skala aslinya.
- **Advanced (4 pts):**
  - Memenuhi seluruh kriteria Skilled.
  - Memverifikasi integrasi data hasil inverse yang tetap menyertakan kolom `Target`.
  - Menyimpan data hasil inverse tersebut ke berkas `data_clustering_inverse.csv`.

---

### Kriteria 5: Membangun dan Mengevaluasi Model Klasifikasi
- **Basic (2 pts):**
  - Memuat dataset hasil clustering yang memuat kolom `Target` (gunakan `data_clustering_inverse.csv` jika menerapkan kriteria Advanced, lalu lakukan One-Hot Encoding dengan `pd.get_dummies()`).
  - Memisahkan data menjadi training set dan test set menggunakan `train_test_split(..., test_size=0.2, random_state=42, stratify=y)`.
  - Membangun dan melatih model klasifikasi `DecisionTreeClassifier(random_state=42)`.
  - Menyimpan model ke berkas `decision_tree_model.h5` menggunakan `joblib.dump()`.
- **Skilled (3 pts):**
  - Memenuhi seluruh kriteria Basic.
  - Melatih minimal satu algoritma klasifikasi scikit-learn tambahan selain Decision Tree (misal `RandomForestClassifier(random_state=42)`).
  - Menampilkan metrik evaluasi lengkap (*accuracy*, *precision*, *recall*, *F1-score*) untuk kedua model pada test set menggunakan `classification_report()`.
  - Menyimpan model pembanding ke berkas `explore_<Nama Algoritma>_classification.h5` (misal `explore_RandomForest_classification.h5`).
- **Advanced (4 pts):**
  - Memenuhi seluruh kriteria Skilled.
  - Melakukan *hyperparameter tuning* (menggunakan `GridSearchCV` atau `RandomizedSearchCV`) pada salah satu model klasifikasi.
  - Menampilkan evaluasi performa model terbaik hasil tuning menggunakan `classification_report()`.
  - Menyimpan model terbaik hasil tuning ke berkas `tuning_classification.h5`.

---

## 3. Ketentuan Penilaian

### Formula Perhitungan Nilai Akhir
$$\text{Nilai Akhir} = \frac{\text{Total Poin}}{\text{Jumlah Kriteria (5)}}$$

Syarat lulus adalah mendapatkan minimal nilai 2.00 (tanpa ada satupun kriteria yang rejected/0 pts).

### Skala Nilai & Tingkat Kelulusan

| Nilai Akhir | Rating Dicoding | Nilai Huruf | Level of Mastery | Predikat | Keterangan |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **< 1.0** | Rejected | E | - | Tidak Lulus | Belum memenuhi kompetensi minimal. |
| **1.0 – < 2.0** | Bintang 2 | D | Below Basic | Kurang | Memenuhi sebagian kompetensi minimal, perlu perbaikan. |
| **2.0 – < 3.0** | Bintang 3 | C | Basic | Cukup | Memenuhi seluruh kompetensi dasar (*passing grade*). |
| **3.0 – < 4.0** | Bintang 4 | B | Skilled | Mahir | Memenuhi seluruh kriteria dengan baik/mahir. |
| **4.0** | Bintang 5 | A | Advanced | Sangat Baik | Menguasai seluruh kriteria tingkat lanjut dengan sempurna. |

### Pelanggaran yang Menyebabkan Otomatis Ditolak (Rejected)
1. Tidak menyertakan berkas wajib yang disyaratkan.
2. Tidak menggunakan template notebook resmi yang disediakan.
3. Menambahkan baris kode (*line code*) atau sel kode (*cell code*) di luar instruksi.
4. Mengubah nama variabel `df` atau merusak struktur hierarki template.
5. Tidak menyertakan penjelasan persona/karakteristik cluster pada bagian interpretasi.
6. Tidak menggunakan label `Target` hasil clustering pada notebook klasifikasi.
7. Model klasifikasi tidak mencetak evaluasi akurasi dan F1-Score pada test set.
8. Menggunakan library/metode AutoML (seperti PyCaret, Auto-sklearn, TPOT, H2O, dsb.).
9. Terdapat cell yang error atau tidak memiliki output yang diharapkan saat diperiksa reviewer.

### Ketentuan Proses Review
- Waktu peninjauan review maksimal 3 hari kerja (tidak termasuk Sabtu, Minggu, dan hari libur nasional).
- Jangan melakukan submit berulang kali sebelum review selesai agar tidak memperpanjang antrean evaluasi.
- Notifikasi hasil review akan dikirimkan melalui email dan dashboard akun Dicoding.

---

## 4. Tips and Trick

### Urutan Pengerjaan Notebook Clustering
1. **Load Data:** Pastikan URL terbaca sempurna menjadi DataFrame `df`.
2. **Standard Output EDA:** Panggil langsung `df.head()`, `df.info()`, dan `df.describe()` di cell masing-masing tanpa dibungkus `print()` atau `display()`.
3. **Pencegahan Label Overlap:** Pada plot korelasi dan boxplot kategorikal, atur ukuran figure secara proporsional dan selalu sertakan rotasi label: `plt.xticks(rotation=45)`.
4. **Data Cleaning:** Selalu gunakan `inplace=True` pada `dropna()` dan `drop_duplicates()`, kemudian ikuti dengan `.isnull().sum()` dan `.duplicated().sum()` untuk membuktikan data telah bersih.
5. **Konsistensi Feature Scaling & Inverse:** Pastikan dictionary `encoders` dan objek `scaler` tersimpan dengan baik agar proses `inverse_transform` numerik dan kategorikal berjalan mulus.
6. **Interpretasi Komprehensif:** Tuliskan analisis persona yang masuk akal bisnis (misal membedakan nasabah berdasarkan frekuensi transaksi, volume nominal belanja, atau tingkat saldo).

### Urutan Pengerjaan Notebook Klasifikasi
1. **Sinkronisasi File:** Jika menyelesaikan kriteria Advanced di clustering, gunakan `data_clustering_inverse.csv` pada notebook klasifikasi.
2. **One-Hot Encoding:** Konversi seluruh kolom kategorikal teks menjadi representasi numerik menggunakan `pd.get_dummies(..., drop_first=True)` sebelum *data splitting*.
3. **Stratifikasi Data:** Selalu gunakan argumen `stratify=y` pada `train_test_split` agar proporsi kelas target seimbang di antara training dan test set.
4. **Evaluasi Berstandar:** Gunakan `classification_report(y_test, y_pred)` yang secara langsung mencakup Precision, Recall, F1-Score, dan Accuracy.
5. **Model Naming:** Perhatikan penamaan file `.h5` agar sesuai format yang ditentukan (misal `explore_RandomForest_classification.h5` dan `tuning_classification.h5`).
6. **Eksekusi Run All:** Sebelum menyimpan dan mengumpulkan, lakukan *Restart Kernel & Run All Cells* untuk memastikan tidak ada celah error atau sel yang terlewat.

---

## 5. Ketentuan Berkas Submission

Seluruh berkas submission dikemas ke dalam satu file arsip zip dengan format penamaan:  
**`BMLP_Nama-siswa.zip`**

### Struktur Berkas Submission (Target Bintang 5 / Advanced)

```text
BMLP_Nama-siswa.zip
├── [Clustering]_Submission_Akhir_BMLP_Your_Name.ipynb       # Notebook Clustering (Wajib, terisi dan ada output)
├── [Klasifikasi]_Submission_Akhir_BMLP_Your_Name.ipynb      # Notebook Klasifikasi (Wajib, terisi dan ada output)
├── model_clustering.h5                                      # Model K-Means Clustering (Wajib)
├── PCA_model_clustering.h5                                  # Model K-Means berbasis PCA (Opsional - Advanced)
├── decision_tree_model.h5                                   # Model Decision Tree Klasifikasi (Wajib)
├── explore_RandomForest_classification.h5                   # Model Klasifikasi Alternatif (Opsional - Skilled)
├── tuning_classification.h5                                 # Model Klasifikasi Hasil Tuning (Opsional - Advanced)
├── data_clustering.csv                                      # Dataset hasil clustering terstandarisasi (Wajib)
└── data_clustering_inverse.csv                              # Dataset hasil clustering ter-inverse (Opsional - Advanced)
```

> **Perhatian:** Seluruh berkas notebook `.ipynb` yang diserahkan harus dalam keadaan sudah selesai dieksekusi (seluruh cell menampilkan output visualisasi dan metrik lengkap) tanpa memerlukan reviewer mengeksekusi ulang dari awal.