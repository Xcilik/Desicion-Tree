# 🚢 Titanic Decision Tree Classifier (Dari Nol / Scratch)

Repositori ini berisi dokumentasi dan penjelasan lengkap mengenai implementasi **Decision Tree Classifier** yang dibangun secara manual dari nol (scratch) tanpa menggunakan pustaka *machine learning* pihak ketiga (seperti `scikit-learn`). Kode ini digunakan untuk memprediksi probabilitas keselamatan penumpang kapal Titanic berdasarkan data historis.

Proyek ini sangat berguna untuk memahami dasar-dasar matematis, struktur data pohon (tree), serta logika di balik pembuatan keputusan berbasis **Gini Impurity** pada algoritma Machine Learning berbasis pohon.

---

## 📋 Daftar Isi
1. [Prasyarat & Pustaka yang Digunakan](#1-prasyarat--pustaka-yang-digunakan)
2. [Eksplorasi & Memuat Dataset](#2-eksplorasi--memuat-dataset)
3. [Pra-pemrosesan Data (Data Preprocessing)](#3-pra-pemrosesan-data-data-preprocessing)
4. [Pemisahan Data Manual (Train-Test Split)](#4-pemisahan-data-manual-train-test-split)
5. [Arsitektur Kelas Node (`Node`)](#5-arsitektur-kelas-node-node)
6. [Algoritma Decision Tree Classifier (`DecisionTree`)](#6-algoritma-decision-tree-classifier-decisiontree)
7. [Pelatihan Model & Evaluasi](#7-pelatihan-model--evaluasi)
8. [Perhitungan Confusion Matrix secara Manual](#8-perhitungan-confusion-matrix-secara-manual)
9. [Analisis Kepentingan Fitur (Feature Importance)](#9-analisis-kepentingan-fitur-feature-importance)
10. [Kesimpulan](#10-kesimpulan)

---

## 1. Prasyarat & Pustaka yang Digunakan

Pada bagian awal kode, diimpor beberapa pustaka Python dasar untuk manipulasi data, perhitungan matematika sederhana, dan visualisasi:

```python
import pandas as pd
import random
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.colors import ListedColormap
```

### Konfigurasi Penting:
* **Reproduksibilitas (Random Seed)**: Diatur `random.seed(42)` dan `np.random.seed(42)` agar proses pengacakan data (split dataset) selalu menghasilkan pembagian yang sama setiap kali dijalankan.
* **Desain Visualisasi**: Menggunakan tema `whitegrid` kustom dari Seaborn dengan garis grid abu-abu tipis (`#f1f5f9`) dan warna tema profesional seperti Slate Gray (`#94a3b8`) dan Steel Blue (`#3182ce`) untuk menghasilkan grafik berestetika premium.

---

## 2. Eksplorasi & Memuat Dataset

Dataset dimuat dari file `train.csv` menggunakan Pandas:
```python
df = pd.read_csv("train.csv")
df.head()
```
Dataset Titanic ini memiliki kolom-kolom seperti `PassengerId`, `Survived` (target kelas), `Pclass` (kelas tiket), `Name`, `Sex`, `Age`, `SibSp` (jumlah saudara/pasangan), `Parch` (jumlah orang tua/anak), `Ticket`, `Fare` (tarif), `Cabin`, dan `Embarked` (pelabuhan keberangkatan).

---

## 3. Pra-pemrosesan Data (Data Preprocessing)

Langkah pra-pemrosesan sangat krusial karena model Decision Tree dari nol ini memerlukan input numerik bersih tanpa nilai kosong (*missing values*).

```python
# 1. Drop kolom yang tidak relevan
df = df.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1)

# 2. Mengisi nilai kosong pada kolom Age dan Fare dengan median
median_age = df["Age"].median()
df["Age"] = df["Age"].fillna(median_age)

median_fare = df["Fare"].median()
df["Fare"] = df["Fare"].fillna(median_fare)

# 3. Mapping nilai kategorikal ke numerik
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2}).fillna(0)
```

### Penjelasan Keputusan Preprocessing:
1. **Dropping Columns**: Kolom `PassengerId` (hanya indeks unik), `Name` (teks nama), `Ticket` (kode tiket), dan `Cabin` (mayoritas datanya kosong) dihapus karena tidak memberikan informasi pola yang bernilai bagi pohon keputusan.
2. **Median Imputation**: Nilai yang hilang pada `Age` dan `Fare` diisi menggunakan **median** (nilai tengah), yang lebih robust terhadap pencilan (*outliers*) dibandingkan rata-rata (*mean*).
3. **Categorical Encoding**: 
   * Jenis kelamin diubah menjadi biner: `male` $\rightarrow$ `0` dan `female` $\rightarrow$ `1`.
   * Pelabuhan asal keberangkatan (`Embarked`) diubah menjadi numerik: `S` $\rightarrow$ `0`, `C` $\rightarrow$ `1`, `Q` $\rightarrow$ `2`. Nilai kosong diisi dengan default `0`.

---

## 4. Pemisahan Data Manual (Train-Test Split)

Alih-alih menggunakan `train_test_split` milik *scikit-learn*, kode ini menerapkan pemisahan data 80% pelatihan dan 20% pengujian secara manual:

```python
# Mengambil semua kolom fitur kecuali Survived
feature_names = [col for col in df.columns if col != "Survived"]
X = df.drop("Survived", axis=1).values.tolist()
y = df["Survived"].tolist()

# Proses shuffling manual secara aman
combined = list(zip(X, y))
random.shuffle(combined)

# Pembagian indeks split 80%
split_index = int(len(combined) * 0.8)
train_data = combined[:split_index]
test_data = combined[split_index:]

# Ekstraksi data latih dan data uji
X_train = [x for x, y in train_data]
y_train = [y for x, y in train_data]
X_test = [x for x, y in test_data]
y_test = [y for x, y in test_data]
```

### Cara Kerja Split Manual:
1. Kolom fitur diubah menjadi list multidimensi (`X`) dan label diubah menjadi list biasa (`y`).
2. Digunakan fungsi `zip(X, y)` untuk menggabungkan fitur dengan labelnya agar saat diacak, hubungan antara fitur dan label tidak terputus.
3. Fungsi `random.shuffle()` digunakan untuk mengacak urutan baris data secara acak.
4. Dataset dipotong menggunakan slicing index pada 80% total data, menghasilkan **712 sampel data latih** dan **179 sampel data uji**.

---

## 5. Arsitektur Kelas Node (`Node`)

Daun (*leaf*) dan titik percabangan (*decision node*) diwakili oleh struktur data objek kelas `Node`.

```python
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature      # Indeks fitur yang digunakan untuk splitting
        self.threshold = threshold  # Batas ambang splitting data
        self.left = left            # Cabang kiri (<= threshold)
        self.right = right          # Cabang kanan (> threshold)
        self.value = value          # Kelas prediksi akhir (hanya di leaf node)
```

Jika sebuah node memiliki atribut `value` yang tidak `None`, maka node tersebut diidentifikasi sebagai **Leaf Node** (daun akhir yang mengembalikan keputusan kelas `Survived: 0` atau `Survived: 1`). Jika `value` bernilai `None`, node tersebut adalah **Decision Node** yang akan mengarahkan alur pencarian ke cabang kiri atau kanan berdasarkan perbandingan `row[feature] <= threshold`.

---

## 6. Algoritma Decision Tree Classifier (`DecisionTree`)

Ini adalah bagian inti dari proyek. Kelas `DecisionTree` diimplementasikan menggunakan konsep **Gini Impurity** untuk mencari pembagian (*split*) data terbaik pada setiap tingkat pohon.

### A. Kriteria Berhenti Rekursi (*Stopping Criteria*)
Proses pembuatan pohon dihentikan dan sebuah node daun dibuat ketika:
1. Pohon mencapai batas kedalaman maksimum (`depth >= max_depth`).
2. Seluruh sampel dalam node memiliki label kelas yang sama (`num_labels == 1`).
3. Jumlah sampel di node tersebut lebih kecil dari jumlah minimum sampel yang diperlukan untuk membagi (`num_samples < min_samples_split`).

### B. Rumus Matematika Gini Impurity
Gini Impurity mengukur seberapa sering elemen yang dipilih secara acak dari subset akan salah diklasifikasikan jika diklasifikasikan secara acak sesuai dengan distribusi label dalam subset.

Rumus Gini Impurity untuk suatu himpunan label $D$:
$$G(D) = 1 - \sum_{i=1}^{C} p_i^2$$
Di mana $p_i$ adalah probabilitas kemunculan kelas $i$ dalam himpunan tersebut, dan $C$ adalah jumlah kelas unik (dalam hal ini, 2 kelas: Selamat / Tidak Selamat).

Ketika dataset dibagi menjadi cabang kiri ($D_L$) dan cabang kanan ($D_R$), Gini Impurity hasil pemisahan dihitung berdasarkan rata-rata tertimbang:
$$\text{Gini Split} = \frac{|D_L|}{|D|} G(D_L) + \frac{|D_R|}{|D|} G(D_R)$$

Fungsi `gini_index` mengimplementasikan rumus tersebut:
```python
def gini_index(self, left_y, right_y):
    total = len(left_y) + len(right_y)
    gini = 0.0
    for group in [left_y, right_y]:
        size = len(group)
        if size == 0:
            continue
        counts = Counter(group).values()
        impurity = 1.0 - sum((c / size) ** 2 for c in counts)
        gini += impurity * (size / total)
    return gini
```

### C. Optimasi Batas Ambang (*Threshold Sampling*)
Untuk mempercepat komputasi pada fitur dengan banyak nilai unik kontinu (seperti `Age` dan `Fare`), algoritma menerapkan optimasi pencarian batas ambang (threshold):
```python
# Sampling threshold untuk efisiensi kecepatan eksekusi fitur numerik kontinu
if len(unique_values) > 15:
    step = len(unique_values) // 15
    thresholds = [unique_values[i] for i in range(0, len(unique_values), step)]
else:
    thresholds = unique_values
```
Jika jumlah nilai unik suatu fitur melebihi 15, algoritma hanya akan menguji maksimal 15 nilai ambang batas yang tersebar merata, alih-alih menguji ratusan nilai unik satu per satu. Ini sangat memangkas waktu latih model tanpa mengorbankan performa secara signifikan.

---

## 7. Pelatihan Model & Evaluasi

Model diinisiasi dengan kedalaman maksimum 7 (`max_depth=7`) dan dilatih menggunakan data pelatihan:

```python
dt_classifier = DecisionTree(max_depth=7)
dt_classifier.latih(X_train, y_train)
```

Setelah proses pelatihan selesai, akurasi diukur pada data latih dan data uji:
* **Train Accuracy**: `88.62%` — Akurasi model dalam memprediksi data yang digunakannya saat belajar.
* **Test Accuracy**: `85.47%` — Akurasi model dalam memprediksi data baru yang belum pernah dilihat sebelumnya. Nilai $\approx 85\%$ ini menunjukkan model memiliki kemampuan generalisasi yang sangat baik dan tidak mengalami *overfitting* yang parah.

---

## 8. Perhitungan Confusion Matrix secara Manual

Untuk mengevaluasi performa klasifikasi biner secara detail, Confusion Matrix dihitung secara manual tanpa bantuan *scikit-learn*:

```python
TP, TN, FP, FN = 0, 0, 0, 0
for i in range(len(X_test)):
    pred = dt_classifier.predict(X_test[i])
    actual = y_test[i]
    if pred == 1 and actual == 1: 
        TP += 1
    elif pred == 0 and actual == 0: 
        TN += 1
    elif pred == 1 and actual == 0: 
        FP += 1
    elif pred == 0 and actual == 1: 
        FN += 1
```

### Definisi Metrik:
* **True Positive (TP)**: Penumpang diprediksi selamat, dan kenyataannya memang selamat.
* **True Negative (TN)**: Penumpang diprediksi tidak selamat, dan kenyataannya memang tidak selamat.
* **False Positive (FP) - Type I Error**: Penumpang diprediksi selamat, tetapi kenyataannya tidak selamat.
* **False Negative (FN) - Type II Error**: Penumpang diprediksi tidak selamat, tetapi kenyataannya selamat.

Hasil matriks ini kemudian divisualisasikan menggunakan peta panas (heatmap) Seaborn untuk mempermudah pembacaan performa model.

---

## 9. Analisis Kepentingan Fitur (Feature Importance)

Salah satu kelebihan Decision Tree adalah kemudahan untuk diinterpretasikan (*explainable AI*). Kode mengimplementasikan fungsi `get_feature_importances` untuk menghitung kontribusi setiap fitur berdasarkan seberapa besar penurunan Gini Impurity yang dihasilkannya di seluruh pohon keputusan.

$$\text{Gini Decrease} = \frac{N_{\text{node}}}{N_{\text{total}}} \left( \text{Gini}_{\text{parent}} - \left( \frac{N_{\text{left}}}{N_{\text{node}}} \text{Gini}_{\text{left}} + \frac{N_{\text{right}}}{N_{\text{node}}} \text{Gini}_{\text{right}} \right) \right)$$

Di mana $N$ adalah jumlah sampel. Penurunan Gini ini diakumulasikan untuk setiap fitur dan kemudian dinormalisasi agar total kontribusi bernilai `1.0` (atau 100%).

### Visualisasi Feature Importance:
Berdasarkan visualisasi grafik batang horizontal pada bagian akhir notebook:
1. **Sex (Jenis Kelamin)** memiliki nilai kepentingan tertinggi secara mutlak. Hal ini logis dan sesuai dengan sejarah penyelamatan Titanic yang mendahulukan wanita dan anak-anak (*"women and children first"*).
2. **Fare (Tarif tiket)** dan **Age (Umur)** menempati posisi penting berikutnya. Umur membedakan kategori anak-anak, sedangkan Fare mencerminkan status sosial ekonomi penumpang.
3. **Pclass (Kelas Penumpang)** juga berkontribusi besar karena penumpang kelas 1 memiliki akses prioritas ke sekoci penyelamat dibandingkan penumpang kelas 3.

---

## 10. Kesimpulan

* Proyek ini berhasil membuktikan bahwa algoritma **Decision Tree Classifier** dapat ditulis secara bersih, efisien, dan akurat menggunakan pemrograman Python dasar (scratch).
* Hasil akurasi pengujian sebesar **85.47%** sangat bersaing dengan pustaka tingkat tinggi seperti Scikit-Learn.
* Fitur **Sex** dan **Fare/Pclass** terbukti secara matematis menjadi faktor penentu paling dominan dalam memprediksi keselamatan penumpang pada tragedi Titanic 1912.
