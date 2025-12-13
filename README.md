# Linear Regression from Scratch

Implementasi algoritma **Linear Regression** dari awal menggunakan Python tanpa menggunakan library machine learning seperti scikit-learn.

## 📖 Daftar Isi

- [Tentang Proyek](#tentang-proyek)
- [Teori Dasar](#teori-dasar)
  - [Apa itu Linear Regression?](#apa-itu-linear-regression)
  - [Gradient Descent](#gradient-descent)
- [Struktur Proyek](#struktur-proyek)
- [Persyaratan](#persyaratan)
- [Instalasi](#instalasi)
- [Cara Menjalankan](#cara-menjalankan)
- [Penjelasan Kode](#penjelasan-kode)
  - [Fungsi Gradient Descent](#fungsi-gradient-descent)
  - [Fungsi Drop Outlier](#fungsi-drop-outlier)
  - [Proses Training](#proses-training)
  - [Visualisasi](#visualisasi)
- [Dataset](#dataset)
- [Output](#output)

---

## Tentang Proyek

Proyek ini mengimplementasikan algoritma **Linear Regression** dari awal (from scratch) menggunakan metode **Gradient Descent** untuk menemukan parameter optimal (slope dan intercept) dari garis regresi. Proyek ini dibuat untuk tujuan pembelajaran dan pemahaman mendalam tentang bagaimana algoritma linear regression bekerja di balik layar.

---

## Teori Dasar

### Apa itu Linear Regression?

**Linear Regression** adalah algoritma supervised learning yang digunakan untuk memprediksi nilai kontinu berdasarkan satu atau lebih variabel independen. Model linear regression mencari hubungan linier antara variabel input (x) dan variabel output (y).

Persamaan garis lurus yang digunakan:

```
y = mx + b
```

Dimana:
- `y` = nilai prediksi (dependent variable)
- `x` = nilai input (independent variable)  
- `m` = slope (kemiringan garis)
- `b` = intercept (titik potong dengan sumbu y)

Tujuan dari linear regression adalah menemukan nilai `m` dan `b` yang optimal sehingga garis yang dihasilkan memiliki error terkecil terhadap data aktual.

### Gradient Descent

**Gradient Descent** adalah algoritma optimasi yang digunakan untuk menemukan nilai minimum dari suatu fungsi. Dalam konteks linear regression, gradient descent digunakan untuk meminimalkan **Mean Squared Error (MSE)**.

#### Rumus MSE:

```
MSE = (1/n) * Σ(yi - ŷi)²
```

Dimana:
- `n` = jumlah data
- `yi` = nilai aktual
- `ŷi` = nilai prediksi (mx + b)

#### Rumus Gradient:

Turunan parsial MSE terhadap m:
```
∂MSE/∂m = -(2/n) * Σ xi * (yi - (m*xi + b))
```

Turunan parsial MSE terhadap b:
```
∂MSE/∂b = -(2/n) * Σ (yi - (m*xi + b))
```

#### Update Parameter:

```
m_new = m_old - learning_rate * ∂MSE/∂m
b_new = b_old - learning_rate * ∂MSE/∂b
```

---

## Struktur Proyek

```
linear-regression-from-scratch/
│
├── linear_regression.py    # File utama implementasi linear regression
├── requirements.txt        # File dependencies Python
├── LICENSE                  # MIT License
├── res/
│   ├── train.csv           # Dataset training (700 data)
│   └── test.csv            # Dataset testing (300 data)
├── .gitignore              # File gitignore untuk Python
└── README.md               # Dokumentasi proyek
```

---

## Persyaratan

- Python 3.8+
- NumPy >= 1.20.0
- Pandas >= 1.3.0
- Matplotlib >= 3.4.0

---

## Instalasi

1. **Clone repository**

   ```bash
   git clone https://github.com/nelsooooon/linear-regression-from-scratch.git
   cd linear-regression-from-scratch
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   Atau install secara manual:

   ```bash
   pip install 'numpy>=1.20.0' 'pandas>=1.3.0' 'matplotlib>=3.4.0'
   ```

---

## Cara Menjalankan

Jalankan script Python dengan perintah:

```bash
python linear_regression.py
```

Program akan:
1. Membaca data training dan testing dari folder `res/`
2. Membersihkan data (menghapus nilai null dan outlier)
3. Menjalankan gradient descent selama 1000 epoch
4. Menampilkan nilai slope (m) dan intercept (b) yang ditemukan
5. Menampilkan visualisasi scatter plot dengan garis regresi

---

## Penjelasan Kode

### Fungsi Gradient Descent

```python
def gradient_descent(m_now, b_now, points, learning_rate):
    m_gradient = 0
    b_gradient = 0
    
    n = len(points)
    
    for i in range(n):
        xi = points.iloc[i].x
        yi = points.iloc[i].y
        
        m_gradient += -(2/n) * xi * (yi - (m_now * xi + b_now))
        b_gradient += -(2/n) * (yi - (m_now * xi + b_now))
        
    m = m_now - m_gradient * learning_rate
    b = b_now - b_gradient * learning_rate
    
    return m, b
```

**Penjelasan:**
- Fungsi ini menghitung gradient dari MSE untuk parameter `m` dan `b`
- Mengiterasi setiap data point untuk menghitung total gradient
- Mengupdate nilai `m` dan `b` berdasarkan learning rate
- Mengembalikan nilai `m` dan `b` yang baru

### Fungsi Drop Outlier

```python
def drop_outlier(df):
    df_clean = df.copy()
    
    for col in ['x', 'y']:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        
        IQR = Q3 - Q1
        lower = Q1 - (1.5 * IQR)
        upper = Q3 + (1.5 * IQR)
        df_clean = df_clean[(df_clean[col] >= lower) & (df_clean[col] <= upper)]
        
    return df_clean
```

**Penjelasan:**
- Menggunakan metode **Interquartile Range (IQR)** untuk mendeteksi outlier
- Menghitung Q1 (kuartil pertama) dan Q3 (kuartil ketiga)
- Data dianggap outlier jika berada di luar range `[Q1 - 1.5*IQR, Q3 + 1.5*IQR]`
- Menghapus outlier untuk kolom `x` dan `y`

### Proses Training

```python
m = 0
b = 0
learning_rate = 0.0001
epochs = 1000

for i in range(epochs):
    if i % 50 == 0:
        print(f"Epoch: {i}")
        
    m, b = gradient_descent(m, b, data_train_clean, learning_rate)
```

**Parameter yang digunakan:**
- `m` dan `b` diinisialisasi dengan nilai 0
- `learning_rate = 0.0001` (langkah pembelajaran yang kecil untuk konvergensi stabil)
- `epochs = 1000` (iterasi gradient descent)

### Visualisasi

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].scatter(x=data_train_clean.x, y=data_train_clean.y, color='black', alpha=0.5)
axes[1].scatter(x=data_test.x, y=data_test.y, color='black', alpha=0.5)

for i in range(len(axes)):
    axes[i].plot(list(range(1, 101)), [m * x + b for x in range(1, 101)], color='red')

fig.tight_layout()
plt.show()
```

**Penjelasan:**
- Membuat 2 subplot side-by-side
- Plot kiri: Data training dengan garis regresi
- Plot kanan: Data testing dengan garis regresi
- Scatter plot berwarna hitam, garis regresi berwarna merah

---

## Dataset

Dataset yang digunakan terdiri dari 2 file CSV:

| File | Jumlah Data | Kolom |
|------|-------------|-------|
| `train.csv` | 700 baris | x, y |
| `test.csv` | 300 baris | x, y |

Dataset memiliki hubungan linear positif antara variabel `x` dan `y`, ideal untuk pembelajaran linear regression.

---

## Output

Setelah menjalankan program, output yang dihasilkan:

1. **Progress Training**: Menampilkan nomor epoch setiap 50 iterasi
   ```
   Epoch: 0
   Epoch: 50
   Epoch: 100
   ...
   Epoch: 950
   ```

2. **Parameter Final**: Nilai slope (m) dan intercept (b) yang ditemukan
   ```
   0.9876... 1.234...
   ```

3. **Visualisasi**: Menampilkan 2 plot:
   - Plot kiri: Data training dengan garis regresi
   - Plot kanan: Data testing dengan garis regresi

---

## 📝 Lisensi

Proyek ini dilisensikan di bawah [MIT License](LICENSE) - lihat file LICENSE untuk detail lebih lanjut.
