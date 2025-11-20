import pandas as pd
import numpy as np

# 1. Veriyi Oku
file_path = 'IMBD.csv'  # Dosya adını kendine göre kontrol et
df = pd.read_csv(file_path)
df_clean = df.copy()

# 2. Certificate sütununu baştan atıyoruz (İstenmediği için)
if 'certificate' in df_clean.columns:
    df_clean = df_clean.drop(columns=['certificate'])
    print("'certificate' sütunu veriden çıkarıldı.")

# 3. Runtime (Süre) Temizliği -> Int64
if 'runtime' in df_clean.columns:
    # Sadece sayıları al, float yap
    df_clean['runtime'] = df_clean['runtime'].astype(str).str.extract(r'(\d+)').astype(float)
    # Sonsuz değerleri temizle ve Int64 (Tamsayı) yap
    df_clean['runtime'] = df_clean['runtime'].replace([np.inf, -np.inf], np.nan).round().astype('Int64')

# 4. Votes (Oy) Temizliği -> Int64
if 'votes' in df_clean.columns:
    # Virgülleri kaldır
    df_clean['votes'] = df_clean['votes'].astype(str).str.replace(',', '', regex=False)
    # Sayıya çevir, hata vereni NaN yap
    df_clean['votes'] = pd.to_numeric(df_clean['votes'], errors='coerce')
    # Sonsuz değerleri temizle ve Int64 (Tamsayı) yap
    df_clean['votes'] = df_clean['votes'].replace([np.inf, -np.inf], np.nan).round().astype('Int64')

# 5. Metin (String) Temizliği ve Stars Düzeltmesi
object_cols = ['movie', 'genre', 'director', 'stars', 'description']
junk_strings = ['', '[]', 'nan', 'NaN', 'Null', 'null', 'NULL'] 

for col in object_cols:
    if col in df_clean.columns:
        # Genel temizlik
        df_clean[col] = df_clean[col].astype(str).str.strip()
        df_clean[col] = df_clean[col].str.replace(r'[\[\]"\']', '', regex=True)
        # Gereksiz stringleri (NaN gibi) gerçek boşluğa (np.nan) çevir
        df_clean[col] = df_clean[col].replace(junk_strings, np.nan)

        # ÖZEL DÜZELTME: Stars sütunundaki virgül hataları
        if col == 'stars':
            df_clean['stars'] = df_clean['stars'].str.replace(r',\s+,', ',', regex=True) # ", , " temizle
            df_clean['stars'] = df_clean['stars'].str.replace(r',+', ',', regex=True)    # ",,," temizle
            df_clean['stars'] = df_clean['stars'].str.strip(',')                         # Baştaki/sondaki virgülü at
            df_clean['stars'] = df_clean['stars'].str.replace(', ', ',').str.replace(',', ', ') # Standartlaştır

        # ÖZEL DÜZELTME: Genre satır sonu boşlukları
        if col == 'genre':
            df_clean['genre'] = df_clean['genre'].str.replace('\n', '').str.strip()

# 6. Eksik Veri ve Tekrar Eden Satır Temizliği
df_clean = df_clean.drop_duplicates()
# Artık certificate olmadığı için tüm sütunlara bakarak boş satırları siliyoruz
df_clean = df_clean.dropna()

# 7. Sonuçları Göster
print("\n--- Temiz Veri Özeti ---")
df_clean.info()
print("\n--- Rastgele 5 Örnek ---")
print(df_clean.sample(5))

# 8. Kaydet
output_filename = 'IMDB_cleaned.csv'
df_clean.to_csv(output_filename, index=False)
print(f"\nİşlem tamam! Dosya '{output_filename}' olarak kaydedildi.")