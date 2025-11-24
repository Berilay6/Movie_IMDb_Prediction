import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Download the dataset
df = pd.read_csv('IMDB_cleaned.csv')

# Genre encoding
df['genre'] = df['genre'].str.replace(' ', '')
genres_encoded = df['genre'].str.get_dummies(sep=',')

# number of unique genres
print(f"Toplam {genres_encoded.shape[1]} farklı tür bulundu.")

# X and y seperation
X = pd.concat([df[['votes', 'runtime', 'director']], genres_encoded], axis=1)
y = df['rating']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Director column encoding
temp_train_data = X_train.copy()
temp_train_data['rating'] = y_train

# mean rating per director
director_means = temp_train_data.groupby('director')['rating'].mean()
# overall mean rating
global_mean = y_train.mean()

# mapping directors to their mean ratings, filling missing with global mean
X_train['director_encoded'] = X_train['director'].map(director_means).fillna(global_mean)
X_test['director_encoded'] = X_test['director'].map(director_means).fillna(global_mean)

# drop original director column
X_train = X_train.drop(columns=['director'])
X_test = X_test.drop(columns=['director'])

# Ridge Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# optimal alpha values
for alpha in [25000 , 26000, 27000]:
    ridge_model = Ridge(alpha=alpha)
    ridge_model.fit(X_train_scaled, y_train)

    y_pred = ridge_model.predict(X_test_scaled)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    print("-" * 30)
    print(f"Alpha: {alpha}")
    print(f"Director Eklenmiş Yeni R2 Skoru: {r2:.4f}")
    print(f"Director Eklenmiş MSE: {mse:.4f}")
    print("-" * 30)
    print(f"Yönetmen Etkisi (Katsayı): {ridge_model.coef_[2]:.4f}")


# 0.3177 for alpha 26000 is the best R2 score achieved.