import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

MIN_VOTES = 150
MIN_RUNTIME = 60
SMOOTH_FACTOR = 5

# Download the dataset
df = pd.read_csv('IMDB_cleaned.csv')

# remove the noise
df = df[df['votes'] > MIN_VOTES]
df = df[df['runtime'] >= MIN_RUNTIME]
print(f"Number of films after those with few votes deleted: {len(df)}")

# Log transformation of votes
df['log_votes'] = np.log1p(df['votes'])

# Genre encoding
df['genre'] = df['genre'].str.replace(' ', '')
genres_encoded = df['genre'].str.get_dummies(sep=',')

# X and y seperation
X = pd.concat([df[['log_votes', 'runtime', 'director', 'stars']], genres_encoded], axis=1)
y = df['rating']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# cast score calculation
# ----------------------------------------------------------------------------------------------------

def calculate_actor_scores(train_df, m):

    exploded = train_df.assign(actor=train_df['stars'].str.split(',')).explode('actor')
    exploded['actor'] = exploded['actor'].str.strip()
    
    global_mean = train_df['rating'].mean()
    agg = exploded.groupby('actor')['rating'].agg(['count', 'mean'])
    
    counts = agg['count']
    means = agg['mean']
    
    actor_score_map = (counts * means + m * global_mean) / (counts + m)
    
    return actor_score_map, global_mean

def get_average_cast_score(df_target, score_map, global_val):

    exploded = df_target.assign(actor=df_target['stars'].str.split(',')).explode('actor')
    exploded['actor'] = exploded['actor'].str.strip()
    
    exploded['actor_score'] = exploded['actor'].map(score_map).fillna(global_val)
    
    avg_scores = exploded.groupby(exploded.index)['actor_score'].mean()
    
    return avg_scores

temp_train = X_train.copy()
temp_train['rating'] = y_train

actor_map, global_avg = calculate_actor_scores(temp_train, SMOOTH_FACTOR)

X_train['cast_score'] = get_average_cast_score(X_train, actor_map, global_avg)

X_test['cast_score'] = get_average_cast_score(X_test, actor_map, global_avg)

X_train = X_train.drop(columns=['stars'], errors='ignore')
X_test = X_test.drop(columns=['stars'], errors='ignore')

# ----------------------------------------------------------------------------------------------------

# smooth encoding
def smoothed_target_encoding(train_df, col_name, target_name, m):
    global_mean = train_df[target_name].mean()
    agg = train_df.groupby(col_name)[target_name].agg(['count', 'mean'])
    counts = agg['count']
    means = agg['mean']
    smooth = (counts * means + m * global_mean) / (counts + m)
    return smooth, global_mean

# Director column encoding
temp_train_data = X_train.copy()
temp_train_data['rating'] = y_train

# mapping directors to their mean ratings, filling missing with global mean
dir_map, global_mean = smoothed_target_encoding(temp_train_data, 'director', 'rating', SMOOTH_FACTOR)
X_train['director_encoded'] = X_train['director'].map(dir_map).fillna(global_mean)
X_test['director_encoded'] = X_test['director'].map(dir_map).fillna(global_mean)

# drop original director column
X_train = X_train.drop(columns=['director'])
X_test = X_test.drop(columns=['director'])

print(X_train.head())

# ----------------------------------------------------------------------------------------------------

# Ridge Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# optimal alpha values
alphas = [0.1, 1, 10, 100, 1000, 5000, 10000, 20000, 30000, 50000]
# accuracy threshold
threshold = 1
print("Accuracy threshold:", threshold)

ridge = RidgeCV(alphas)
ridge.fit(X_train_scaled, y_train)
y_pred = ridge.predict(X_test_scaled)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt( mean_squared_error(y_test, y_pred) )

    
# Accuracy calculation
errors = abs(y_test - y_pred)
correct_predictions = (errors <= threshold).sum()
total_predictions = len(y_test)
accuracy_score = correct_predictions / total_predictions
alpha = ridge.alpha_

# Results
print(f"Alpha: {alpha} | R2 Score: {r2:.4f} | RMSE: {rmse:.4f} | Accuracy: % {accuracy_score * 100:.2f}")