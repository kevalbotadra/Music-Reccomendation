import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, mean_squared_error
from functions import get_for_popularity, scale_data


filename = 'popularity_predictor.sav'



df = pd.read_csv("data.csv")
df2 = df.drop(["artists", "name", "id", "release_date", "year"], axis=1)
scaler = MinMaxScaler()
scaled_df = scaler.fit_transform(df2)

scaled_df = pd.DataFrame(scaled_df, columns=df2.columns)

scaled_df.insert(1, "artists", df["artists"], True )
scaled_df.insert(6, "id", df["id"], True)
scaled_df.insert(12, "name", df["name"], True)
scaled_df.insert(14, "release_date", df["release_date"], True)
scaled_df.insert(18, "year", df["year"], True)
scaled_df.head()



X = scaled_df[['acousticness', 'danceability', 'duration_ms', 'energy', 'explicit', 
        'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'speechiness',
        'tempo', 'valence']]
y = scaled_df["popularity"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)



# """ comment when model has already been trained """
# emotion_dmatrix = xgb.DMatrix(X_train, y_train)

# hyper_param_opts = {
#     "learning_rate" : [0.01,0.1,0.5,0.9],
#     "n_estimators" : [100],
#     "sub_sample" : [0.3,0.5,0.9]
# }

# boost = xgb.XGBRegressor()

# boost_model = GridSearchCV(estimator=boost, param_grid=hyper_param_opts, scoring='neg_mean_squared_error', cv=4, verbose=1)

# boost_model.fit(X,y)

# print(f"Best Parameters: {boost_model.best_params_}")
# print(f"Best Score: {np.sqrt(np.abs(boost_model.best_score_))}")

# pickle.dump(boost_model, open(filename, 'wb'))




""" uncomment when model has already been trained """
with open(filename, "rb") as file:
  boost_model = pickle.load(file)

print(f"Best Parameters: {boost_model.best_params_}")
print(f"Best Score: {np.sqrt(np.abs(boost_model.best_score_))}")

song = get_for_popularity("cable car", "abhi the nomad")
song_pred = (boost_model.predict(song)[0]*100)
print(song_pred)





