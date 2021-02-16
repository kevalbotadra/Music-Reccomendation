import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
filename = 'popularity_predictor.sav'



df = pd.read_csv("data.csv")
df2 = df.drop(["artists", "name", "id", "release_date", "year"], axis=1)

clusters = range(1, 11)
inertia = []
data = df2.to_numpy()
for i in clusters:
  k_means = KMeans(n_clusters=i)
  k_clusters = k_means.fit_predict(data)
  inertia.append(k_means.inertia_)


plt.plot(clusters, inertia, 'bx-')
plt.xlabel('k')
plt.ylabel('Inertia')
plt.show()


scaler = MinMaxScaler()
df2 = df.drop(["artists", "name", "id", "release_date", "year"], axis=1)
df2["cluster"] = k_clusters
scaled_df = scaler.fit_transform(df2)

scaled_df = pd.DataFrame(scaled_df, columns=df2.columns)
print(scaled_df.head())

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



with open(filename, "rb") as file:
  model = pickle.load(file)


model.score(X_test, y_test)



# from sklearn.model_selection import cross_val_score
# train_score = cross_val_score(model, X_train, y_train, scoring="neg_mean_squared_error")
# print(f"Train Score: {train_score}")
# test_score = cross_val_score(model, X_test, y_test, scoring="neg_mean_squared_error")
# print(f"Test Score: {test_score}")