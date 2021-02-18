import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from tqdm import tqdm

file = "data.csv"


def read_a_csv(use_dataframe, dataframe, data_file, all_cols, dropped_cols, cols_to_drop):
    if all_cols == True:
        if use_dataframe == True:
            df = dataframe
            df2 = df.drop(cols_to_drop, axis=1)
            return df2
        else:
            df = pd.read_csv(data_file)
            return df

    if dropped_cols == True:
        if use_dataframe == True:
            df = dataframe
            df2 = df.drop(cols_to_drop, axis=1)
        else:
            df = pd.read_csv(data_file)
            df2 = df.drop(cols_to_drop, axis=1)
        return df2

    return df, df2


def kMeans_clustering(numpy, numpy_array, data_file, num_clusters, iteration, num_iters, inertia_plot):
    if numpy == True:
        data = numpy_array
        k_means = KMeans(n_clusters=num_clusters)
        k_clusters = k_means.fit_predict(data)
        df = read_a_csv(False, None, data_file, False, True, ["artists", "name", "id", "release_date", "year"])
        df["cluster"] = k_clusters
    elif iteration == True:
        df = read_a_csv(False, None, data_file, False, True, ["artists", "name", "id", "release_date", "year"])
        clusters = range(1, num_iters+1)
        inertia = []
        data = df.to_numpy()
        print("Now Formulating Clusters:")
        for i in tqdm(clusters):
            k_means = KMeans(n_clusters=i)
            k_clusters = k_means.fit_predict(data)
            inertia.append(k_means.inertia_)
        df = read_a_csv(False, None, data_file, False, True, ["artists", "name", "id", "release_date", "year"])
        df["cluster"] = k_clusters

        if inertia_plot == True:
            plt.plot(clusters, inertia, 'bx-')
            plt.xlabel('k')
            plt.ylabel('Inertia')
            plt.show()
    else:
        df = read_a_csv(False, None, data_file, False, True, ["artists", "name", "id", "release_date", "year"])
        data = df.to_numpy()
        k_means = KMeans(n_clusters=num_clusters)
        k_clusters = k_means.fit_predict(data)
        df = read_a_csv(False, None, data_file, False, True, ["artists", "name", "id", "release_date", "year"])
        df["cluster"] = k_means

    
    return k_means.labels_, df

def data_visaulization(data_file, no_cluters, yes_clusters, num_clusters, full_featured_graph, pca_graph, heatmap):
    if no_cluters == True:
        df = read_a_csv(False, None, data_file, False, True, ["artists", "name", "id", "release_date", "year"])
        if full_featured_graph == True:
            sns.pairplot(df)

    elif yes_clusters == True:

        _, df = kMeans_clustering(False, None, data_file, num_clusters, True, 10, False)
            
        if full_featured_graph == True:
            sns.pairplot(df)
        
        if heatmap == True:
            fig, ax = plt.subplots(figsize=(13,13))    
            sns.heatmap(df.corr(), annot=True, square=True, ax=ax)
            ax.plot()
            plt.show()
            
        if pca_graph == True:
            df = read_a_csv(False, None, data_file, False, True, ["artists", "name", "id", "release_date", "year"])
            pca = PCA(2).fit(df)
            pca_data = pd.DataFrame(pca.transform(df)).values
        
            _, df2 = kMeans_clustering(True, df, file, 10, False, 0, False)
            df2 = read_a_csv(True, df2, None, False, True, ["cluster"])
            y_kmeans = KMeans(n_clusters=num_clusters).fit_predict(df2.values)
            u_labels = np.unique(y_kmeans)

            for i in u_labels:
                plt.scatter(pca_data[y_kmeans == i , 0] , pca_data[y_kmeans == i , 1], label = i)
            plt.legend()
            plt.show()

def scale_data(data_file, num_clusters):
    _, df = kMeans_clustering(False, None, data_file, num_clusters, False, 0, False)
    df = df.drop(["cluster"], axis=1)

    scaler = MinMaxScaler()
    scaled_df = scaler.fit_transform(df)

    df2 = read_a_csv(False, None, data_file, True, False, False)

    scaled_df = pd.DataFrame(scaled_df, columns=df.columns)

    scaled_df.insert(1, "artists", df2["artists"], True )
    scaled_df.insert(6, "id", df2["id"], True)
    scaled_df.insert(12, "name", df2["name"], True)
    scaled_df.insert(14, "release_date", df2["release_date"], True)
    scaled_df.insert(18, "year", df2["year"], True)

    return scaled_df


def recommend_songs(data1, song, num_of_recommendations):
    data = scale_data(data1, 10)
    df = read_a_csv(False, None, data1, True, False, None)
    song = data[data.name.str.lower() == song.lower()].head(1)
    song = song.drop(["artists", "name", "id", "release_date", "year"], axis=1)
    song = song.values[0]
    dists = []
    data = data.drop_duplicates(subset="name", keep="first", inplace=False)
    data = data.drop(["artists", "name", "id", "release_date", "year"], axis=1)
    for song_to_rec in data.values:
        dist = 0
        for col in range(len(data.columns)):
            dist = dist + np.absolute((float(song[col]) - (song_to_rec[col])))
        dists.append(dist)
    data.insert(1, "artists", df["artists"], True)
    data.insert(6, "id", df["id"], True)
    data.insert(12, "name", df["name"], True)
    data.insert(14, "release_date", df["release_date"], True)
    data.insert(18, "year", df["year"], True)
    data["dists"] = dists
    data = data.sort_values("dists")
    return data[["artists", "name", "year"]][1:num_of_recommendations]

from sklearn.model_selection import train_test_split
def predict_feature_dataset(data, feature, t_size):
  scaled_df = scale_data(data, 10)
  data = ['acousticness', 'danceability', 'duration_ms', 'energy', 'explicit', 
        'instrumentalness', 'key', 'liveness', 'loudness', 'mode',  'popularity', 'speechiness',
        'tempo', 'valence']

  for sub_feature in list(data):
    if feature == sub_feature:
      data.remove(sub_feature)
  feature = [feature]

  X = scaled_df[data]
  y = scaled_df[feature]

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=t_size)

  return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = predict_feature_dataset("data.csv", "energy", 0.2)
print(X_test)



    












    

    




    


        
