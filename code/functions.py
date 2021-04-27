import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import requests, json, logging
import base64
import six
import pandas as pd


file = ".data/data.csv"

def get_spot_data(all_cols=True):
    if all_cols == True:
        df = pd.read_csv(".data/data.csv")
        return df
    else:
        df = pd.read_csv(".data/data.csv")
        df = df.drop(["artists", "name", "id", "release_date", "year"], axis=1)
        return df


def read_a_csv(use_dataframe=False, dataframe=None, data_file=".data/data.csv", all_cols=False, dropped_cols=True, cols_to_drop=["artists", "name", "id", "release_date", "year"]):
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


def kMeans_clustering(numpy=False, numpy_array=None, data_file=".data/data.csv", num_clusters=10, iteration=True, num_iters=10, inertia_plot=False):
    if numpy == True:
        data = numpy_array
        k_means = KMeans(n_clusters=num_clusters)
        k_clusters = k_means.fit_predict(data)
        df = read_a_csv()
        df["cluster"] = k_clusters
    elif iteration == True:
        df = read_a_csv()
        clusters = range(1, num_iters+1)
        inertia = {}
        data = df.to_numpy()
        print("Now Formulating Clusters:")
        for i in tqdm(clusters):
            k_means = KMeans(n_clusters=i)
            k_clusters = k_means.fit_predict(data)
            inertia.append(k_means.inertia_)
        df = read_a_csv()
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

def data_visaulization(data_file=".data/data.csv", no_cluters=False, yes_clusters=True, num_clusters=10, full_featured_graph=True, pca_graph=True, heatmap=True):
    if no_cluters == True:
        df = read_a_csv()
        if full_featured_graph == True:
            sns.pairplot(df)

    elif yes_clusters == True:

        _, df = kMeans_clustering()
            
        if full_featured_graph == True:
            sns.pairplot(df)
        
        if heatmap == True:
            fig, ax = plt.subplots(figsize=(13,13))    
            sns.heatmap(df.corr(), annot=True, square=True, ax=ax)
            ax.plot()
            plt.show()
            
        if pca_graph == True:
            df = read_a_csv()
            pca = PCA(2).fit(df)
            pca_data = pd.DataFrame(pca.transform(df)).values
        
            _, df2 = kMeans_clustering(numpy=True, numpy_array=df, data_file=file, num_clusters=10, iteration=False, num_iters=0, inertia_plot=False)
            df2 = read_a_csv(use_dataframe=True, dataframe=df2, data_file=None, dropped_cols=True, cols_to_drop=["cluster"])
            y_kmeans = KMeans(n_clusters=num_clusters).fit_predict(df2.values)
            u_labels = np.unique(y_kmeans)

            for i in u_labels:
                plt.scatter(pca_data[y_kmeans == i , 0] , pca_data[y_kmeans == i , 1], label = i)
            plt.legend()
            plt.show()

def scale_data(data_file):
    df = pd.read_csv(data_file)
    df = df.drop(["artists", "name", "id", "release_date", "year"], axis=1)

    scaler = MinMaxScaler()
    scaled_df = scaler.fit_transform(df)

    df2 = read_a_csv(all_cols=True, dropped_cols=False, cols_to_drop=None)

    scaled_df = pd.DataFrame(scaled_df, columns=df.columns)

    scaled_df.insert(1, "artists", df2["artists"], True )
    scaled_df.insert(6, "id", df2["id"], True)
    scaled_df.insert(12, "name", df2["name"], True)
    scaled_df.insert(14, "release_date", df2["release_date"], True)
    scaled_df.insert(18, "year", df2["year"], True)

    return scaled_df

def is_song_in_data(song_name):
    data = get_spot_data(all_cols=False)
    try:
        song = data[data.name.str.lower() == song_name.lower()].head(1)
        if song.empty:
            return False
        else:
            print("Song is already in the dataset!")
        return True
    except:
        print("Adding Song to Data......")
        return False

def get_song_info(song_name, artist_name, req_type = "track"):
    df = pd.read_csv(".data/data.csv")
    in_data = is_song_in_data(song_name=song_name)
    if in_data == True:
        return song_name
    else:
        client_id = "642acf340c4d4ca5bb23401b63982a5f"
        client_secret = "c7c4f95c2d244c1a88ee379d8d1f0223"
        auth_header = {'Authorization' : 'Basic %s' % base64.b64encode(six.text_type(client_id + ':' +                              client_secret).encode('ascii')).decode('ascii')}
        
        r = requests.post('https://accounts.spotify.com/api/token', headers = auth_header, data= {'grant_type': 'client_credentials'})
        token = 'Bearer {}'.format(r.json()['access_token'])
        headers = {'Authorization': token, "Accept": 'application/json', 'Content-Type': "application/json"}

        payload = {"q" : "artist:{} track:{}".format(artist_name, song_name), "type": req_type, "limit": "1"}

        res = requests.get('https://api.spotify.com/v1/search', params = payload, headers = headers)
        res = res.json()['tracks']['items'][0]
        whole_date = res['album']['release_date']
        year = res['album']['release_date'][:4]
        month = res["album"]["release_date"]
        day = res['album']['release_date'][8:10]
        explicit = res['explicit']
        if explicit == True:
            explicit = 0
        else:
            explicit = 1
        popularity = res["popularity"]

        artists = []
        for i in range(len(res["artists"])):
            artists.append(res["artists"][i]["name"])
            
        song_name = res['name'].lower()
        track_id = res['id']

        res = requests.get('https://api.spotify.com/v1/audio-analysis/{}'.format(track_id), headers = headers)
        res = res.json()['track']
        duration = res['duration']
        key = res['key']
        mode = res['mode']
        tempo = res['tempo']


        res = requests.get('https://api.spotify.com/v1/audio-features/{}'.format(track_id), headers = headers)
        res = res.json()
        acousticness =  res['acousticness']
        danceability = res['danceability']
        energy = res['energy']
        instrumentalness = res['instrumentalness']
        liveness = res['liveness']
        loudness = res['loudness']
        speechiness = res['speechiness']
        valence = res['valence']


        df.loc[len(df)]=[acousticness, artists, danceability, duration, energy, explicit, track_id, instrumentalness, key, liveness, loudness, mode, song_name, popularity, whole_date, speechiness, tempo, valence, year]
        df.to_csv(".data/data.csv", index=False)
    
        

    return song_name



from fuzzywuzzy.process import extractOne
def recommend_songs(song_name, artist_name, num_of_recommendations):
    data = scale_data("data.csv")
    df = read_a_csv(all_cols=True, dropped_cols=False, cols_to_drop=None)
    songs = data[data["name"].str.contains(song_name)]
    if songs.empty:
        songs_names = data["name"].tolist()
        song_tuple = extractOne(song_name, songs_names)
        songs = data[data["name"] == song_tuple[0]]
    artist_name_lower = artist_name.lower()
    artist_name_titled = artist_name.title()
    artist_name_upper = artist_name.upper()
    artist_name_list = [artist_name_lower, artist_name_titled, artist_name_upper]
    for i in artist_name_list:
        try:
            song = songs[songs['artists'].str.contains(i)].head()
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
        except:
            pass


def recommend_music(song_name, artist_name, num_of_reccomendations):
    song = get_song_info(song_name, artist_name)
    recommended_songs = recommend_songs(song, artist_name, num_of_reccomendations)
    print(recommended_songs)

    return recommended_songs

    
def get_for_popularity(song_name, artist_name):
    scaled_df = scale_data("data.csv")
    song = get_song_info(song_name, artist_name)
    song_name_lower = artist_name.lower()
    song_name_titled = artist_name.title()
    song_name_upper = artist_name.upper()
    song_name_list = [song_name_lower, song_name_titled, song_name_upper]
    for i in song_name_list:
        try:
            song = scaled_df[scaled_df["name"].str.contains(i)].head()
            song = song.drop(["artists", "name", "popularity", "id", "release_date", "year"], axis=1)
            return song
        except ValueError:
            print("Value Error: Can't Get Data for this Song")
    


# from sklearn.model_selection import train_test_split
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




    












    

    




    


        
