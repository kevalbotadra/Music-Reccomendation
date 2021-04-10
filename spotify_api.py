import requests, json, logging
import base64
import six
import pandas as pd
from functions import read_a_csv, scale_data, recommend_songs
from sklearn.preprocessing import MinMaxScaler

the_song_name = "Pleaser"
the_artists_name = "wallows"


df = pd.read_csv("data.csv")
df2 = df.drop(["artists", "name", "id", "release_date", "year"], axis=1)

def get_song_info(song_name, artist_name, req_type = "track"):
    client_id = "642acf340c4d4ca5bb23401b63982a5f"
    client_secret = "c7c4f95c2d244c1a88ee379d8d1f0223"
    auth_header = {'Authorization' : 'Basic %s' % base64.b64encode(six.text_type(client_id + ':' + client_secret).encode('ascii')).decode('ascii')}
    
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
    df.to_csv("data.csv", index=False)

    return song_name


song_name = get_song_info(the_song_name, the_artists_name)

print(recommend_songs("data.csv", song_name, 10))