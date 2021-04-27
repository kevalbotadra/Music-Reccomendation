from functions import recommend_music, get_song_info

song_name = "cable car"
artists_name = "abhi the nomad"
num_of_reccomendations = 10


# song = get_song_info("montero", "lil nas x")
# print(song)

recommended_songs = recommend_music(song_name, artists_name, int(num_of_reccomendations))