import pylast

# You have to have your own unique two values for API_KEY and API_SECRET
# Obtain yours from http://www.last.fm/api/account for Last.fm
API_KEY = "1486a76d68af81e81d5c9d1e861e5cc2" # this is a sample key
API_SECRET = "180dacfc2cbd029bceda21feac60d940"

# In order to perform a write operation you need to authenticate yourself
username = "kroening"
password_hash = pylast.md5("H_mm3rj1")

network = pylast.LastFMNetwork(api_key=API_KEY, api_secret=API_SECRET, username=username, password_hash=password_hash)

def collectTags(artist_name):
    # collecting a list of possible tags for labeling in the database
    artist = network.get_artist(artist_name)
    results = artist.get_top_tags()
    tag_list = []

    for r in results:
        if int(r[1]) >= 40: # only select more popular tags
            tag_list.append(str(r[0]).lower())
    return tag_list

def getArtistTags(artist_name, db):
    # get tags for a specific artist
    artist = network.get_artist(artist_name)
    results = artist.get_top_tags()
    tags = []

    if len(results) >= 10:
        for r in results[:10]:
            name = str(r[0]).lower()
            if name in db:
                db[name] = round((float(r[1]) / 10.0), 1) # normalize weight from scale of 1-100 to 1-10, to scale with AllMusic data
    else:
        for r in results:
            name = str(r[0]).lower()
            if name in db:
                db[name] = round((float(r[1]) / 10.0), 1) # normalize weight from scale of 1-100 to 1-10, to scale with AllMusic data

    return db
