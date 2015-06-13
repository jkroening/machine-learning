import sys
import json
import urllib
import time
import hashlib
import requests
import warnings
from unidecode import unidecode
from time import sleep # necessary to avoid the calls/second limit of AllMusic API

api_url = 'http://api.rovicorp.com'

key = '6ray5zcnumxeuge9sb75yqgu'
secret = 'CsnrmAAEnu'

# def __init__(self, contact=None):
#   self.contact = contact

def _sig():
    timestamp = int(time.time())

    m = hashlib.md5()
    m.update(key)
    m.update(secret)
    m.update(str(timestamp))

    return m.hexdigest()

def get(resource, params=None):
    """Take a dict of params, and return what we get from the api"""

    if not params:
        params = {}

    params = urllib.urlencode(params)

    sig = _sig()

    url = "%s/%s?%s&apikey=%s&sig=%s" % (api_url, resource, params, key, sig)

    resp = requests.get(url)

    if resp.status_code != 200:
        print resp.status_code
        print "Search failed."
        if resp.status_code == 403:
            print "Daily quota reached. Try again tomorrow."
            sys.exit(0)
        if resp.status_code == 404:
            print "Metadata doesn't exist."
            return None
        pass

    return resp.json()

def getAlbum(artist_name, album_title):
    ## filter out RuntimeWarnings thrown by attempts to unicode decode str types
    warnings.simplefilter("ignore", RuntimeWarning)

    albumID, artist, title = [None] * 3
    artist_name = artist_name.strip('-')
    query = artist_name + "+" + album_title
    payload = { 'entitytype' : 'album', 'query' : query, 'rep' : 1, 'size' : 3, 'offset' : 0, 'language' : 'en', 'country' : 'US', 'format' : 'json' }
    json = get('search/v2.1/music/search', payload)
    for item in json.get('searchResponse').get('results'):
        amg_title = "".join(unidecode(item['album']['title']).lower().strip().split(" "))
        query_title = "".join(album_title.lower().strip().split(" "))
        amg_artist = "".join(unidecode(item['album']['primaryArtists'][0]['name']).lower().strip().split(" "))
        query_artist = "".join(artist_name.lower().strip().split(" "))
        if amg_title in query_title or amg_artist in query_artist:
            artist = unidecode(item.get('album').get('primaryArtists')[0].get('name'))
            title = unidecode(item.get('album').get('title'))
            albumID = item.get('id')
        sleep(0.25)
        if albumID is not None:
            return artist, title, albumID
        else:
            print "Album not found in AllMusic database."
            return None, None, None

def getAlbumData(albumID, db):
    ## filter out RuntimeWarnings thrown by attempts to unicode decode str types
    warnings.simplefilter("ignore", RuntimeWarning)

    genres, moods, styles, themes = [], [], [], []
    values = { 'genres' : None, 'moods' : None, 'styles' : None, 'themes' : None }
    payload = { 'albumid' : albumID, 'country' : 'US', 'language' : 'en', 'format' : 'json' }

    try:
        genre_json = get('data/v1.1/album/info', payload)
        for genre in genre_json.get('album').get('genres'):
            name = genre.get('name').encode()
            if name in db:
                db[name] = genre.get('weight')
    except Exception as e:
        pass
    sleep(0.25)

    try:
        mood_json = get('data/v1.1/album/moods', payload)
        for mood in mood_json.get('moods'):
            name = mood.get('name').encode()
            if name in db:
                if name == 'Playful':
                    db['Playful (Mood)'] = mood.get('weight')
                if name == 'Yearning':
                    db['Yearning (Mood)'] = mood.get('weight')
                else:
                    db[name] = mood.get('weight')
    except Exception as e:
        pass
    sleep(0.25)

    try:
        style_json = get('data/v1.1/album/styles', payload)
        for style in style_json.get('styles'):
            name = style.get('name').encode()
            if name in db:
                db[name] = style.get('weight')
    except Exception as e:
        pass
    sleep(0.25)

    try:
        theme_json = get('data/v1.1/album/themes', payload)
        for theme in theme_json.get('themes'):
            name = theme.get('name').encode()
            if name in db:
                if name == 'Playful':
                    db['Playful (Theme)'] = theme.get('weight')
                if name == 'Yearning':
                    db['Yearning (Theme)'] = theme.get('weight')
                else:
                    db[name] = theme.get('weight')
    except Exception as e:
        pass
    sleep(0.25)

    return db


