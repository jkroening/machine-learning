import csv
import warnings
import numpy as np
import AllMusic as amg
import Last_fm as lastfm
from unidecode import unidecode

def mk_int(s):
    s = s.strip()
    return int(s) if s else 0

def getRatings(f):
    ## filter out RuntimeWarnings thrown by attempts to unicode decode str types
    warnings.simplefilter("ignore", RuntimeWarning)

    albums = []
    entry = None
    plays = []
    artist = None
    title = None

    for row in f:
        if entry is None:
            # first album coming through
            artist = unidecode(row.get('Artist'))
            title = unidecode(row.get('Album'))
            plays.append(mk_int(row.get('Plays')))
            rating = row.get('Rating')
            entry = { 'albumID' : None, 'artist' : artist, 'title' : title, 'plays' : plays, 'rating' : rating }
        elif unidecode(row.get('Artist')) == artist and unidecode(row.get('Album')) == title:
            # keep appending play counts while album is the same
            entry['plays'].append(mk_int(row.get('Plays')))
            if rating is None: # still hasn't found a rating so keep looking
                rating = row.get('Rating')
        else:
            # moving on to the next album, so get median of plays and write out previous album entry
            entry['plays'] = np.median(entry['plays'])
            albums.append(entry)
            # start next album entry
            entry = None
            artist = None
            title = None
            plays = []
            artist = unidecode(row.get('Artist'))
            title = unidecode(row.get('Album'))
            plays.append(mk_int(row.get('Plays')))
            rating = row.get('Rating')
            entry = { 'albumID' : None, 'artist' : artist, 'title' : title, 'plays' : plays, 'rating' : rating }

    return albums

def collectLastFMTags():
    with open('features/top_tags.txt') as tt:
        top_tags = tt.read().splitlines()
    f = csv.DictReader(open('2009_5.csv', 'rU'))
    my_ratings = getRatings(f)
    tags = []
    for item in my_ratings:
        tags.extend(lastfm.collectTags(item['artist']))
        print tags
    tags.extend(top_tags)
    tags_set = set(tags)
    tags = sorted(tags_set)
    with open('features/tags.csv', 'ab') as t:
        for tag in tags:
            t.write("%s\n" % tag)

def main():

    with open('features/genres.txt') as g:
        genres = g.read().splitlines()
    with open('features/moods.txt') as m:
        moods = m.read().splitlines()
    with open('features/styles_and_subgenres.txt') as s:
        styles = s.read().splitlines()
    with open('features/themes.txt') as t:
        themes = t.read().splitlines()

    with open('features/tags.txt') as l:
        tags = l.read().splitlines()

    features = ['albumID','artist','title']
    features.extend(genres)
    features.extend(moods)
    features.extend(styles)
    features.extend(themes)
    features.extend(tags)

    with open('features/features.csv', 'ab') as f:
        for feat in features:
            f.write("%s\n" % feat)

    ratings_labels = ['userID', 'albumID', 'artist', 'title', 'rating', 'plays']

    # # only do this once... initializes the new files with header rows
    # with open('albumsDB.csv', 'ab') as a:
    #         dict_writer = csv.DictWriter(a, features, dialect='excel')
    #         dict_writer.writer.writerow(features)
    # with open('ratingsDB.csv', 'ab') as r:
    #         dict_writer = csv.DictWriter(r, features, dialect='excel')
    #         dict_writer.writer.writerow(ratings_labels)

    # get all the data from the csv file
    f = csv.DictReader(open('2009_5.csv', 'rU'))
    my_ratings = getRatings(f)

    tags = []
    for item in my_ratings:
        # initialize database for each album
        db = dict.fromkeys(features, 0)

        # get ratings info for the albums in the csv file
        rating = item['rating']
        plays = item['plays']

        # get album info for the albums in the csv file
        artist = str(item['artist'])
        title = str(item['title'])

        # find the albumID and authentic AllMusic artist and title field values
        amg_artist, amg_title, albumID = amg.getAlbum(artist, title)

        if albumID is None:
            rating = { 'userID' : '000000001', 'albumID' : albumID, 'artist' : artist, 'title' : title, 'rating' : rating, 'plays' : plays }
        else:
            rating = { 'userID' : '000000001', 'albumID' : albumID, 'artist' : amg_artist, 'title' : amg_title, 'rating' : rating, 'plays' : plays }

        print rating

        # if album was found in AllMusic add it and it's metadata to the database
        if albumID is not None:
            db['albumID'] = albumID
            db['artist'] = amg_artist
            db['title'] = amg_title
            db = amg.getAlbumData(albumID, db)
        else:
            db['artist'] = artist
            db['title'] = title

        # add Last.fm tag metadata to the database (tags are per artist only, so they will be the same across albums)
        db = lastfm.getArtistTags(artist, db)

        # NOTE:  If you see a tag and a genre/mood/style/theme that are the same you can tell it's from last.fm because it's in lowercase

        with open('albumsDB.csv', 'ab') as a:
            dict_writer = csv.DictWriter(a, delimiter=',', fieldnames=features)
            dict_writer.writerow(db)

        with open('ratingsDB.csv', 'ab') as r:
            dict_writer = csv.DictWriter(r, delimiter=',', fieldnames=ratings_labels)
            dict_writer.writerow(rating)


if __name__ == "__main__":
    main()
