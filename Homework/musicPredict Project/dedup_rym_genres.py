from lxml import html
import requests
from unidecode import unidecode

genres = []
with open('rym_genres.txt', 'rb') as f:
    for line in f:
        genres.append(line.strip('\n'))

genres = sorted(set(genres))

with open('rym_genres.txt', 'wb') as f:
    for g in genres:
        f.write("%s\n" % unidecode(g))
