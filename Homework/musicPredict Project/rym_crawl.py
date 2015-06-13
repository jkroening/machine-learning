from lxml import html
import requests
from unidecode import unidecode

base_genres = []
with open('genres_to_search.txt', 'rb') as f:
    for line in f:
        base_genres.append(line.strip('\n'))

base_genres = sorted(set(base_genres))
genres = []

for base in base_genres:
    search_term = base.split(' ')[0]
    for b in base.split(' ')[1:]:
        search_term = '%s+%s' % (search_term, b)
    page = requests.get('https://rateyourmusic.com/genre/%s' % search_term)
    tree = html.fromstring(page.text)
    sub_genres = tree.xpath('//a[@class="genre"]/text()')
    print base
    with open('rym_genres.txt', 'ab') as f:
        for genre in sub_genres:
            f.write("%s\n" % unidecode(genre))

# all_genres = sorted(set(base_genres.extend(genres)))

# page = requests.get('https://rateyourmusic.com/rgenre/')
# # page = requests.get('https://rateyourmusic.com/release/album/jmsn/%E2%80%A0priscilla%E2%80%A0/')
# tree = html.fromstring(page.text)

# # this will create a list of genres
# genres = tree.xpath('//a[@class="genre"]/text()')

