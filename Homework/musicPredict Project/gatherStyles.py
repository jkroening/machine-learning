import json

file = open('styles.json')
json = json.load(file)

styles = []
genres = []
subgenres = []

# # get complete info
# for s in json['styles']:
#     styles.append(s['name'])
#     genres.append(s['genre']['name'])
#     subgenres.append(s['subgenre']['name'])

# alternate loop for gathering only specific info
for s in json['styles']:
    if s['genre']['name'] in ['Country','Electronic','Pop/Rock','R&B','Religious']:
        styles.append(s['name'])
        genres.append(s['genre']['name'])
        subgenres.append(s['subgenre']['name'])

styles_out = open('styles.txt', 'w')
genres_out = open('genres.txt', 'w')
subgenres_out = open('subgenres.txt', 'w')

for item in sorted(set(styles)):
    styles_out.write("%s\n" % item.encode('utf8'))
for item in sorted(set(genres)):
    genres_out.write("%s\n" % item.encode('utf8'))
for item in sorted(set(subgenres)):
    subgenres_out.write("%s\n" % item.encode('utf8'))
