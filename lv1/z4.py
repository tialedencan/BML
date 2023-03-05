# Napišite Python skriptu koja ´ce uˇcitati tekstualnu datoteku naziva song.txt.
# Potrebno je napraviti rjeˇcnik koji kao kljuˇceve koristi sve razliˇcite rijeˇci koje se pojavljuju u
# datoteci, dok su vrijednosti jednake broju puta koliko se svaka rijeˇc (kljuˇc) pojavljuje u datoteci.
# Koliko je rijeˇci koje se pojavljuju samo jednom u datoteci? Ispišite ih

f = open("song.txt",'r') #read i text je defaultno (rt)
dictionary={}
for line in f:
    for word in line.split():
        if(word in dictionary):
            new_value=dictionary[f"{word}"]+1
            dictionary.update({f"{word}":new_value})
        else:
            dictionary[f"{word}"] = 1

only_ones=0
for v in dictionary.values():
    if v == 1:
        only_ones+=1

print(only_ones)

# for item in dictionary.items():
#     print(item)
