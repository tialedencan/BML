
f = open("song.txt",'r') 
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

