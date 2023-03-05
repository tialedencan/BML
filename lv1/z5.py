# Napišite Python skriptu koja ´ce uˇcitati tekstualnu datoteku naziva SMSSpamCollection.txt
# [1]. Ova datoteka sadrži 5574 SMS poruka pri ˇcemu su neke oznaˇcene kao spam, a neke kao ham.
# Primjer dijela datoteke:
# ham Yup next stop.
# ham Ok lar... Joking wif u oni...
# spam Did you hear about the new "Divorce Barbie"? It comes with all of Ken’s stuff!
# a) Izraˇcunajte koliki je prosjeˇcan broj rijeˇci u SMS porukama koje su tipa ham, a koliko je
# prosjeˇcan broj rijeˇci u porukama koje su tipa spam.
# b) Koliko SMS poruka koje su tipa spam završava uskliˇcnikom ?


ham_sentences = 0
spam_sentences = 0
ham_words = 0
spam_words = 0
spam_with_exclamation_mark = 0
f = open("SMSSpamCollection.txt", encoding="utf8")

for line in f:
    words = line.split()
    if(words[0] == "ham"):
        ham_sentences += 1
        ham_words+=len(words)
    else:
        spam_sentences += 1
        spam_words+=len(words)
        if(words[len(words)-1]=='!'):
            spam_with_exclamation_mark+=1


print(ham_words/ham_sentences)
print(spam_words/spam_sentences)
print(spam_with_exclamation_mark)