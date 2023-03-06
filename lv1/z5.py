
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

# a)
print(ham_words/ham_sentences)
print(spam_words/spam_sentences)

# b)
print(spam_with_exclamation_mark)