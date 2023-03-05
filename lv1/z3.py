# Napišite program koji od korisnika zahtijeva unos brojeva u beskonaˇcnoj petlji
# sve dok korisnik ne upiše „Done“ (bez navodnika). Pri tome brojeve spremajte u listu. Nakon toga
# potrebno je ispisati koliko brojeva je korisnik unio, njihovu srednju, minimalnu i maksimalnu
# vrijednost. Sortirajte listu i ispišite je na ekran. Dodatno: osigurajte program od pogrešnog unosa
# (npr. slovo umjesto brojke) na naˇcin da program zanemari taj unos i ispiše odgovaraju´cu poruku
from statistics import mean

def isfloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False


array=[] #that is a list 

while(True):
    entered_value = input()
    if(entered_value == "Done"):
        break
     
    if(isfloat(entered_value)):
        array.append(float(entered_value))
    else:
        print("Not a number.")

print(f"Entered {len(array)} numbers.")

print(mean(array))
print(min(array))
print(max(array))

array.sort()
print(array)
    