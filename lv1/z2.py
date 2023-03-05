# Napišite program koji od korisnika zahtijeva upis jednog broja koji predstavlja
# nekakvu ocjenu i nalazi se izme ¯ du 0.0 i 1.0. Ispišite kojoj kategoriji pripada ocjena na temelju
# sljede´cih uvjeta:
# >= 0.9 A
# >= 0.8 B
# >= 0.7 C
# >= 0.6 D
# < 0.6 F
# Ako korisnik nije utipkao broj, ispišite na ekran poruku o grešci (koristite try i except naredbe).
# Takod¯er, ako je broj izvan intervala [0.0 i 1.0] potrebno je ispisati odgovarajuc´u poruku.

try:
    grade_number = input("Enter: ")

    if(float(grade_number) >= 0.9):
        print("A")
    elif(float(grade_number) >= 0.8 and float(grade_number) < 0.9):
        print("B")
    elif(float(grade_number) >= 0.7 and float(grade_number) < 0.8):
        print("C")
    elif(float(grade_number) >= 0.6 and float(grade_number) < 0.7):
        print("D")
    elif(float(grade_number) < 0.5):
        print("F")
    else:
        print("Number out of range.")

except:
    print("You didn't enter a number.")


