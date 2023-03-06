
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


