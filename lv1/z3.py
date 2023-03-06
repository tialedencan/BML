
from statistics import mean

def isfloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False


array=[] #list 

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
    