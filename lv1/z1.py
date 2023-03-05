#z1

def total_euro(working_hours, price_per_hour):
    return working_hours * price_per_hour

working_hours = input("Enter hours: ")
price_h = input("Enter price: ")

print("Radni sati: {sati} h".format(sati=working_hours))
print(f"eura/h: {price_h} ")
print("Ukupno: {ukupno} eura".format(ukupno = total_euro(float(working_hours), float(price_h))))
