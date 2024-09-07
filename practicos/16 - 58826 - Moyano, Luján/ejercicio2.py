def coeficientes():
    for a in range(-10, 11):
        for b in range(-10, 11):
            for c in range(-10, 11):
                if (a*0**2 + b*0 + c == 0 and
                    a*1**2 + b*1 + c == 8 and
                    a*2**2 + b*2 + c == 12 and
                    a*3**2 + b*3 + c == 12 and
                    a*5**2 + b*5 + c == 0):
                    return a, b, c

a, b, c = coeficientes()
print(f" los coeficientes son: a={a}, b={b}, c={c}")
