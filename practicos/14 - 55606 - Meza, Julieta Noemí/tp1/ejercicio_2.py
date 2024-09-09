def funcion_cuadratica(a, b, c,x):
    f_c= a * x**2 + b * x + c
    return f_c


x_y = [(0, 0),
    (1, 8),
    (2, 12),
    (3, 12),
    (5, 0)]

for a in range(-10, 11):
    for b in range(-10, 11):
        c = 0 
        puntos = True
        for x, y in x_y:
            if funcion_cuadratica(a, b, c, x) != y:
                puntos = False
                break
        if puntos:
            print(f'los puntos son: \n a={a}\n b={b}\n c={c}\n ')
print(f"Los coeficientes son:\n a = {a} \n b = {b} \n c = {c}")