def f(x, a, b, c):
    return a * x**2 + b * x + c

puntos = [(0, 0), (1, 8), (2, 12), (3, 12), (5, 0)]

for a in range(-10, 11):
    for b in range(-10, 11):
        c = 0  
        todos_los_puntos = True
        for x, y in puntos:
            if f(x, a, b, c) != y:
                todos_los_puntos = False
                break
        if todos_los_puntos:
            print(f"Los posibles valores que funcionan son: a={a}, b={b}, c={c}")
print(f"Los coeficientes son: a = {a}, b = {b}, c = {c}")
