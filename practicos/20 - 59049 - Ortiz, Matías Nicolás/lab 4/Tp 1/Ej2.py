def funcion_cuadratica(a, b, c, x):
    return a * x ** 2 + b * x + c

def prueba_y_error():
    puntos = [(0, 0), (1, 8), (2, 12), (3, 12), (5, 0)]
    
    min_valor = -10
    max_valor = 10
    paso = 1
    
    a = min_valor
    while a <= max_valor:
        b = min_valor
        while b <= max_valor:
            c = min_valor
            while c <= max_valor:
                error = 0
                for x, y in puntos:
                    y_calculado = funcion_cuadratica(a, b, c, x)
                    if round(y_calculado, 5) != round(y, 5):
                        error += abs(y_calculado - y)
                if error == 0:
                    return a, b, c
                c += paso
            b += paso
        a += paso
    
    return None

resultado = prueba_y_error()
if resultado is not None:
    a, b, c = resultado
    print(f"Solución encontrada: a = {a}, b = {b}, c = {c}")
else:
    print("No se encontró solución exacta.")
