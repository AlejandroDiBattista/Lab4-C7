def prueba_y_error():
    puntos = [(0,0), (1,8), (2,12), (3,12), (5,0)]
    
    for a in range(-10,11):
        for b in range(-10,11):
            for c in range(-10,11):
                es_correcto = True 
                
                for x, y in puntos:
                    y_calculado = a * x**2 + b * x + c
                    
                    if y_calculado != y:
                        es_correcto = False
                        break
                    
                if es_correcto:
                    return a, b, c
                
a, b, c = prueba_y_error()
print(f"Los valores correctos serían: a = {a}, b = {b} y c = {c}")