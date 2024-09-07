#1. Tomar una lista de frases y convertir cada frase en una lista de palabras.
#2. Luego, debe crear una nueva lista que contenga la cantidad de palabras en cada frase.
#3. Tambien debe crear un lista que contenga la cantidad de caracteres en cada frase.
#4. Finalmente, debe imprimir cada frase original junto con la cantidad de palabras que contiene.
frases = ["Gonzalo Javier Manzano de edad 21 ahora mismo est cursando 2do año de programacion", "Josefina Medina Maria desaprobo matematicas I", "Argentina le gano por 3 golas a chile en el monumental"]
def contarcaracteres (frase) :    
    return len(frase)
def contarpalabras (frase) :    
    return len(frase.split())

for x in frases:
    print(f"La frase '{x}' \ntiene {contarpalabras(x)} palabras y tiene {contarcaracteres(x)} caracteres")

# Escribe una función en Python que encuentre los valores de `a`, `b`, y `c` para que la función cuadrática `f(x) = aX^2 + bX + c` pase exactamente por los siguientes puntos:

# | x  | y  |
# |---:|---:|
# |  0 |  0 |
# |  1 |  8 |
# |  2 | 12 |
# |  3 | 12 |
# |  5 |  0 |
puntos = [(0, 0), (1, 8), (2, 12), (3, 12), (5, 0)]

def f(x, a, b, c):
    return a * x**2 + b * x + c

def buscarResul():
    for a in range(-10, 11):
        for b in range(-10, 11):
            for c in range(-10, 11):
                coincide = True
                for x, y in puntos:
                    if f(x, a, b, c) != y:
                        coincide = False
                        break
                if coincide:
                    return a, b, c
print(f"Los valores de a, b y c son: {buscarResul()}")
