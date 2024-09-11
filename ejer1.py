frases = [
    "Python es el lenguaje de programación mejor pagado",
    "Este ejercicio es muy bueno",
    "El siguiente ejercicio seria el 2, basado en funciones de x"
]

def procesar_frases(frases):
    for frase in frases:
        cantidad_caracteres = len(frase)
        palabras = frase.split()
        cantidad_palabras = len(palabras)
        
        print(f"La frase: \"{frase}\"")
        print(f" tiene {cantidad_palabras} palabras y {cantidad_caracteres} caracteres\n")
    
procesar_frases(frases)