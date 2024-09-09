frases = ["Mi nombre es Paula","soy estudiante de Programación","vivo en Tucumán","estudio en la UTN"]

def cantidad_frases(frases):
    for frase in frases:
        cantidad_palabras=len(frase.split())
        cantidad_caracteres=len(frase)
        
        print(f"La frase es: {frase}, tiene {cantidad_palabras} palabras y {cantidad_caracteres} caracteres.")
    
cantidad_frases(frases)