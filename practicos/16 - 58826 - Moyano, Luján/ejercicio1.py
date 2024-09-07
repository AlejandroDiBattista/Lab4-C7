frases = ["Mi nombre es Lujan","Tengo 20 años","Soy estudiante de la UTN"]

def procesar_frases (frases):
 for frase in frases:
        palabras = len (frase.split())

        letras = len (frase)

        print(f"la frase es: {frase} , tiene {palabras} palabras y tiene {letras} letras")

procesar_frases (frases)