def procesar_frases(frases):
    frases_procesadas = []
    palabras = []
    caracteres = []

    for frase in frases:
        frases_procesadas.append(frase.split(" "))

    
    for frase in frases_procesadas:
        cant_palabras = 0

        for palabra in frase:
            cant_palabras = cant_palabras + 1

        palabras.append(cant_palabras)

    for frase in frases:
        cant_caracteres = 0

        cant_caracteres = cant_caracteres + len(frase)

        caracteres.append(cant_caracteres)

    i = 0
    for frase in frases:
        print('La frase: "{}" tiene {} palabras y {} caracteres.'.format(frase, palabras[i], caracteres[i]))
        i = i + 1

    return 0

    


frases = ["Python es un lenguaje de programación.", "Me gusta resolver problemas con codigo.", "Las listas y los bucles son muy útiles."]

procesar_frases(frases)