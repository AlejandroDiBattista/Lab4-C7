tres_frases = ["La práctica hace al maestro","Resolver problemas es la clave del éxito","la programacion es muy buena"]

def cantidad_de_frase(frases):
    for frase in frases:
        palabras=len(frase.split()) #aqui cuento la cantidad de palabras ya que split divide la frase en palabras
        caracteres=len(frase) #aqui solo los caracteres :)
        
        print(f"la frase es: {frase}, tiene {palabras} palabras y {caracteres} caracteres")
    
cantidad_de_frase(tres_frases)