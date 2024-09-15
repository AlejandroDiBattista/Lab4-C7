def extraer_token(expresion):
    tokens = []
    numeros = ""  

    for caracter in expresion:
        if caracter.isdigit(): 
            numeros += caracter
        else:
            if numeros:  
                tokens.append(numeros)
                numeros = ""  
            if caracter in "+-*/()":  
                tokens.append(caracter)
    if numeros: 
        tokens.append(numeros)
    
    return tokens

expresion_usuario = input("Introduce una expresión matemática: ")

resultado = extraer_token(expresion_usuario)

print("Tokens extraídos:", resultado)
