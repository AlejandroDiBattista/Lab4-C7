def extraer_token(expresion):
    tokens = []
    num = ""  

    for caracter in expresion:
        if caracter.isdigit(): 
            num += caracter
        else:
            if num:  
                tokens.append(num)
                num = ""  
            if caracter in "+-*/()":  
                tokens.append(caracter)
    if num: 
        tokens.append(num)
    
    return tokens

expresion_ingresada = input("escriba una expresion matematica: ")

resultado = extraer_token(expresion_ingresada)

print("tokens extraidos:",resultado)