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

def verificar_parentesis(tokens):
    contador = 0
    for token in tokens:
        if token == '(':  
            contador += 1
        elif token == ')': 
            contador -= 1
            if contador < 0:  
                return False
    
    return contador == 0  

expresion_usuario = input("Introduce una expresión matemática: ")
tokens = extraer_token(expresion_usuario)
print (verificar_parentesis(tokens))