def extraer_token(expresion):
    tokens = []
    num = "" 
    
    for caracter in expresion:
        if caracter.isdigit(): 
            num += caracter
        else:
            if num :  
                tokens.append(num)
                num = ""  
            if caracter in "+-*/()": 
                tokens.append(caracter) 
    if num: 
        tokens.append(num)
    
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

expresion_ingresada = input("escriba una expresion matematica: ")
tokens = extraer_token(expresion_ingresada)
print (verificar_parentesis(tokens))