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

def expresion_valida(tokens):
    esperando_numeros = True  
    n = len(tokens)
    for i in range(n):
        token = tokens[i]
        if token.isdigit():  
            esperando_numeros = False  
        elif token == '(':
            j = i + 1
            nivel_par = 1  
            while j < n and nivel_par > 0:
                if tokens[j] == '(':
                    nivel_par += 1
                elif tokens[j] == ')':
                    nivel_par -= 1
                j += 1
            if nivel_par != 0:
                return False
            if not expresion_valida(tokens[i + 1:j - 1]):
                return False
            esperando_numeros = False  
            i = j - 1  
        elif token == ')':
            return False  
        elif token in "+-*/":
            if esperando_numeros:
                return False  
            esperando_numeros = True 
        else:
            return False  
    return not esperando_numeros

expresion_usuario = input("Introduce una expresión matemática: ")

tokens = extraer_token(expresion_usuario)

print(expresion_valida(tokens))
