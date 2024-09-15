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

def evaluar_parentesis(tokens):
    while '(' in tokens:
        i = 0
        while i < len(tokens):
            if tokens[i] == '(':
                j = i + 1
                nivel = 1
                while j < len(tokens) and nivel > 0:
                    if tokens[j] == '(':
                        nivel += 1
                    elif tokens[j] == ')':
                        nivel -= 1
                    j += 1
                sub_exp = tokens[i + 1:j - 1]
                resultado = resolver_sumas(resolver_multiplicaciones(sub_exp))
                tokens = tokens[:i] + [resultado] + tokens[j:]
                break
            i += 1
    return tokens

def resolver_multiplicaciones(tokens):
    i = 0
    while i < len(tokens):
        if tokens[i] == '*':
            resultado = int(tokens[i - 1]) * int(tokens[i + 1])
            tokens = tokens[:i - 1] + [resultado] + tokens[i + 2:]
            i -= 1
        else:
            i += 1
    return tokens

def resolver_sumas(tokens):
    resultado = int(tokens[0])
    i = 1
    while i < len(tokens):
        if tokens[i] == '+':
            resultado += int(tokens[i + 1])
        i += 2
    return resultado

def evaluar_expresion(tokens):
    tokens = evaluar_parentesis(tokens)
    tokens = resolver_multiplicaciones(tokens)
    resultado = resolver_sumas(tokens)
    return resultado

expresion_usuario = input("Introduce una expresión matemática: ")
tokens = extraer_token(expresion_usuario)
print(evaluar_expresion(tokens))
