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

def expresion_valida(tokens):
    esperando_num = True  
    n = len(tokens)
    for i in range(n):
        token = tokens[i]
        if token.isdigit():  
            esperando_num = False  
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
            esperando_num = False  
            i = j - 1  
        elif token == ')':
            return False  
        elif token in "+-*/":
            if esperando_num:
                return False  
            esperando_num = True 
        else:
            return False  
    return not esperando_num

expresion_ingresada = input("escriba una expresion matematica: ")

tokens = extraer_token(expresion_ingresada)

print(expresion_valida(tokens))