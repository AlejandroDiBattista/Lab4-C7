def evaluar(tokens):
    while "(" in tokens:
        for i in range(len(tokens)):
            if tokens[i] == "(":
                inicio = i
            elif tokens[i] == ")":
                fin = i
                
                sublista = tokens[inicio + 1:fin]
                resultado = evaluar(sublista)
                
                tokens = tokens[:inicio] + [str(resultado)] + tokens[fin + 1:]
                break

    i = 0
    while i < len(tokens):
        if tokens[i] == "*":

            resultado = int(tokens[i - 1]) * int(tokens[i + 1]) 
            tokens = tokens[:i - 1] + [str(resultado)] + tokens[i + 2:]
            i = 0  
        else:
            i += 1
    resultado = int(tokens[0])
    i = 1
    while i < len(tokens):
        if tokens[i] == "+":
            resultado += int(tokens[i + 1])
        i += 2
    return resultado

#prueba
assert evaluar(["(", "1", "+", "2", ")", "*", "3"]) == 9
assert evaluar(["1", "+", "2", "*", "3"]) == 7
assert evaluar(["(", "1", "+", "2", ")", "+", "(", "3", "*", "4", ")"]) == 15
assert evaluar(["10", "+", "(", "5", "*", "3", ")", "+", "2"]) == 27
assert evaluar(["(", "2", "+", "3", ")", "*", "(", "4", "+", "1", ")"]) == 25

print("Los casos son correctos")