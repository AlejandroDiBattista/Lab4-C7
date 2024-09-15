def es_expresion_valida(lista):
    op={"+","-","*","/"}
    num=""    
    for i in range(len(lista)):
        token = lista[i]

        if token in op:
            if i == 0 or i == len(lista) - 1:
                return False
            elif lista[i-1] in op:
                return False

        elif token == "(":
            if i < len(lista) - 1 and lista[i+1] == ")": 
                return False
            
    return True

print(es_expresion_valida (["(", "1", "+", "2", ")", "*", "3"]))
print(es_expresion_valida(["1", "+", "(", ")"]))

#prueba
assert es_expresion_valida(["(", "1", "+", "2", ")", "*", "3"]) == True
assert es_expresion_valida(["1", "+", "(", ")"]) == False
assert es_expresion_valida(["1", "*", "*", "2"]) == False
assert es_expresion_valida(["(", "1", "+", "2", ")", "/", "(", "3", "-", "4", ")"]) == True

print("todos los casos son correctos")
