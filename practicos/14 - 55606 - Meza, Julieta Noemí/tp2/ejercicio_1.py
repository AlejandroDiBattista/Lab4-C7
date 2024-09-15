def extraer_token(lista):
    tokens=[]
    num=""
    for i in lista:
        if i.isdigit():
            num += i
        else:
                if num:
                    tokens.append(num)
                    num = "" 
                
                if i in "+-*/":
                    tokens.append(i)
            
                if i in "()":
                    tokens.append(i)
            
    return tokens

print(extraer_token("(1 + 23 * 34 + (15 + 10))"))

#pruebas
assert extraer_token("(1 + (2 * (3 + 4)))") == ["(", "1", "+", "(", "2", "*", "(", "3", "+", "4", ")", ")", ")"]
assert extraer_token("7 - 2 * (3 + 4)") == ["7", "-", "2", "*", "(", "3", "+", "4", ")"]
assert extraer_token("10 / (5 - 3)") == ["10", "/", "(", "5", "-", "3", ")"] 

print("todos los casos son correctos")           