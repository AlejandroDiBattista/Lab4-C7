def verificar_parentesis(tokens):
    contador=0
    for i in tokens:
        if i == "(":
            contador+=1
        elif i == ")":
            contador-=1 

        if contador < 0:
            return False 

    return contador==0       
            
print(verificar_parentesis(["(", "1", "+", "2", "+", "(", "3", "", "4", ")", "+", "(", "5", "", "6", ")", ")"]))            
print(verificar_parentesis(["(", "(", "1", "+", "2", ")", "+", "3"]))  


#pruebas
assert verificar_parentesis(["(", "1", "+", "2", "+", "(", "3", "*", "4", ")", "+", "(", "5", "*", "6", ")", ")"]) == True
assert verificar_parentesis(["(", "1", "+", "2", "*", "3"]) == False
assert verificar_parentesis(["(", "(", "1", "+", "2", ")"]) == False

print("todos los casos pasaron :) ")