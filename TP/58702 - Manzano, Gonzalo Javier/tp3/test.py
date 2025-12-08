
# def cargar_datos():
#     productos = []
#     lineas = open("datos.dat", "r").readlines()
    
#     for linea in lineas:
  
#         fecha = linea[:10].strip()
#         producto = linea[10:40].strip()
#         precio = float(linea[40:50].strip())
#         cantidad = int(linea[50:55].strip())

#         productos.append({
#             "fecha": fecha,
#             "producto": producto,
#             "precio": precio,
#             "cantidad": cantidad
#         })
    
#     return productos

# datos = cargar_datos()


def cargar_datos():
    lineas = open("datos.dat", "r").readlines()
    
    contadorMirinda = 0
    contadorSprite = 0
    contadorPepsi = 0
    contadorTorasso = 0
    
    for linea in lineas:
        fecha = linea[:10].strip()
        producto = linea[10:40].strip()
        precio = float(linea[40:50].strip())
        cantidad = int(linea[50:55].strip())

        if producto == "Mirinda":
            contadorMirinda = contadorMirinda + cantidad
        elif producto == "Pepsi Cola":
            contadorPepsi = contadorPepsi + cantidad
        elif producto == "Spite":
            contadorSprite += cantidad
        elif producto == "Torasso":
            contadorTorasso += cantidad
        else:
            print("error")

    print(f"Cantidad vendida de Pepsi cola es de {contadorPepsi} \n Cantidad vendida de Torasso {contadorTorasso}\n Cantidad vendida de Sprite {contadorSprite}\n Cantidad vendida de Mirinda {contadorMirinda}")
            
        
cargar_datos()

    

