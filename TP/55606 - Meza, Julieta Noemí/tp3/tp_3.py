def leer_ventas():
    with open("datos.dat", "r") as archivo:
        lineas = archivo.readlines()
    
    ventas_registradas = []
    
    for linea in lineas:
        linea = linea.strip()
        if len(linea) < 55:
            continue
        
        fecha = linea[:10].strip()
        producto = linea[10:40].strip()
        precio_unitario = float(linea[40:50].strip())
        
        try:
            cantidad_vendida = int(linea[50:55].strip())
        except ValueError:
            print(f"Problema al convertir la cantidad en: {linea}")
            continue
        
        venta = {
            "Fecha": fecha,
            "Producto": producto,
            "Precio": precio_unitario,
            "Cantidad": cantidad_vendida
        }
        ventas_registradas.append(venta)
    
    return ventas_registradas


def imprimir_ventas(ventas):
    for venta in ventas:
        print("Fecha:", venta['Fecha'])
        print("Producto:", venta['Producto'])
        print("Precio:", venta['Precio'])
        print("Cantidad:", venta['Cantidad'])
        print("-" * 40)


def total_ventas(ventas):
    total_precio = 0
    total_cantidades = 0
    for venta in ventas:
        total_precio += venta['Precio'] * venta['Cantidad']
        total_cantidades += venta['Cantidad']
    return total_precio, total_cantidades


def contar_cantidad_por_producto(ventas):
    conteo_productos = {}
    for venta in ventas:
        producto = venta['Producto']
        cantidad = venta['Cantidad']
        if producto in conteo_productos:
            conteo_productos[producto] += cantidad
        else:
            conteo_productos[producto] = cantidad
    return conteo_productos


def mostrar_cantidad_productos(conteo_productos):
    for producto, total in conteo_productos.items():
        print(f"Producto: {producto}, Cantidades Vendidas: {total}")


def calcular_precio_medio(ventas):
    total_precio_por_producto = {}
    total_cantidades_por_producto = {}
    
    for venta in ventas:
        producto = venta['Producto']
        precio = venta['Precio']
        cantidad = venta['Cantidad']
        
        if producto in total_precio_por_producto:
            total_precio_por_producto[producto] += precio * cantidad
            total_cantidades_por_producto[producto] += cantidad
        else:
            total_precio_por_producto[producto] = precio * cantidad
            total_cantidades_por_producto[producto] = cantidad
            
    for producto in total_precio_por_producto:
        promedio = total_precio_por_producto[producto] / total_cantidades_por_producto[producto]
        print(f"Producto: {producto}, Precio Promedio: ${promedio:.2f}")


def ventas_por_mes(ventas):
    resumen_mensual = {}
    for venta in ventas:
        producto = venta['Producto']
        fecha = venta['Fecha']
        mes = fecha[5:7] + '-' + fecha[:4]
        cantidad = venta['Cantidad']
        
        if producto not in resumen_mensual:
            resumen_mensual[producto] = {}
        
        if mes in resumen_mensual[producto]:
            resumen_mensual[producto][mes] += cantidad
        else:
            resumen_mensual[producto][mes] = cantidad
            
    return resumen_mensual


def mostrar_resumen_mensual(resumen):
    for producto, meses in resumen.items():
        for mes, cantidad in meses.items():
            print(f"Producto: {producto}, Mes: {mes}, Cantidades Vendidas: {cantidad}")


def resumen_total(ventas):
    total_cantidades = {}
    total_importes = {}
    
    for venta in ventas:
        producto = venta['Producto']
        precio = venta['Precio']
        cantidad = venta['Cantidad']
        
        if producto in total_cantidades:
            total_cantidades[producto] += cantidad
            total_importes[producto] += precio * cantidad
        else:
            total_cantidades[producto] = cantidad
            total_importes[producto] = precio * cantidad

    for producto in total_cantidades:
        promedio = total_importes[producto] / total_cantidades[producto]
        print(f"-> Producto: {producto}, Precio Promedio: ${promedio:.2f}, Total Vendido: {total_cantidades[producto]}, Importe Total: ${total_importes[producto]:.2f}")

ventas_info = leer_ventas()
imprimir_ventas(ventas_info)

total_importe, total_unidades = total_ventas(ventas_info)
print(f"Las ventas totales fueron de ${total_importe:.2f} en {total_unidades} unidades")

conteo_productos = contar_cantidad_por_producto(ventas_info)
mostrar_cantidad_productos(conteo_productos)

calcular_precio_medio(ventas_info)

resumen_mensual = ventas_por_mes(ventas_info)
mostrar_resumen_mensual(resumen_mensual)

resumen_total(ventas_info)