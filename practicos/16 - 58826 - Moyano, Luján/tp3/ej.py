def leer_datos_ventas():
    with open("datos.dat", "r") as archivo:
        contenido = archivo.readlines()
    registro_ventas = []
    
    for linea in contenido:
        linea = linea.strip()
        if len(linea) < 55:
            continue
        
        fecha_transaccion = linea[:10].strip()
        nombre_item = linea[10:40].strip()
        valor_unitario = float(linea[40:50].strip())
        
        try:
            unidades_vendidas = int(linea[50:55].strip())
        except ValueError:
            print(f"Error al convertir la cantidad en la línea: {linea}")
            continue
        
        venta = {
            "Fecha": fecha_transaccion,
            "Producto": nombre_item,
            "Precio": valor_unitario,
            "Cantidad": unidades_vendidas
        }
        registro_ventas.append(venta)
    
    return registro_ventas


def mostrar_detalle_ventas(ventas):
    for venta in ventas:
        print("Fecha:", venta['Fecha'])
        print("Producto:", venta['Producto'])
        print("Precio:", venta['Precio'])
        print("Cantidad:", venta['Cantidad'])
        print("-" * 40)


def calcular_total_ventas(ventas):
    total_importe = 0
    total_cantidades = 0
    for registro in ventas:
        total_importe += registro['Precio'] * registro['Cantidad']
        total_cantidades += registro['Cantidad']
    return total_importe, total_cantidades


def contar_unidades_por_producto(ventas):
    conteo_unidades = {}
    for registro in ventas:
        producto = registro['Producto']
        cantidad = registro['Cantidad']
        if producto in conteo_unidades:
            conteo_unidades[producto] += cantidad
        else:
            conteo_unidades[producto] = cantidad
    return conteo_unidades


def mostrar_resumen_unidades(conteo_total):
    for producto, total in conteo_total.items():
        print(f"Producto: {producto}, Unidades Vendidas: {total}")


def calcular_precio_promedio(ventas):
    total_precio = {}
    conteo_unidades = {}
    
    for registro in ventas:
        producto = registro['Producto']
        precio = registro['Precio']
        cantidad = registro['Cantidad']
        
        if producto in total_precio:
            total_precio[producto] += precio * cantidad
            conteo_unidades[producto] += cantidad
        else:
            total_precio[producto] = precio * cantidad
            conteo_unidades[producto] = cantidad
            
    def mostrar_precio_promedio(precios_acumulados):
        for producto in precios_acumulados:
            promedio = precios_acumulados[producto] / conteo_unidades[producto]
            print(f"Producto: {producto}, Precio Promedio: ${promedio:.2f}")
    
    mostrar_precio_promedio(total_precio)


def agrupar_ventas_por_mes(ventas):
    ventas_mensuales = {}
    for registro in ventas:
        producto = registro['Producto']
        fecha = registro['Fecha']
        mes = fecha[5:7] + '-' + fecha[:4]
        cantidad = registro['Cantidad']
        
        if producto not in ventas_mensuales:
            ventas_mensuales[producto] = {}
        
        if mes in ventas_mensuales[producto]:
            ventas_mensuales[producto][mes] += cantidad
        else:
            ventas_mensuales[producto][mes] = cantidad
            
    return ventas_mensuales


def mostrar_ventas_por_mes(ventas):
    for producto, meses in ventas.items():
        for mes, total in meses.items():
            print(f"Producto: {producto}, Mes: {mes}, Unidades Vendidas: {total}")


def resumen_de_ventas_totales(ventas):
    total_unidades = {}
    total_importes = {}
    
    for registro in ventas:
        producto = registro['Producto']
        precio = registro['Precio']
        cantidad = registro['Cantidad']
        
        if producto in total_unidades:
            total_unidades[producto] += cantidad
            total_importes[producto] += precio * cantidad
        else:
            total_unidades[producto] = cantidad
            total_importes[producto] = precio * cantidad

    for producto in total_unidades:
        promedio = total_importes[producto] / total_unidades[producto]
        print(f"-> Producto: {producto}, Precio Promedio: ${promedio:.2f}, Unidades Vendidas: {total_unidades[producto]}, Importe Total: ${total_importes[producto]:.2f}")


# Procesar y mostrar los datos
ventas_info = leer_datos_ventas()
print(ventas_info)
mostrar_detalle_ventas(ventas_info)

importe_final, cantidad_total = calcular_total_ventas(ventas_info)
print(f"Las ventas fueron de ${importe_final:.2f} en {cantidad_total} unidades")

total_unidades_vendidas = contar_unidades_por_producto(ventas_info)
mostrar_resumen_unidades(total_unidades_vendidas)

calcular_precio_promedio(ventas_info)

ventas_mensuales = agrupar_ventas_por_mes(ventas_info)
mostrar_ventas_por_mes(ventas_mensuales)

resumen_de_ventas_totales(ventas_info)