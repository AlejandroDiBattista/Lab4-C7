import os

def cargar_datos_ventas():
    ruta_archivo = os.path.join(os.path.dirname(__file__), 'datos.dat')
    
    if not os.path.exists(ruta_archivo):
        print(f"Error: El archivo no está en la ruta: {ruta_archivo}")
        return []
    
    with open(ruta_archivo, "r") as archivo:
        lineas = archivo.readlines()
    
    lista_ventas = []
    
    for linea in lineas:
        linea = linea.strip()
        if len(linea) < 55:
            continue
        
        fecha = linea[:10].strip()
        producto = linea[10:40].strip()
        precio_unitario = float(linea[40:50].strip())
        
        try:
            cantidad = int(linea[50:55].strip())
        except ValueError:
            print(f"Error al convertir la cantidad en la línea: {linea}")
            continue
        
        venta = {
            "Fecha": fecha,
            "Producto": producto,
            "Precio": precio_unitario,
            "Cantidad": cantidad
        }
        lista_ventas.append(venta)
    
    return lista_ventas

def imprimir_ventas(ventas):
    for venta in ventas:
        print("Fecha:", venta['Fecha'])
        print("Producto:", venta['Producto'])
        print("Precio:", venta['Precio'])
        print("Cantidad:", venta['Cantidad'])
        print("-" * 40)

def total_ventas(ventas):
    importe_total = 0
    total_cantidades = 0
    for registro in ventas:
        importe_total += registro['Precio'] * registro['Cantidad']
        total_cantidades += registro['Cantidad']
    return importe_total, total_cantidades

def total_unidades_vendidas_por_producto(ventas):
    unidades_por_producto = {}
    for registro in ventas:
        producto = registro['Producto']
        cantidad = registro['Cantidad']
        if producto in unidades_por_producto:
            unidades_por_producto[producto] += cantidad
        else:
            unidades_por_producto[producto] = cantidad
    return unidades_por_producto

def listar_unidades_vendidas(unidades_totales):
    for producto, total in unidades_totales.items():
        print(f"Producto: {producto}, Unidades Vendidas: {total}")

def precio_promedio_por_producto(ventas):
    acumulado_precio = {}
    total_cantidades = {}
    
    for registro in ventas:
        producto = registro['Producto']
        precio = registro['Precio']
        cantidad = registro['Cantidad']
        
        if producto in acumulado_precio:
            acumulado_precio[producto] += precio * cantidad
            total_cantidades[producto] += cantidad
        else:
            acumulado_precio[producto] = precio * cantidad
            total_cantidades[producto] = cantidad
            
    def mostrar_precios(promedios):
        for producto in promedios:
            promedio = promedios[producto] / total_cantidades[producto]
            print(f"Producto: {producto}, Precio Promedio: ${promedio:.2f}")
    
    mostrar_precios(acumulado_precio)

def ventas_mensuales(ventas):
    ventas_por_mes = {}
    for registro in ventas:
        producto = registro['Producto']
        fecha = registro['Fecha']
        mes = fecha[5:7] + '-' + fecha[:4]
        cantidad = registro['Cantidad']
        
        if producto not in ventas_por_mes:
            ventas_por_mes[producto] = {}
        
        if mes in ventas_por_mes[producto]:
            ventas_por_mes[producto][mes] += cantidad
        else:
            ventas_por_mes[producto][mes] = cantidad
            
    return ventas_por_mes

def listar_ventas_por_periodo(ventas):
    for producto, meses in ventas.items():
        for mes, total in meses.items():
            print(f"Producto: {producto}, Mes: {mes}, Unidades Vendidas: {total}")

def resumen_total_ventas(ventas):
    total_cantidades = {}
    total_importes = {}
    
    for registro in ventas:
        producto = registro['Producto']
        precio = registro['Precio']
        cantidad = registro['Cantidad']
        
        if producto in total_cantidades:
            total_cantidades[producto] += cantidad
            total_importes[producto] += precio * cantidad
        else:
            total_cantidades[producto] = cantidad
            total_importes[producto] = precio * cantidad

    for producto in total_cantidades:
        promedio = total_importes[producto] / total_cantidades[producto]
        print(f"-> Producto: {producto}, Precio Promedio: ${promedio:.2f}, Unidades Vendidas: {total_cantidades[producto]}, Importe Total: ${total_importes[producto]:.2f}")

ventas_datos = cargar_datos_ventas()

if ventas_datos:
    imprimir_ventas(ventas_datos)

    importe_total, cantidad_total = total_ventas(ventas_datos)
    print(f"Las ventas fueron de ${importe_total:.2f} en {cantidad_total} unidades")

    unidades_vendidas_totales = total_unidades_vendidas_por_producto(ventas_datos)
    listar_unidades_vendidas(unidades_vendidas_totales)

    precio_promedio_por_producto(ventas_datos)

    ventas_por_mensualidades = ventas_mensuales(ventas_datos)
    listar_ventas_por_periodo(ventas_por_mensualidades)

    resumen_total_ventas(ventas_datos)
