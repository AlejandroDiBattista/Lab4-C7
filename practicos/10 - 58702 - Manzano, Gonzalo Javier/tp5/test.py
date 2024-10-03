class Producto:
    def __init__(self, codigo, nombre, precio, tipo, cantidad):
        if len(codigo) != 4:
            raise ValueError("Código incorrecto. Debe tener 4 caracteres.")
        if not (1 <= len(nombre) <= 100):
            raise ValueError("Nombre incorrecto. Debe tener entre 1 y 100 caracteres.")
        if not (10 <= precio <= 10000):
            raise ValueError("Precio incorrecto. Debe estar entre 10 y 10000.")
        if not (0 <= len(tipo) <= 20):
            raise ValueError("Tipo incorrecto. Debe tener entre 0 y 20 caracteres.")
        if not (0 <= cantidad <= 100):
            raise ValueError("Cantidad incorrecta. Debe estar entre 0 y 100.")

        # Asignamos los valores si pasan las validaciones
        self.codigo = codigo
        self.nombre = nombre
        self.precio = precio
        self.tipo = tipo
        self.cantidad = cantidad
    
    def set_precio(self, nuevo_precio):
        if 10 <= nuevo_precio <= 10000:
            self.precio = nuevo_precio
        else:
            print("Error: Precio debe estar entre 10 y 10000.")
    
    def set_cantidad(self, nueva_cantidad):
        if 0 <= nueva_cantidad <= 100:
            self.cantidad = nueva_cantidad
        else:
            print("Error: Cantidad debe estar entre 0 y 100.")
    
    def valorTotal(self):
        return self.precio * self.cantidad

# Ejemplo de uso:
p1 = Producto('0001', 'c', 50, 'gaseosa', 10)
p1.precio = -1
print(p1.precio)
print(p1.valorTotal())