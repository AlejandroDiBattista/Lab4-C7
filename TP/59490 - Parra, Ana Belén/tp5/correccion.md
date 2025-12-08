# Informe de errores encontrados

Se ha detectado **1 error significativo** en la implementación del código proporcionado. A continuación, se detalla el error junto con las indicaciones necesarias para corregirlo.

## Error 1: La clase `Factura` no utiliza el método `calcularDescuento` de la clase `Catalogo` para calcular los descuentos

### Detalle del error

En la clase `Factura`, el cálculo de los descuentos se realiza de manera independiente, implementando lógica propia para aplicar ofertas como **2x1** y **descuentos porcentuales**. Esto no cumple con el requerimiento de utilizar el método `calcularDescuento` de la clase `Catalogo`, lo que puede llevar a inconsistencias y duplicación de lógica.

### Corrección propuesta

Modificar la propiedad `descuentos` de la clase `Factura` para que utilice el método `calcularDescuento` de la instancia de `Catalogo`. De esta manera, se asegura que el cálculo de descuentos se centraliza en una única clase, facilitando el mantenimiento y evitando errores.

#### Pasos a seguir:

1. **Eliminar la lógica de cálculo de descuentos existente en la clase `Factura`.**

   Actualmente, la propiedad `descuentos` implementa su propia lógica para aplicar ofertas. Esta lógica debe eliminarse para delegar el cálculo al `Catalogo`.

2. **Utilizar el método `calcularDescuento` del `Catalogo` para obtener el descuento aplicado a cada producto.**

   Reemplazar la lógica existente por llamadas al método `calcularDescuento` de la clase `Catalogo`.

3. **Actualizar la generación de detalles de descuentos para reflejar correctamente las ofertas aplicadas.**

   Dado que `calcularDescuento` retorna solo el monto del descuento, se puede mantener una descripción genérica o mejorar la implementación de `Catalogo` para que también proporcione descripciones de las ofertas aplicadas si es necesario.

#### Código específico a cambiar:

Modificar la propiedad `descuentos` en la clase `Factura` de la siguiente manera:

**Código Original:**

```python
@property
def descuentos(self):
    total_descuento = 0
    descuentos_detalle = []

    for item in self.items:
        producto = item['producto']
        cantidad = item['cantidad']
        print(f"Revisando producto: {producto.nombre}, cantidad: {cantidad}")

    # Verificar si se aplica la oferta 2x1
        if any(isinstance(oferta, Oferta2x1) and producto.tipo in oferta.tipos for oferta in self.catalogo.ofertas):
            cantidad_descuento = cantidad // 2
            descuento_2x1 = cantidad_descuento * producto.precio
            total_descuento += descuento_2x1
            descuentos_detalle.append("Oferta 2x1")
            print(f"Aplicado descuento 2x1: {descuento_2x1}")

    # Verificar si se aplica un descuento porcentual
        for oferta in self.catalogo.ofertas:
            if isinstance(oferta, OfertaDescuento) and (producto.codigo in oferta.codigos or producto.tipo in oferta.tipos):
                descuento_porcentaje = oferta.descuento / 100
                descuento = producto.precio * cantidad * descuento_porcentaje
                total_descuento += descuento
                descuentos_detalle.append(f"Descuento {oferta.descuento}%")
                print(f"Aplicado descuento {oferta.descuento}%: {descuento}")
                break

    self.descuentos_detalle = descuentos_detalle
    print(f"Total descuentos: {total_descuento}")
    return total_descuento
```

**Código Corregido:**

```python
@property
def descuentos(self):
    total_descuento = 0
    descuentos_detalle = []

    for item in self.items:
        producto = item['producto']
        cantidad = item['cantidad']
        
        descuento = self.catalogo.calcularDescuento(producto, cantidad)
        if descuento > 0:
            total_descuento += descuento
            descuentos_detalle.append(f"Descuento aplicado: -${descuento:.2f}")
    
    self.descuentos_detalle = descuentos_detalle
    return total_descuento
```

### Explicación de los cambios

- **Delegación del cálculo de descuentos:** En lugar de implementar la lógica de descuentos directamente en la clase `Factura`, se utiliza el método `calcularDescuento` del `Catalogo` para obtener el monto del descuento aplicado a cada producto.
  
- **Actualización de detalles de descuentos:** Se agrega una descripción genérica que indica el monto del descuento aplicado. Si se requiere una descripción más específica de la oferta, se podría modificar el método `calcularDescuento` para que también retorne la descripción de la oferta aplicada.

## Puntaje Final

**9/10**

Se reduce 1 punto por el error significativo encontrado en la clase `Factura`. El ajuste propuesto asegura una mejor adherencia a los requerimientos y mejora la mantenibilidad del código.