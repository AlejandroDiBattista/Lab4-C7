import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

## ATENCION: Debe colocar la direccion en la que ha sido publicada la aplicacion en la siguiente linea\
# url = 'https://tp8-59348.streamlit.app/'
# Hola profesor, tuve problemas para deployar debido a problemas con la dependencia maltpotlib y sus versiones. Podria ejecutar este codigo?

def mostrar_informacion_alumno():
    with st.container(border=True):
        st.markdown('**Legajo:** 59.348')
        st.markdown('**Nombre:** Ignacio Conort')
        st.markdown('**Comisión:** C7')

mostrar_informacion_alumno()

def cargarArchivo(archivo):
    try:
        dataProductos = pd.read_csv(archivo)
        return dataProductos
    except Exception as e:
        st.error(f"Error al cargar el archivo: {e}")
        return None

def calcularValores(dataProductos):
    dataProductos['Precio_promedio'] = (dataProductos['Ingreso_total'] / dataProductos['Unidades_vendidas'])
    dataProductos['Margen_promedio'] = (dataProductos['Ingreso_total'] - dataProductos['Costo_total']) / dataProductos['Ingreso_total']
    valoresProducto = dataProductos.groupby('Producto').agg(
        Unidades_vendidas=('Unidades_vendidas', 'sum'),
        Precio_promedio=('Precio_promedio', 'mean'),
        Margen_promedio=('Margen_promedio', 'mean')
    ).reset_index()
    return valoresProducto

def graficarProducto(dataProductos, producto):
    productoDP = dataProductos[dataProductos['Producto'] == producto]
    productoDP['Fecha'] = pd.to_datetime(productoDP['Año'].astype(str) + '-' + productoDP['Mes'].astype(str) + '-01', format='%Y-%m-%d')
    dataProductosSum = productoDP.groupby(['Fecha'])[['Unidades_vendidas']].sum().reset_index()

    grafico, eje = plt.subplots(figsize=(10, 6))
    eje.plot(dataProductosSum['Fecha'], dataProductosSum['Unidades_vendidas'], label=producto)
    z = np.polyfit(mdates.date2num(dataProductosSum['Fecha']), dataProductosSum['Unidades_vendidas'], 1)
    p = np.poly1d(z)
    eje.plot(dataProductosSum['Fecha'], p(mdates.date2num(dataProductosSum['Fecha'])), linestyle='--', color='red', label='Tendencia')
    eje.set_title(f"Evolución de Ventas Mensual")
    eje.set_xlabel("Año-Mes")
    eje.set_ylabel("Unidades Vendidas")
    eje.tick_params(axis='x', rotation=45)
    eje.legend(title="Producto")
    plt.tight_layout()

    return grafico

def app():
    st.sidebar.header("Carga archivo de datos")
    st.sidebar.write("Subir archivo CSV")

    archivo = st.sidebar.file_uploader("Sube tu archivo CSV", type="csv", label_visibility="collapsed", accept_multiple_files=False, key="archivo")
    
    if archivo is not None:
        dataProductos = cargarArchivo(archivo)
        if dataProductos is not None:
            sucursalSeleccionada = st.sidebar.selectbox('Seleccionar Sucursal', ['Todas'] + dataProductos['Sucursal'].unique().tolist())
            if sucursalSeleccionada == 'Todas':
                st.title("Datos de Todas las Sucursales")
            else:
                st.title(f"Datos de {sucursalSeleccionada}")
            
            if sucursalSeleccionada != 'Todas':
                dataProductos = dataProductos[dataProductos['Sucursal'] == sucursalSeleccionada]

            valoresProducto = calcularValores(dataProductos)

            for _, row in valoresProducto.iterrows():
                producto = row['Producto']
                precio_promedio = row['Precio_promedio']
                margen_promedio = row['Margen_promedio']
                unidades_vendidas = row['Unidades_vendidas']
                st.subheader(f"{producto}")
                st.write(f"**Precio Promedio:** ${precio_promedio:.2f}")
                st.write(f"**Margen Promedio:** {margen_promedio * 100:.2f}%")
                st.write(f"**Unidades Vendidas:** {unidades_vendidas}")
                st.pyplot(graficarProducto(dataProductos, producto))
        else:
            st.error("No se pudo cargar el archivo correctamente.")
    else:
        st.info("Por favor, sube un archivo CSV para comenzar el análisis.")

if __name__ == '__main__':
    app()