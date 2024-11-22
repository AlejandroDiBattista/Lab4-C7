import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## ATENCION: Debe colocar la direccion en la que ha sido publicada la aplicacion en la siguiente linea\
# url = 'https://tp8-58794.streamlit.app/'


def visualizar_informacion_usuario():
    with st.container():
        st.markdown('**Legajo:** 58.794')
        st.markdown('**Nombre:** Patti Mauro Luciano')
        st.markdown('**Comision:** C7')

def crear_grafico_tendencia(datos, etiqueta_producto):
    datos_agrupados = datos.groupby(['Año', 'Mes'])['Unidades_vendidas'].sum().reset_index()

    figura, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(len(datos_agrupados)), datos_agrupados['Unidades_vendidas'], label=etiqueta_producto, color='green')

    x = np.arange(len(datos_agrupados))
    y = datos_agrupados['Unidades_vendidas']
    coef = np.polyfit(x, y, 1)
    tendencia = np.poly1d(coef)
    ax.plot(x, tendencia(x), linestyle='--', color='red', label='Tendencia')

    ax.set_title(f'Evolución de Ventas: {etiqueta_producto}', fontsize=14)
    ax.set_xlabel('Año-Mes', fontsize=12)
    ax.set_ylabel('Unidades', fontsize=12)
    ax.set_xticks(range(len(datos_agrupados)))
    etiquetas = [
        f"{fila.Año}" if fila.Mes == 1 else ""
        for fila in datos_agrupados.itertuples()
    ]
    ax.set_xticklabels(etiquetas, rotation=45)
    ax.legend(title='Detalles')
    ax.grid(True, linestyle='--', alpha=0.5)

    return figura

st.sidebar.header("Subir un Archivo")
archivo = st.sidebar.file_uploader("Cargar archivo CSV", type=["csv"])

if archivo is not None:
    datos = pd.read_csv(archivo)

    sucursales_disponibles = ["Todas"] + datos['Sucursal'].unique().tolist()
    sucursal_actual = st.sidebar.selectbox("Elige Sucursal", sucursales_disponibles)

    if sucursal_actual != "Todas":
        datos = datos[datos['Sucursal'] == sucursal_actual]
        st.title(f"Información para: {sucursal_actual}")
    else:
        st.title("Datos Consolidados de Todas las Sucursales")

    productos_disponibles = datos['Producto'].unique()
    for item_producto in productos_disponibles:
        with st.container():
            st.subheader(f"Análisis del Producto: {item_producto}")
            datos_producto = datos[datos['Producto'] == item_producto]

            datos_producto['Precio_unitario'] = datos_producto['Ingreso_total'] / datos_producto['Unidades_vendidas']
            precio_promedio = datos_producto['Precio_unitario'].mean()

            precio_anual = datos_producto.groupby('Año')['Precio_unitario'].mean()
            cambio_precio_anual = precio_anual.pct_change().mean() * 100

            datos_producto['Beneficio'] = datos_producto['Ingreso_total'] - datos_producto['Costo_total']
            datos_producto['Margen_beneficio'] = (datos_producto['Beneficio'] / datos_producto['Ingreso_total']) * 100
            beneficio_medio = datos_producto['Margen_beneficio'].mean()

            beneficio_anual = datos_producto.groupby('Año')['Margen_beneficio'].mean()
            cambio_beneficio_anual = beneficio_anual.pct_change().mean() * 100

            unidades_vendidas = datos_producto['Unidades_vendidas'].sum()
            unidades_anuales = datos_producto.groupby('Año')['Unidades_vendidas'].sum()
            cambio_unidades_anuales = unidades_anuales.pct_change().mean() * 100

            st.metric(label="Precio Promedio", value=f"${precio_promedio:,.2f}", delta=f"{cambio_precio_anual:.2f}%")
            st.metric(label="Beneficio Medio", value=f"{beneficio_medio:.2f}%", delta=f"{cambio_beneficio_anual:.2f}%")
            st.metric(label="Total Unidades", value=f"{unidades_vendidas:,}", delta=f"{cambio_unidades_anuales:.2f}%")

            grafico_ventas = crear_grafico_tendencia(datos_producto, item_producto)
            st.pyplot(grafico_ventas)
else:
    st.subheader("Por favor, carga un archivo CSV para comenzar.")
    visualizar_informacion_usuario()

