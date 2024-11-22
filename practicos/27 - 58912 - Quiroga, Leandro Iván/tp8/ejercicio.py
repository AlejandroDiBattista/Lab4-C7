import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## ATENCION: Debe colocar la direccion en la que ha sido publicada la aplicacion en la siguiente linea\
# url = 'https://tp8-58912.streamlit.app/'

def mostrar_datos_usuario():
    with st.container(border=True):
        st.markdown('**Legajo:** 58.912')
        st.markdown('**Nombre:** Leandro Ivan Quiroga')
        st.markdown('**Comisión:** C7')

mostrar_datos_usuario()


def generar_grafico(data, nombre_producto):
    agrupado = data.groupby(['Año', 'Mes'])['Unidades_vendidas'].sum().reset_index()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(len(agrupado)), agrupado['Unidades_vendidas'], label=nombre_producto)
    x = np.arange(len(agrupado))
    y = agrupado['Unidades_vendidas']
    coef = np.polyfit(x, y, 1)
    tendencia = np.poly1d(coef)
    ax.plot(x, tendencia(x), linestyle='--', color='red', label='Tendencia')
    ax.set_title('Evolución Mensual de Ventas')
    ax.set_xlabel('Año-Mes')
    ax.set_xticks(range(len(agrupado)))
    etiquetas = [f"{row.Año}" if row.Mes == 1 else "" for row in agrupado.itertuples()]
    ax.set_xticklabels(etiquetas)
    ax.set_ylabel('Unidades Vendidas')
    ax.set_ylim(0, None)
    ax.legend(title='Producto')
    ax.grid(True)
    return fig

st.sidebar.header("Aqui Puedes Cargar Archivo")
archivo = st.sidebar.file_uploader("Sube el archivo csv", type=["csv"])

if archivo is not None:
    datos = pd.read_csv(archivo)
    sucursales = ["Todas"] + datos['Sucursal'].unique().tolist()
    sucursal = st.sidebar.selectbox("Sucursal", sucursales)
    if sucursal != "Todas":
        datos = datos[datos['Sucursal'] == sucursal]
        st.title(f"📊 Datos de {sucursal}")
    else:
        st.title("📊 Datos de Todas las Sucursales")
    productos = datos['Producto'].unique()
    for producto in productos:
        with st.container( border= True):
            st.subheader(f"📦 Producto: {producto}")
            
            data_producto = datos[datos['Producto'] == producto]
            data_producto['Precio_promedio'] = data_producto['Ingreso_total'] / data_producto['Unidades_vendidas']
            precio_promedio = data_producto['Precio_promedio'].mean()
            variacion_precio = data_producto.groupby('Año')['Precio_promedio'].mean().pct_change().mean() * 100
            data_producto['Ganancia'] = data_producto['Ingreso_total'] - data_producto['Costo_total']
            data_producto['Margen'] = (data_producto['Ganancia'] / data_producto['Ingreso_total']) * 100
            margen_promedio = data_producto['Margen'].mean()
            variacion_margen = data_producto.groupby('Año')['Margen'].mean().pct_change().mean() * 100
            unidades_promedio = data_producto['Unidades_vendidas'].mean()
            unidades_total = data_producto['Unidades_vendidas'].sum()
            variacion_unidades = data_producto.groupby('Año')['Unidades_vendidas'].sum().pct_change().mean() * 100
            
            
            col1, col2, col3 = st.columns(3)

            col1.metric(label="💲 Precio Promedio", value=f"${precio_promedio:,.0f}".replace(",", "."), delta=f"{variacion_precio:.2f}%")
            col2.metric(label="📈 Margen Promedio", value=f"{margen_promedio:.0f}%".replace(",", "."), delta=f"{variacion_margen:.2f}%")
            col3.metric(label="📊 Unidades Vendidas", value=f"{unidades_total:,.0f}".replace(",", "."), delta=f"{variacion_unidades:.2f}%")

            fig = generar_grafico(data_producto, producto)
            st.pyplot(fig)