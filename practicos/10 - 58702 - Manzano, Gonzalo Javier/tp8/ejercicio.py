import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ATENCION: Debe colocar la direccion en la que ha sido publicada la aplicacion en la siguiente linea\
# link del deploy de el Trabajo Practico Numero 8: 
# URL = 'https://ejeciciolabgonzalomanzano.streamlit.app/'

# Función para generar gráficos de evolución de ventas
def generar_grafico(datos_producto, nombre_producto):
    ventas_mensuales = datos_producto.groupby(['Año', 'Mes'])['Unidades_vendidas'].sum().reset_index()
    figura, eje = plt.subplots(figsize=(10, 6))
    eje.plot(ventas_mensuales.index, ventas_mensuales['Unidades_vendidas'], label=nombre_producto)

    # Línea de tendencia
    x_vals = np.arange(len(ventas_mensuales))
    y_vals = ventas_mensuales['Unidades_vendidas']
    coeficientes = np.polyfit(x_vals, y_vals, 1)
    linea_tendencia = np.poly1d(coeficientes)
    eje.plot(x_vals, linea_tendencia(x_vals), linestyle='--', color='red', label='Tendencia')

    eje.set_title('Evolución de Ventas')
    eje.set_xlabel('Meses')
    eje.set_ylabel('Unidades Vendidas')
    eje.set_ylim(0)
    eje.legend(title="Producto")
    eje.grid(True)

    etiquetas = [f"{row.Año}" if row.Mes == 1 else "" for row in ventas_mensuales.itertuples()]
    eje.set_xticks(range(len(ventas_mensuales)))
    eje.set_xticklabels(etiquetas)

    return figura

# Función para mostrar datos de Gonzalo Manzano
def mostrar_datos_usuario():
    st.markdown("""
    ### Información del Usuario
    **Nombre:** Gonzalo Manzano  
    **Legajo:** 58702  
    **Comisión:** 7
    """)

# Configuración de la barra lateral
st.sidebar.header("Carga de Datos")
archivo_subido = st.sidebar.file_uploader("Cargar archivo CSV", type=["csv"])

# Función auxiliar para aplicar colores
def formatear_delta(valor):
    color = "red" if valor < 0 else "green"
    return f'<span style="color:{color};">{valor:.2f}%</span>'

# Procesar datos si se sube un archivo
if archivo_subido:
    datos = pd.read_csv(archivo_subido)

    # Selección de sucursales
    lista_sucursales = ["Todas"] + datos['Sucursal'].unique().tolist()
    sucursal_actual = st.sidebar.selectbox("Filtrar por Sucursal", lista_sucursales)

    if sucursal_actual != "Todas":
        datos = datos[datos['Sucursal'] == sucursal_actual]
        st.title(f"Análisis para {sucursal_actual}")
    else:
        st.title("Análisis de Todas las Sucursales")

    # Análisis por producto
    for producto in datos['Producto'].unique():
        datos_producto = datos[datos['Producto'] == producto]
        st.subheader(f"Producto: {producto}")

        # Cálculo de métricas
        datos_producto['Precio_promedio'] = datos_producto['Ingreso_total'] / datos_producto['Unidades_vendidas']
        precio_promedio = datos_producto['Precio_promedio'].mean()
        precio_anual = datos_producto.groupby('Año')['Precio_promedio'].mean()
        variacion_precio = (precio_anual.diff() / precio_anual.shift(1)).iloc[-1] * 100

        datos_producto['Ganancia'] = datos_producto['Ingreso_total'] - datos_producto['Costo_total']
        datos_producto['Margen'] = (datos_producto['Ganancia'] / datos_producto['Ingreso_total']) * 100
        margen_promedio = datos_producto['Margen'].mean()
        margen_anual = datos_producto.groupby('Año')['Margen'].mean()
        variacion_margen = (margen_anual.diff() / margen_anual.shift(1)).iloc[-1] * 100

        unidades_vendidas = datos_producto['Unidades_vendidas'].sum()
        unidades_anual = datos_producto.groupby('Año')['Unidades_vendidas'].sum()
        variacion_unidades = (unidades_anual.diff() / unidades_anual.shift(1)).iloc[-1] * 100

        # Mostrar métricas y gráfico en columnas
        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown(
                f"**Precio Promedio:** ${precio_promedio:,.2f} ({formatear_delta(variacion_precio)})",
                unsafe_allow_html=True
            )
            st.markdown(
                f"**Margen Promedio:** {margen_promedio:.2f}% ({formatear_delta(variacion_margen)})",
                unsafe_allow_html=True
            )
            st.markdown(
                f"**Unidades Vendidas:** {unidades_vendidas:,} ({formatear_delta(variacion_unidades)})",
                unsafe_allow_html=True
            )

        with col2:
            figura = generar_grafico(datos_producto, producto)
            st.pyplot(figura)

else:
    st.warning("Por favor, sube un archivo CSV para comenzar el análisis.")
    mostrar_datos_usuario()
