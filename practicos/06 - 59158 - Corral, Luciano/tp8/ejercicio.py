import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from scipy import stats

# url ='https://luciano-corral-tp8-59158.streamlit.app'

# Función para mostrar los detalles del estudiante
def mostrar_detalles_usuario():
    with st.expander("Detalles del Alumno", expanded=True):
        st.markdown("#### Información del Alumno")
        st.markdown("**Legajo:** 59158")
        st.markdown("**Nombre Completo:** Luciano Corral")
        st.markdown("**Comisión:** C7")

# Función para cargar archivo CSV en el centro de la página
def cargar_datos_csv():
    st.title("Carga de Datos CSV")
    archivo = st.file_uploader("Selecciona tu archivo CSV", type="csv", label_visibility="collapsed")
    if archivo:
        return pd.read_csv(archivo)
    return None

# Función para procesar los datos y generar gráficos
def mostrar_metricas_y_graficos(datos, sucursal_seleccionada):
    if sucursal_seleccionada != "Todas":
        datos = datos[datos["Sucursal"] == sucursal_seleccionada]

    productos_distintos = datos["Producto"].unique()

    for producto in productos_distintos:
        datos_producto = datos[datos["Producto"] == producto]

        # Validaciones de datos
        if datos_producto["Ingreso_total"].isnull().any() or datos_producto["Unidades_vendidas"].isnull().any():
            st.error(f"El producto '{producto}' tiene datos faltantes.")
            continue
        if (datos_producto["Ingreso_total"] < 0).any():
            st.error(f"El producto '{producto}' tiene valores negativos en 'Ingreso_total'.")
            continue
        if (datos_producto["Unidades_vendidas"] <= 0).any():
            st.error(f"El producto '{producto}' tiene valores no positivos en 'Unidades_vendidas'.")
            continue

        # Cálculo de métricas
        unidades_totales = datos_producto["Unidades_vendidas"].sum()
        ingresos_totales = datos_producto["Ingreso_total"].sum()
        costos_totales = datos_producto["Costo_total"].sum()

        precio_unitario_promedio = ingresos_totales / unidades_totales
        margen_unitario_promedio = (ingresos_totales - costos_totales) / ingresos_totales * 100

        # Mostrar las métricas calculadas
        st.subheader(f"Análisis del Producto: {producto}")
        st.write(f"**Precio Promedio:** ${precio_unitario_promedio:,.2f}")
        st.write(f"**Margen Promedio:** {margen_unitario_promedio:.2f}%")
        st.write(f"**Unidades Vendidas:** {unidades_totales:,}")

        # Gráfico de evolución de ventas
        mostrar_grafico_evolucion(datos_producto, producto)

# Función para mostrar el gráfico de evolución de ventas con Matplotlib
def mostrar_grafico_evolucion(datos_producto, producto):
    # Convertir la columna de fechas
    datos_producto['Fecha'] = pd.to_datetime(datos_producto['Año'].astype(str) + '-' + datos_producto['Mes'].astype(str) + '-01')
    datos_producto.sort_values('Fecha', inplace=True)

    # Crear gráfico de líneas
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(datos_producto['Fecha'], datos_producto['Unidades_vendidas'], label=f"Ventas de {producto}", color='royalblue')

    # Agregar línea de tendencia
    x = np.arange(len(datos_producto))
    y = datos_producto["Unidades_vendidas"].values
    slope, intercept, _, _, _ = stats.linregress(x, y)
    tendencia = slope * x + intercept
    ax.plot(datos_producto['Fecha'], tendencia, label="Tendencia", linestyle='--', color='tomato')

    # Etiquetas y título
    ax.set_title(f"Evolución de Ventas Mensuales - {producto}")
    ax.set_xlabel("Mes")
    ax.set_ylabel("Unidades Vendidas")
    ax.legend()

    # Mostrar gráfico interactivo en Streamlit
    st.pyplot(fig)

# Funciones auxiliares para cálculos por año
def calcular_precio_por_ano(datos_producto, anio):
    df = datos_producto[datos_producto["Año"] == anio]
    return df["Ingreso_total"].sum() / df["Unidades_vendidas"].sum() if df["Unidades_vendidas"].sum() > 0 else 0

def calcular_margen_por_ano(datos_producto, anio):
    df = datos_producto[datos_producto["Año"] == anio]
    ingreso = df["Ingreso_total"].sum()
    costo = df["Costo_total"].sum()
    return (ingreso - costo) / ingreso * 100 if ingreso > 0 else 0

def calcular_unidades_por_ano(datos_producto, anio):
    df = datos_producto[datos_producto["Año"] == anio]
    return df["Unidades_vendidas"].sum()

# Función principal para ejecutar la aplicación
def ejecutar_app():
    # Configuración de la página para quitar la cabecera de Streamlit
    st.set_page_config(page_title="Análisis de Ventas", layout="centered", initial_sidebar_state="collapsed")

    # Usar un fondo con gradiente animado
    html_string = """
    <style>
    /* Eliminar la cabecera de Streamlit */
    header {visibility: hidden;}

    /* Fondo animado con gradiente */
    body {
        background: linear-gradient(45deg, #34D399, #2F855A);
        font-family: 'Lato', sans-serif;
        animation: fadein 3s ease-in-out;
        margin: 0;
        padding: 0;
    }

    @keyframes fadein {
        0% { opacity: 0; }
        100% { opacity: 1; }
    }

    /* Estilos de las cabeceras */
    h1, h2, h3, h4, h5 {
        font-family: 'Lato', sans-serif;
        color: #ffffff;
        text-align: center;
        margin: 20px 0;
    }

    /* Estilo para los botones */
    .css-1d391kg {
        background-color: #38B2AC;
        color: white;
        border-radius: 12px;
        padding: 12px 24px;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
        font-weight: bold;
        transition: all 0.3s ease;
    }

    .css-1d391kg:hover {
        background-color: #319795;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.3);
        transform: translateY(-2px);
    }

    /* Estilo para las tablas */
    .stTable {
        border-radius: 10px;
        overflow: hidden;
    }

    .stTable th, .stTable td {
        padding: 12px 16px;
        text-align: left;
        border-bottom: 1px solid #ddd;
    }

    .stTable th {
        background-color: #2F855A;
        color: #ffffff;
    }

    .stTable td {
        background-color: #F7FAFC;
    }

    /* Estilo para los contenedores */
    .container {
        background-color: rgba(255, 255, 255, 0.9);
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.15);
        margin-bottom: 20px;
    }

    </style>
    """
    # Insertar el HTML y CSS personalizado
    st.markdown(html_string, unsafe_allow_html=True)

    # Mostrar los detalles del usuario
    mostrar_detalles_usuario()

    # Cargar los datos
    datos = cargar_datos_csv()

    if datos is not None:
        # Selección de sucursal
        sucursales_disponibles = ["Todas"] + datos["Sucursal"].unique().tolist()
        sucursal_seleccionada = st.selectbox("Selecciona la Sucursal para análisis", sucursales_disponibles)

        # Mostrar métricas y gráficos
        mostrar_metricas_y_graficos(datos, sucursal_seleccionada)
    else:
        st.warning("Por favor, sube un archivo CSV para continuar.")

if __name__ == "__main__":
    ejecutar_app()
