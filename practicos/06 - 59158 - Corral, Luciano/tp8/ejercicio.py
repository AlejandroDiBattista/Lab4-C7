import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import streamlit as st

# url = 'https://tp-8-59158-luciano-corral.streamlit.app'
# Función que muestra detalles personales
def mostrar_detalles_usuario():
    st.markdown("#### Datos del alumno")
    st.markdown("**Legajo:** 59158")
    st.markdown("**Nombre Completo:** Luciano Corral")
    st.markdown("**Comision:** C7")

# Función para cargar datos CSV desde el archivo
def cargar_datos_csv():
    archivo = st.sidebar.file_uploader("Selecciona un archivo CSV", type="csv")
    if archivo:
        return pd.read_csv(archivo)
    return None

# Función para procesar los datos y generar gráficos
def generar_metricas_y_visualizaciones(datos, sucursal_seleccionada):
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

        # Cálculo de las métricas
        unidades_totales = datos_producto["Unidades_vendidas"].sum()
        ingresos_totales = datos_producto["Ingreso_total"].sum()
        costos_totales = datos_producto["Costo_total"].sum()

        precio_unitario_promedio = ingresos_totales / unidades_totales
        margen_unitario_promedio = (ingresos_totales - costos_totales) / ingresos_totales * 100

        # Comparar precio promedio con el global
        precio_promedio_global = datos["Ingreso_total"].sum() / datos["Unidades_vendidas"].sum()
        delta_precio = precio_unitario_promedio - precio_promedio_global
        
        precio_promedio_2024 = calcular_precio_por_ano(datos_producto, 2024)
        precio_promedio_2023 = calcular_precio_por_ano(datos_producto, 2023)
        
        margen_2024 = calcular_margen_por_ano(datos_producto, 2024)
        margen_2023 = calcular_margen_por_ano(datos_producto, 2023)
        
        unidades_2024 = calcular_unidades_por_ano(datos_producto, 2024)
        unidades_2023 = calcular_unidades_por_ano(datos_producto, 2023)

        # Mostrar las métricas calculadas
        st.header(f"Producto: {producto}")
        st.metric("Precio Promedio", f"${precio_unitario_promedio:,.2f}", delta=f"{((precio_promedio_2024 / precio_promedio_2023) - 1) * 100:.2f}%")
        st.metric("Margen Promedio", f"{margen_unitario_promedio:.2f}%", delta=f"{((margen_2024 / margen_2023) - 1) * 100:.2f}%")
        st.metric("Unidades Vendidas", f"{unidades_totales:,}", delta=f"{((unidades_2024 / unidades_2023) - 1) * 100:.2f}%")

        # Convertir a formato de fecha
        datos_producto['Fecha'] = pd.to_datetime(datos_producto['Año'].astype(str) + '-' + datos_producto['Mes'].astype(str) + '-01')
        datos_producto.sort_values('Fecha', inplace=True)

        # Generación del gráfico de ventas
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(datos_producto["Fecha"], datos_producto["Unidades_vendidas"], label=f"Ventas de {producto}", color="purple")

        # Agregar línea de tendencia
        x = np.arange(len(datos_producto))
        y = datos_producto["Unidades_vendidas"].values
        pendiente, intercepto, _, _, _ = stats.linregress(x, y)
        tendencia_linea = pendiente * x + intercepto
        ax.plot(datos_producto["Fecha"], tendencia_linea, label="Tendencia", color="orange", linestyle="--")

        ax.set_title(f"Evolución de Ventas Mensuales para {producto}", fontsize=14)
        ax.set_xlabel("Fecha", fontsize=12)
        ax.set_ylabel("Unidades Vendidas", fontsize=12)
        ax.legend(loc="upper left")
        plt.xticks(rotation=45)
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
    st.sidebar.title("Subir Datos")
    mostrar_detalles_usuario()
    datos = cargar_datos_csv()
    if datos is not None:
        sucursales_disponibles = ["Todas"] + datos["Sucursal"].unique().tolist()
        sucursal_seleccionada = st.sidebar.selectbox("Seleccionar Sucursal", sucursales_disponibles)
        st.header(f"Análisis de {'Todas las Sucursales' if sucursal_seleccionada == 'Todas' else sucursal_seleccionada}")
        generar_metricas_y_visualizaciones(datos, sucursal_seleccionada)
    else:
        st.write("Por favor, carga un archivo CSV desde el panel lateral.")

if __name__ == "__main__":
    ejecutar_app()
