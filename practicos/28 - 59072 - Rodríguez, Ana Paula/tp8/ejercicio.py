import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

url = 'https://tp8-59072paula.streamlit.app/'

# Configuración inicial y tema fucsia oscuro y morado
st.set_page_config(page_title="📊 Análisis de Datos de Productos", layout="wide")

# Estilos CSS personalizados con tonos de fucsia
st.markdown("""
    <style>
        body {
            background-color: #9C27B0; /* Fondo fucsia oscuro */
            font-family: "Helvetica", sans-serif; /* Fuente personalizada */
        }
        .stButton>button {
            background-color: #D81B60; /* Botones fucsia claro */
            color: white;
            border-radius: 12px;
            box-shadow: 0 5px 10px rgba(0, 0, 0, 0.15);
        }
        .stMetric {
            background-color: #F48FB1; /* Fondo fucsia claro para métricas */
            border-radius: 10px;
            padding: 15px;
            color: #880E4F; /* Texto fucsia oscuro */
        }
        .stSelectbox, .stMultiselect {
            background-color: #D81B60; /* Fondo fucsia claro para opciones */
            color: white;
            border-radius: 12px;
            padding: 10px;
        }
        .stMarkdown h1, h2, h3 {
            color: #F3E5F5; /* Títulos en fucsia suave */
        }
        .stTextInput input {
            background-color: #D81B60; /* Fondo fucsia claro para inputs */
            color: white;
        }
        .stFileUploader {
            background-color: #D81B60; /* Fondo fucsia claro para el panel de carga de archivos */
            padding: 10px;
            border-radius: 12px;
            box-shadow: 0 5px 10px rgba(0, 0, 0, 0.1);
        }
        .stSidebar {
            background-color: #880E4F; /* Sidebar fucsia oscuro */
        }
    </style>
""", unsafe_allow_html=True)

# Función para mostrar información del alumno
def mostrar_informacion_alumno():
    with st.container():
        st.markdown('**Legajo1:** 59072')
        st.markdown('**Nombre Completo:** Rodriguez Ana Paula')
        st.markdown('**Comisión:** C7')

# Mostrar la información del alumno al principio
mostrar_informacion_alumno()

# Encabezado de la página
st.title("💜 Ventas Productos 💜")

st.sidebar.header(" Cargar todos los Datos")

# Cargar archivo CSV
archivo = st.sidebar.file_uploader(" Subir archivo CSV", type="csv")

# Verificar si se subió un archivo
if archivo is not None:
    try:
        # Leer el archivo CSV cargado
        data = pd.read_csv(archivo)

        # Verificar las primeras filas del archivo cargado
        st.write("Primera fila del archivo cargado:")
        st.write(data.head())

        # Renombrar las columnas para que coincidan con las esperadas
        data = data.rename(columns={
            "Producto": "Producto",
            "Unidades_vendidas": "CantidadVendida",
            "Ingreso_total": "ValorTotal",
            "Costo_total": "CostoTotal"
        })

        # Comprobar que las columnas necesarias están presentes
        columnas_requeridas = ["Sucursal", "Producto", "Año", "Mes", "CantidadVendida", "ValorTotal", "CostoTotal"]
        if all(col in data.columns for col in columnas_requeridas):

            # Asegurarse de que las columnas numéricas estén en el formato correcto
            data["ValorTotal"] = pd.to_numeric(data["ValorTotal"], errors="coerce")
            data["CostoTotal"] = pd.to_numeric(data["CostoTotal"], errors="coerce")
            data["CantidadVendida"] = pd.to_numeric(data["CantidadVendida"], errors="coerce")

            # Eliminar filas con datos faltantes (si es necesario)
            data.dropna(subset=["ValorTotal", "CostoTotal", "CantidadVendida"], inplace=True)

            # Realizar cálculos adicionales
            data["PrecioUnitario"] = data["ValorTotal"] / data["CantidadVendida"]
            data["MargenBruto"] = (data["ValorTotal"] - data["CostoTotal"]) / data["ValorTotal"]

            # Resumen por producto
            resumen_producto = data.groupby("Producto").agg({
                "PrecioUnitario": "mean",
                "MargenBruto": "mean",
                "CantidadVendida": "sum"
            }).reset_index()

            # Filtro por sucursal
            sucursales = ["Todas"] + data["Sucursal"].unique().tolist()
            sucursal_seleccionada = st.sidebar.selectbox("🏢 Seleccionar Sucursal", sucursales)

            if sucursal_seleccionada != "Todas":
                data = data[data["Sucursal"] == sucursal_seleccionada]

            # Título de análisis
            st.header(f"📊 Datos de {'Todas las Sucursales' if sucursal_seleccionada == 'Todas' else sucursal_seleccionada}")

            # Mostrar métricas y gráficos por producto
            for _, fila in resumen_producto.iterrows():
                producto = fila["Producto"]
                precio = fila["PrecioUnitario"]
                margen = fila["MargenBruto"]
                cantidad = fila["CantidadVendida"]

                # Diseño en columnas con nombres más creativos
                columna_datos, columna_grafico = st.columns([1, 3])

                # Métricas
                with columna_datos:
                    st.markdown(f"### 💜 Producto: {producto}")
                    st.metric("💵 Precio Promedio", f"${precio:,.2f}")
                    st.metric("📈 Margen Promedio", f"{margen * 100:.2f}%")
                    st.metric("📦 Unidades Vendidas", f"{int(cantidad):,}")

                # Gráfico
                with columna_grafico:
                    data_producto = data[data["Producto"] == producto]
                    data_producto["Fecha"] = pd.to_datetime(data_producto["Año"].astype(str) + "-" + data_producto["Mes"].astype(str) + "-01")
                    ventas_mensuales = data_producto.groupby("Fecha")["CantidadVendida"].sum().reset_index()

                    plt.figure(figsize=(8, 5))
                    plt.plot(ventas_mensuales["Fecha"], ventas_mensuales["CantidadVendida"], marker="o", color="#4A148C", label=producto)

                    # Tendencia
                    z = np.polyfit(ventas_mensuales.index, ventas_mensuales["CantidadVendida"], 1)
                    p = np.poly1d(z)
                    plt.plot(ventas_mensuales["Fecha"], p(ventas_mensuales.index), "r--", label="Tendencia")

                    # Configuración del gráfico
                    plt.title(f"Evolución de Ventas - {producto}", fontsize=14, color="#4A148C")
                    plt.xlabel("Mes", fontsize=12)
                    plt.ylabel("Cantidad Vendida", fontsize=12)
                    plt.grid(alpha=0.3)
                    plt.legend()
                    st.pyplot(plt)

        else:
            st.error("⚠️ El archivo no contiene las columnas requeridas.")

    except Exception as e:
        st.error(f"⚠️ Hubo un error al procesar el archivo: {e}")
