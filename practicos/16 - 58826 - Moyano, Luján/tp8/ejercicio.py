import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Configuración inicial y tema rosado oscuro
st.set_page_config(page_title="Panel de Ventas", layout="wide")

# Estilos CSS personalizados
st.markdown("""
    <style>
        body {
            background-color: #9c2c50;
            font-family: "Arial", sans-serif;
        }
        .stButton>button {
            background-color: #f6b1d3;
            color: white;
            border-radius: 10px;
        }
        .stMetric {
            background-color: #f9e1e8;
            border-radius: 8px;
            padding: 10px;
            color: #5a173d;
        }
        .stSelectbox, .stMultiselect {
            background-color: #f6b1d3;
            color: white;
            border-radius: 10px;
            padding: 10px;
        }
        .stMarkdown h1, h2, h3 {
            color: #ffccd8;
        }
        .stTextInput input {
            background-color: #f6b1d3;
            color: white;
        }
        .stSlider {
            background-color: #f6b1d3;
        }
    </style>
""", unsafe_allow_html=True)

# Función para mostrar información del alumno
def mostrar_informacion_alumno():
    with st.container():
        st.markdown('**Legajo:** 58826')
        st.markdown('**Nombre:** Moyano Lujan')
        st.markdown('**Comisión:** C7')

# Mostrar la información del alumno al principio
mostrar_informacion_alumno()

# Encabezado de la página
st.title("🌸 Panel de Ventas 🌸")

st.sidebar.header("⚙️ Opciones de Archivos")

# Cargar archivo CSV
archivos = st.sidebar.file_uploader("📂 Subir archivo CSV", type="csv")

# Verificar si se subió un archivo
if archivos is not None:
    try:
        datos = pd.read_csv(archivos)

        # Renombrar columnas
        datos = datos.rename(columns={
            "Producto": "Prod",
            "Unidades_vendidas": "uniVendidas",
            "Ingreso_total": "Ingresos",
            "Costo_total": "Costos"
        })

        # Comprobar columnas necesarias
        columnas_necesarias = ["Sucursal", "Prod", "Año", "Mes", "uniVendidas", "Ingresos", "Costos"]
        if all(col in datos.columns for col in columnas_necesarias):
            datos["precioP"] = datos["Ingresos"] / datos["uniVendidas"]
            datos["margenP"] = (datos["Ingresos"] - datos["Costos"]) / datos["Ingresos"]

            # Resumen por producto
            resumen = datos.groupby("Prod").agg({
                "precioP": "mean",
                "margenP": "mean",
                "uniVendidas": "sum"
            }).reset_index()

            # Filtro por sucursal
            sucursales = ["Todas"] + datos["Sucursal"].unique().tolist()
            Sucursal = st.sidebar.selectbox("🏢 Seleccionar Sucursal", sucursales)

            if Sucursal != "Todas":
                datos = datos[datos["Sucursal"] == Sucursal]

            st.header(f"📊 Datos de {'Todas las Sucursales' if Sucursal == 'Todas' else Sucursal}")

            for _, fila in resumen.iterrows():
                prod = fila["Prod"]
                precio = fila["precioP"]
                margen = fila["margenP"]
                unidades = fila["uniVendidas"]

                col1, col2 = st.columns([1, 3])
                with col1:
                    st.markdown(f"### 🎯 Producto: {prod}")
                    st.metric("💵 Precio Promedio", f"${precio:,.2f}")
                    st.metric("📈 Margen Promedio", f"{margen * 100:.2f}%")
                    st.metric("📦 Unidades Vendidas", f"{int(unidades):,}")

                with col2:
                    datos_prod = datos[datos["Prod"] == prod]
                    datos_prod["Fecha"] = pd.to_datetime(datos_prod["Año"].astype(str) + "-" + datos_prod["Mes"].astype(str) + "-01")
                    ventas_mensuales = datos_prod.groupby("Fecha")["uniVendidas"].sum().reset_index()

                    # Gráfico con Matplotlib
                    plt.figure(figsize=(8, 5))
                    plt.plot(ventas_mensuales["Fecha"], ventas_mensuales["uniVendidas"], marker="o", color="#b72e61", label=prod)

                    z = np.polyfit(np.arange(len(ventas_mensuales)), ventas_mensuales["uniVendidas"], 1)
                    p = np.poly1d(z)
                    plt.plot(ventas_mensuales["Fecha"], p(np.arange(len(ventas_mensuales))), "r--", label="Tendencia")

                    plt.title(f"Evolución de Ventas - {prod}", fontsize=14, color="#801b42")
                    plt.xlabel("Mes", fontsize=12)
                    plt.ylabel("Unidades Vendidas", fontsize=12)
                    plt.grid(alpha=0.4)
                    plt.legend()
                    st.pyplot(plt)
        else:
            st.error("⚠️ El archivo no contiene las columnas requeridas.")

    except Exception as e:
        st.error(f"⚠️ Hubo un error al procesar el archivo: {e}")

## ATENCION: Debe colocar la direccion en la que ha sido publicada la aplicacion en la siguiente linea\
url = 'https://tp8-58826.streamlit.app/'