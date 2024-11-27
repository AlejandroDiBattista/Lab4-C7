import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

##ATENCION: Debe colocar la direccion en la que ha sido publicada la aplicacion en la siguiente linea \
# url = https://tp8-59134-c7.streamlit.app/

st.set_page_config(page_title="Análisis de Ventas", layout="wide")

def mostrar_informacion_alumno():
    with st.container():
        st.markdown("**Legajo:** 59134")
        st.markdown("**Nombre:** Tomas Molina")
        st.markdown("**Comisión:** C7")

mostrar_informacion_alumno()

st.title("Análisis de Ventas")
st.sidebar.header("Cargar archivo de datos")
uploaded_file = st.sidebar.file_uploader("Seleccione el archivo CSV", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)

        columnas_esperadas = ["Sucursal", "Producto", "Año", "Mes", "Unidades_vendidas", "Ingreso_total", "Costo_total"]
        if not all(col in df.columns for col in columnas_esperadas):
            st.error("El archivo CSV no tiene las columnas esperadas.")
        else:
            df["Precio_promedio"] = df["Ingreso_total"] / df["Unidades_vendidas"]
            df["Margen_promedio"] = (df["Ingreso_total"] - df["Costo_total"]) / df["Ingreso_total"]
            df["Fecha"] = pd.to_datetime(df["Año"].astype(str) + "-" + df["Mes"].astype(str) + "-01", format="%Y-%m-%d")

            sucursales = ["Todas"] + list(df["Sucursal"].unique())
            sucursal_seleccionada = st.sidebar.selectbox("Seleccionar Sucursal", sucursales)

            if sucursal_seleccionada != "Todas":
                df = df[df["Sucursal"] == sucursal_seleccionada]

            resumen = df.groupby("Producto").agg({
                "Precio_promedio": "mean",
                "Margen_promedio": "mean",
                "Unidades_vendidas": "sum",
            }).reset_index()

            st.header(f"Datos de {'Todas las Sucursales' if sucursal_seleccionada == 'Todas' else sucursal_seleccionada}")
            for _, row in resumen.iterrows():
                with st.container():
                    st.markdown("---")
                    with st.container():
                        col1, col2 = st.columns([1, 3])

                        with col1:
                            st.markdown(f"### {row['Producto']}")
                            st.markdown(f"**Precio Promedio:** ${row['Precio_promedio']:.2f}")
                            st.markdown(f"**Margen Promedio:** {row['Margen_promedio'] * 100:.2f}%")
                            st.markdown(f"**Unidades Vendidas:** {int(row['Unidades_vendidas']):,}")

                        with col2:
                            datos_producto = df[df["Producto"] == row["Producto"]].groupby("Fecha").agg({
                                "Unidades_vendidas": "sum"
                            }).reset_index()

                            fig, ax = plt.subplots(figsize=(12, 5))
                            ax.plot(datos_producto["Fecha"], datos_producto["Unidades_vendidas"], marker="o", label=row["Producto"])
                            z = np.polyfit(np.arange(len(datos_producto)), datos_producto["Unidades_vendidas"], 1)
                            p = np.poly1d(z)
                            ax.plot(datos_producto["Fecha"], p(np.arange(len(datos_producto))), linestyle="--", color="red", label="Tendencia")
                            
                            ax.set_xlim(datos_producto["Fecha"].iloc[0], datos_producto["Fecha"].iloc[-1])
                            ax.set_ylim(min(datos_producto["Unidades_vendidas"]) * 0.9, max(datos_producto["Unidades_vendidas"]) * 1.1)
                            
                            ax.set_title("Evolución de Ventas Mensual", fontsize=14)
                            ax.set_xlabel("Fecha", fontsize=12)
                            ax.set_ylabel("Unidades Vendidas", fontsize=12)
                            ax.legend()
                            plt.xticks(rotation=45)
                            plt.tight_layout()
                            st.pyplot(fig)

            st.header("Datos Completos")
            st.dataframe(df)

    except Exception as e:
        st.error(f"Error al procesar el archivo: {str(e)}")
else:
    st.info("Esperando la carga del archivo CSV para realizar el análisis.")

