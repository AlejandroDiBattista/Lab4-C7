import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## ATENCION: Debe colocar la direccion en la que ha sido publicada la aplicacion en la siguiente linea\
# url = 'https://tp8-555555.streamlit.app/'

def mostrar_informacion_alumno():
    with st.container(border=True):
        st.markdown('**Legajo:** 58.699')
        st.markdown('**Nombre:** Facundo Perea Fernandez')
        st.markdown('**Comisión:** C7')

st.set_page_config(page_title="Análisis de Ventas", layout="wide")
mostrar_informacion_alumno()
st.title("Análisis de Ventas")

if "datos_cargados" not in st.session_state:
    st.session_state["datos_cargados"] = None

uploaded_file = st.sidebar.file_uploader("Subir archivo CSV", type="csv")

if uploaded_file is not None:
    datos = pd.read_csv(uploaded_file)
    st.session_state["datos_cargados"] = datos

if st.session_state["datos_cargados"] is not None:
    datos = st.session_state["datos_cargados"]
    sucursal_seleccionada = st.sidebar.selectbox(
        "Seleccionar Sucursal", 
        options=["Todas", "Sucursal Norte", "Sucursal Centro", "Sucursal Sur"]
    )
    
    if sucursal_seleccionada != "Todas":
        datos = datos[datos["Sucursal"] == sucursal_seleccionada]

    resumen = datos.groupby("Producto").agg(
        Precio_Promedio=("Ingreso_total", lambda x: x.sum() / datos.loc[x.index, "Unidades_vendidas"].sum()),
        Margen_Promedio=("Ingreso_total", lambda x: (x.sum() - datos.loc[x.index, "Costo_total"].sum()) / x.sum()),
        Unidades_Vendidas=("Unidades_vendidas", "sum")
    ).reset_index()

    for _, producto in resumen.iterrows():
        col1, col2 = st.columns([1, 4])

        col1.subheader(f"{producto['Producto']}")
        col1.write("Precio promedio:")
        col1.subheader(f"${producto['Precio_Promedio']:.2f}")
        col1.write("Margen Promedio:")
        col1.subheader(f"{producto['Margen_Promedio']:.2%}")
        col1.write("Unidades Vendidas:")
        col1.subheader(f"{producto['Unidades_Vendidas']:,}")

        producto_datos = datos[datos["Producto"] == producto["Producto"]]
        producto_datos["Fecha"] = pd.to_datetime(
            producto_datos["Año"].astype(str) + "-" + producto_datos["Mes"].astype(str)
        )
        ventas_mensuales = producto_datos.groupby(producto_datos["Fecha"].dt.to_period("M")).agg(
            {"Unidades_vendidas": "sum"}
        ).reset_index()
        ventas_mensuales["Fecha"] = ventas_mensuales["Fecha"].dt.to_timestamp()

        fig, ax = plt.subplots(figsize=(14, 8))
        ax.plot(
            ventas_mensuales["Fecha"], 
            ventas_mensuales["Unidades_vendidas"], 
            label="Unidades Vendidas", 
            linewidth=3, 
            color="#39FF14"
        )

        if len(ventas_mensuales) > 1:
            z = np.polyfit(
                ventas_mensuales["Fecha"].apply(lambda x: x.toordinal()), 
                ventas_mensuales["Unidades_vendidas"], 
                1
            )
            p = np.poly1d(z)
            ax.plot(
                ventas_mensuales["Fecha"], 
                p(ventas_mensuales["Fecha"].apply(lambda x: x.toordinal())), 
                linestyle="--", 
                color="#DC143C", 
                linewidth=2, 
                label="Tendencia"
            )

        ax.set_title(f"Evolución de Ventas - {producto['Producto']}", fontsize=18, fontweight='bold', color="darkslategray")
        ax.set_xlabel("Fecha", fontsize=14, fontweight='bold', color="dimgray")
        ax.set_ylabel("Unidades Vendidas", fontsize=14, fontweight='bold', color="dimgray")
        ax.set_facecolor("#f7f7f7")
        ax.grid(True, linestyle="--", alpha=0.6, color="gray")
        ax.legend(fontsize=12, loc="upper left", frameon=True, facecolor="white", edgecolor="gray")

        col2.pyplot(fig)
else:
    st.warning("Por favor, suba un archivo CSV para comenzar.")




