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

        producto_datos = datos[datos["Producto"] == producto["Producto"]]
        if "Año" in producto_datos.columns and "Mes" in producto_datos.columns:
            producto_datos["Fecha"] = pd.to_datetime(
                producto_datos["Año"].astype(str) + "-" + producto_datos["Mes"].astype(str),
                errors="coerce"
            )
        else:
            st.error("El archivo debe contener las columnas 'Año' y 'Mes' para calcular la columna 'Fecha'.")
            continue
        
        if "Fecha" in producto_datos.columns and not producto_datos["Fecha"].isnull().all():
            ventas_mensuales = producto_datos.groupby(producto_datos["Fecha"].dt.to_period("M")).agg(
                {"Unidades_vendidas": "sum"}
            ).reset_index()
            ventas_mensuales["Fecha"] = ventas_mensuales["Fecha"].dt.to_timestamp()
        else:
            st.error(f"No se pudo generar la columna 'Fecha' para el producto {producto['Producto']}.")
            continue
        ventas_anuales = producto_datos.groupby("Año").agg(
            Precio_Promedio=("Ingreso_total", lambda x: x.sum() / producto_datos.loc[x.index, "Unidades_vendidas"].sum()),
            Margen_Promedio=("Ingreso_total", lambda x: (x.sum() - producto_datos.loc[x.index, "Costo_total"].sum()) / x.sum()),
            Unidades_Vendidas=("Unidades_vendidas", "sum")
        ).reset_index()

        if len(ventas_anuales) > 1:
            variacion_precio = (ventas_anuales.iloc[-1]["Precio_Promedio"] - ventas_anuales.iloc[-2]["Precio_Promedio"]) / ventas_anuales.iloc[-2]["Precio_Promedio"] * 100
            variacion_margen = (ventas_anuales.iloc[-1]["Margen_Promedio"] - ventas_anuales.iloc[-2]["Margen_Promedio"]) / ventas_anuales.iloc[-2]["Margen_Promedio"] * 100
            variacion_unidades = (ventas_anuales.iloc[-1]["Unidades_Vendidas"] - ventas_anuales.iloc[-2]["Unidades_Vendidas"]) / ventas_anuales.iloc[-2]["Unidades_Vendidas"] * 100
        else:
            variacion_precio = None
            variacion_margen = None
            variacion_unidades = None

        col1.subheader("")
        col1.subheader(f"{producto['Producto']}")

        # Precio promedio
        col1.metric(
            label="Precio promedio:",
            value=f"${producto['Precio_Promedio']:.2f}", 
            delta=f"{variacion_precio:.2f}%" if variacion_precio is not None else "N/A"
        )

        # Margen promedio
        col1.metric(
            label="Margen Promedio:",
            value=f"{producto['Margen_Promedio']:.2%}", 
            delta=f"{variacion_margen:.2f}%" if variacion_margen is not None else "N/A"
        )

        # Unidades vendidas
        col1.metric(
            label="Unidades Vendidas:",
            value=f"{producto['Unidades_Vendidas']:,}", 
            delta=f"{variacion_unidades:.2f}%" if variacion_unidades is not None else "N/A"
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
            
            # Este es el color de fondo
            ax.plot(
                ventas_mensuales["Fecha"], 
                p(ventas_mensuales["Fecha"].apply(lambda x: x.toordinal())), 
                linestyle="--", 
                color="#FFD700",  # Color dorado brillante para la línea de tendencia
                linewidth=2, 
                label="Tendencia",
            )

        ax.set_title(f"Evolución de Ventas - {producto['Producto']}", fontsize=18, fontweight='bold', color="darkslategray")
        ax.set_xlabel("Fecha", fontsize=14, fontweight='bold', color="dimgray")
        ax.set_ylabel("Unidades Vendidas", fontsize=14, fontweight='bold', color="dimgray")
        ax.set_facecolor("#000000")  # Fondo claro para el área de trazado
        ax.grid(True, linestyle="--", alpha=0.6, color="gray")
        ax.legend(fontsize=12, loc="upper left", frameon=True, facecolor="white", edgecolor="gray")

        col2.pyplot(fig)


else:
    st.warning("Por favor, suba un archivo CSV para comenzar.")




