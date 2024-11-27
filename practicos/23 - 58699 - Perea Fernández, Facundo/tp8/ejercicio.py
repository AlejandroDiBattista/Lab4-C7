import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## ATENCION: Debe colocar la direccion en la que ha sido publicada la aplicacion en la siguiente linea\
# url = 'https://tp8-58699-facundopff.streamlit.app/'

def mostrar_informacion_alumno():
    with st.container(border=True):
        st.markdown('**Legajo:** 58.699')
        st.markdown('**Nombre:** Facundo Perea Fernandez')
        st.markdown('**Comisión:** C7')

st.set_page_config(page_title="Análisis de Ventas")


if "datos_cargados" not in st.session_state:
    st.session_state["datos_cargados"] = None

uploaded_file = st.sidebar.file_uploader("Subir archivo CSV", type="csv")

if uploaded_file is not None:
    datos = pd.read_csv(uploaded_file)
    st.session_state["datos_cargados"] = datos

def graficar(ventas_mensuales, producto_nombre):
    ventas_mensuales['Fecha'] = pd.to_datetime(
        ventas_mensuales['Año'].astype(str) + '-' + ventas_mensuales['Mes'].astype(str) + '-01', 
        format='%Y-%m-%d'
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        ventas_mensuales["Fecha"], 
        ventas_mensuales["Unidades_vendidas"], 
        label="Unidades Vendidas", 
        linewidth=1, 
        color="#39FF14"
    )

    #Línea de tendencia
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
            color="white",
            linewidth=2, 
            label="Tendencia",
        )

    ax.set_title(f"Evolución de Ventas - {producto_nombre}", fontsize=18, fontweight='bold', color="darkslategray")
    ax.set_xlabel("Fecha", fontsize=14, fontweight='bold', color="dimgray")
    ax.set_ylabel("Unidades Vendidas", fontsize=14, fontweight='bold', color="dimgray")
    ax.set_facecolor("#000000")  #Fondo del área de trazado
    ax.grid(True, linestyle="--", alpha=0.6, color="gray")
    ax.legend(fontsize=12, loc="upper left", frameon=True, facecolor="white", edgecolor="gray")

    return fig


if st.session_state["datos_cargados"] is not None:
    datos = st.session_state["datos_cargados"]
    sucursal_seleccionada = st.sidebar.selectbox(
        "Seleccionar Sucursal", 
        options=["Todas", "Sucursal Norte", "Sucursal Centro", "Sucursal Sur"]
    )
    
    if sucursal_seleccionada != "Todas":
        datos = datos[datos["Sucursal"] == sucursal_seleccionada]
        st.title(f"Análisis de Ventas - {sucursal_seleccionada}")
    else:
        st.title("Análisis de Ventas de Todas las sucursales")

    productos = datos['Producto'].unique()

    for producto in productos:
        with st.container(border=True):
            st.subheader(f"{producto}")
            datos_producto = datos[datos['Producto'] == producto]
            
            #Cálculo de PROMEDIOS

            #Precio promedio
            datos_producto['PrecioPromedio'] = datos_producto['Ingreso_total'] / datos_producto['Unidades_vendidas']
            precio_promedio = datos_producto['PrecioPromedio'].mean()
            #Ganancias promedio y el margen
            datos_producto['Ganancia'] = datos_producto['Ingreso_total'] - datos_producto['Costo_total']
            datos_producto['Margen'] = (datos_producto['Ganancia'] / datos_producto['Ingreso_total']) * 100
            margen_promedio = datos_producto['Margen'].mean()
            #Unidades vendidas
            unidades_promedio = datos_producto['Unidades_vendidas'].mean()
            unidades_vendidas = datos_producto['Unidades_vendidas'].sum()

            #Cálculo de las VARIACIONES

            #Variación anual del precio promedio
            precio_promedio_anual = datos_producto.groupby('Año')['PrecioPromedio'].mean()
            variacion_precio_promedio_anual = precio_promedio_anual.pct_change().mean() * 100
            #Variación anual del margen promedio
            margen_promedio_anual = datos_producto.groupby('Año')['Margen'].mean()
            variacion_margen_promedio_anual = margen_promedio_anual.pct_change().mean() * 100
            #Variación anual de las unidades vendidas
            unidades_por_ano = datos_producto.groupby('Año')['Unidades_vendidas'].sum()
            variacion_anual_unidades = unidades_por_ano.pct_change().mean() * 100
            
            col1, col2 = st.columns([1, 3]) #25%/75%
            
            with col1:
                st.metric(label="Precio Promedio", value=f"${precio_promedio:,.0f}", delta=f"{variacion_precio_promedio_anual:.2f}%")
                st.metric(label="Margen Promedio", value=f"{margen_promedio:.0f}%", delta=f"{variacion_margen_promedio_anual:.2f}%")
                st.metric(label="Unidades Vendidas", value=f"{unidades_vendidas:,.0f}", delta=f"{variacion_anual_unidades:.2f}%")

            with col2:
                fig = graficar(datos_producto, producto)
                col2.pyplot(fig)

else:
    mostrar_informacion_alumno()
    st.warning("Por favor, suba un archivo CSV en la barra lateral para comenzar.")
