import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## ATENCION: Debe colocar la direccion en la que ha sido publicada la aplicacion en la siguiente linea\
# url = 'https://parcial59047torresfrancisco.streamlit.app/'

st.set_page_config(
    page_title="Examen Parcial",
    layout="wide",
    
)
def mostrarAlumno():
    st.markdown('**Legajo:** 59047')
    st.markdown('**Nombre:** Torres Francisco Gabriel')
    st.markdown('**Comisión:** 7')
def Variacion(df, col, agrupar_por='Año'):
    return df.groupby(agrupar_por)[col].mean().pct_change().mean() * 100
#Grafico
def VentasGraph(datos_producto, producto):
    ventas_por_producto = datos_producto.groupby(['Año', 'Mes'])['Unidades_vendidas'].sum().reset_index()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(ventas_por_producto.index, ventas_por_producto['Unidades_vendidas'], label=producto, color='Red')
    
    z = np.polyfit(ventas_por_producto.index, ventas_por_producto['Unidades_vendidas'], 1)
    p = np.poly1d(z)
    ax.plot(ventas_por_producto.index, p(ventas_por_producto.index), linestyle='--', color='Green', label='Tendencia')
    
    
    ax.set_title('Evolución de Ventas Mensual')
    ax.set_xlabel('Año-Mes')
    ax.set_xticks(ventas_por_producto.index)
    ax.set_xticklabels([f"{row.Año}-{row.Mes}" if i % 6 == 0 else "" for i, row in ventas_por_producto.iterrows()])
    ax.set_ylabel('Unidades Vendidas')
    ax.set_ylim(0, None)
    ax.legend(title='Producto')
    ax.grid(True)
    
    return fig

st.sidebar.header("Cargar Datos")
archivo_cargado = st.sidebar.file_uploader("Subir archivo CSV", type=["csv"])
#Procesar
if archivo_cargado is not None:
    datos = pd.read_csv(archivo_cargado)
    
    sucursales = ["Todas"] + datos['Sucursal'].unique().tolist()
    SucursalElegida = st.sidebar.selectbox("Seleccionar Sucursal", sucursales)
    
    
    if SucursalElegida != "Todas":
        datos = datos[datos['Sucursal'] == SucursalElegida]
        st.title(f"Datos de {SucursalElegida}")
    else:
        st.title("Datos de Todas las Sucursales")
    
    productos = datos['Producto'].unique()

    for producto in productos:
        st.subheader(f"{producto}")
        datos_producto = datos[datos['Producto'] == producto]
        
        datos_producto['Precio_promedio'] = datos_producto['Ingreso_total'] / datos_producto['Unidades_vendidas']
        precio_promedio = datos_producto['Precio_promedio'].mean()
        variacion_precio = Variacion(datos_producto, 'Precio_promedio')

        datos_producto['Ganancia'] = datos_producto['Ingreso_total'] - datos_producto['Costo_total']
        datos_producto['Margen'] = (datos_producto['Ganancia'] / datos_producto['Ingreso_total']) * 100
        margen_promedio = datos_producto['Margen'].mean()
        variacion_margen = Variacion(datos_producto, 'Margen')
        
        
        unidades_vendidas = datos_producto['Unidades_vendidas'].sum()
        variacion_unidades = Variacion(datos_producto, 'Unidades_vendidas')
        col1, col2 = st.columns([0.25, 0.75])
        
        with col1:
            st.metric(label="Precio Promedio", value=f"${precio_promedio:,.0f}".replace(",", "."), delta=f"{variacion_precio:.2f}%")
            st.metric(label="Margen Promedio", value=f"{margen_promedio:.0f}%".replace(",", "."), delta=f"{variacion_margen:.2f}%")
            st.metric(label="Unidades Vendidas", value=f"{unidades_vendidas:,.0f}".replace(",", "."), delta=f"{variacion_unidades:.2f}%")
        with col2:
            fig = VentasGraph(datos_producto, producto)
            st.pyplot(fig)

else:
    st.subheader("Por favor, sube un archivo CSV desde la barra lateral.")
    st.info ("Trabajo Practico NO.8")
    mostrarAlumno()