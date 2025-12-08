import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# URL = 'https://tp8-59358.streamlit.app/'


#Lo uso para modificar mediante CSS los componentes de streamlit
st.markdown(
    """
    <style>
    /* Cambiar el color de fondo del sidebar */
    [data-testid="stSidebar"] {
        background-color: #021C1E;
    }
    
    .stApp {
        background-color: #004445;
    }
    
    .stSidebar, .stApp {
        color: white; 
    }

    #Estilos del botón Browse Files
    [data-testid="stFileUploadButton"]:hover {
        background-color: #2C7873;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)



def crear_grafico_ventas(datos_producto, producto):

    ventas_mensuales = datos_producto.groupby(['Año', 'Mes'])['Unidades_vendidas'].sum().reset_index()
    
    # Configuro el gráfico
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(len(ventas_mensuales)), ventas_mensuales['Unidades_vendidas'], label=f"Evolución - {producto}")
    
    # Línea de tendencia con regresión lineal
    x = np.arange(len(ventas_mensuales))
    y = ventas_mensuales['Unidades_vendidas']
    coeficientes = np.polyfit(x, y, 1)
    tendencia = np.poly1d(coeficientes)
    
    ax.plot(x, tendencia(x), linestyle='--', color='red', label='Tendencia')
    
    # Personalizo ejes y títulos
    ax.set_title(f"Evolución de Ventas: {producto}", fontsize=14)
    ax.set_xlabel('Año-Mes', fontsize=12)
    ax.set_ylabel('Unidades Vendidas', fontsize=12)
    
    # Etiquetas del eje X (solo muestro el año en enero para mayor claridad)
    etiquetas = [f"{row.Año}" if row.Mes == 1 else "" for row in ventas_mensuales.itertuples()]
    ax.set_xticks(range(len(ventas_mensuales)))
    ax.set_xticklabels(etiquetas, rotation=45)
    
    ax.legend(title="Datos")
    ax.grid(True, linestyle='--', alpha=0.7)
    return fig

# En caso de no cargarse archivos se muestra esto
def mostrar_informacion_alumno():
    with st.container():
        st.markdown('**Legajo:** 59358')
        st.markdown('**Nombre:** Gonzalo Alejandro Nieva Pastoriza')
        st.markdown('**Comisión:** C7')

# Aquí se carga el archivo CSV
st.sidebar.header("Cargar archivo de datos")
archivo_cargado = st.sidebar.file_uploader("Subir archivo CSV", type=["csv"])

if archivo_cargado is not None:
    datos = pd.read_csv(archivo_cargado)
    

    sucursales = ["Todas"] + datos['Sucursal'].unique().tolist()
    

    sucursal_seleccionada = st.sidebar.selectbox("Seleccionar Sucursal", sucursales)
    

    if sucursal_seleccionada != "Todas":
        datos = datos[datos['Sucursal'] == sucursal_seleccionada]
        st.title(f"Datos de {sucursal_seleccionada}")
    else:
        st.title("Datos de Todas las Sucursales")
    

    for producto in datos['Producto'].unique():
        with st.container():
            st.subheader(f"Análisis de Ventas: {producto}")
            datos_producto = datos[datos['Producto'] == producto]
            
            # Calculo el precio promedio
            datos_producto['Precio_promedio'] = datos_producto['Ingreso_total'] / datos_producto['Unidades_vendidas']
            precio_promedio = datos_producto['Precio_promedio'].mean()

            variacion_precio = datos_producto.groupby('Año')['Precio_promedio'].mean().pct_change().mean() * 100
            
            # Calculo margen promedio y su variación anual
            datos_producto['Ganancia'] = datos_producto['Ingreso_total'] - datos_producto['Costo_total']
            datos_producto['Margen'] = (datos_producto['Ganancia'] / datos_producto['Ingreso_total']) * 100
            margen_promedio = datos_producto['Margen'].mean()
            variacion_margen = datos_producto.groupby('Año')['Margen'].mean().pct_change().mean() * 100
            
            # Calculo total y promedio de unidades vendidas
            unidades_vendidas = datos_producto['Unidades_vendidas'].sum()
            variacion_unidades = datos_producto.groupby('Año')['Unidades_vendidas'].sum().pct_change().mean() * 100
            
            col1, col2 = st.columns([0.3, 0.7])
            
            with col1:
                st.metric("Precio Promedio", f"${precio_promedio:,.0f}".replace(",", "."), f"{variacion_precio:.2f}%".replace(".", ","))
                st.metric("Margen Promedio", f"{margen_promedio:.0f}%".replace(",", "."), f"{variacion_margen:.2f}%".replace(".", ","))
                st.metric("Unidades Vendidas", f"{unidades_vendidas:,.0f}".replace(",", "."), f"{variacion_unidades:.2f}%".replace(".", ","))
            with col2:
                fig = crear_grafico_ventas(datos_producto, producto)
                st.pyplot(fig)
else:
    st.info("Por favor, sube un archivo CSV desde la barra lateral para empezar")
    mostrar_informacion_alumno()