import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## ATENCION: Debe colocar la dirección en la que ha sido publicada la aplicación en la siguiente línea
# url = https://tp8-59251.streamlit.app/

# Función para crear el gráfico de evolución de ventas
def crear_grafico_ventas(datos_producto, producto):
    ventas_por_producto = datos_producto.groupby(['Año', 'Mes'])['Unidades_vendidas'].sum().reset_index()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(len(ventas_por_producto)), ventas_por_producto['Unidades_vendidas'], label=producto)
    
    # Línea de tendencia
    x = np.arange(len(ventas_por_producto))
    y = ventas_por_producto['Unidades_vendidas']
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    ax.plot(x, p(x), linestyle='--', color='red', label='Tendencia')
    
    ax.set_title(f'Evolución de Ventas Mensual - {producto}')
    ax.set_xlabel('Año-Mes')
    ax.set_xticks(range(len(ventas_por_producto)))
    
    # Configurar etiquetas del eje x
    etiquetas = [f"{row.Año}" if row.Mes == 1 else "" for row in ventas_por_producto.itertuples()]
    ax.set_xticklabels(etiquetas)
    
    ax.set_ylabel('Unidades Vendidas')
    ax.set_ylim(0, None)
    ax.legend(title='Producto')
    ax.grid(True)
    
    return fig

# Función para mostrar información del alumno
def mostrar_informacion_alumno():
    st.markdown("""
    **Legajo:** 59251  
    **Nombre:** Mamani Daniel Fernando  
    **Comisión:** C7  
    """)

# Configuración del sidebar para cargar archivos
st.sidebar.header("Cargar archivo de datos")
archivo_cargado = st.sidebar.file_uploader("Subir archivo CSV", type=["csv"])

if archivo_cargado is not None:
    datos = pd.read_csv(archivo_cargado)
    sucursales = ["Todas"] + datos['Sucursal'].unique().tolist()
    
    # Filtro de sucursal
    sucursal_seleccionada = st.sidebar.selectbox("Seleccionar Sucursal", sucursales)
    if sucursal_seleccionada != "Todas":
        datos = datos[datos['Sucursal'] == sucursal_seleccionada]
        st.title(f"Datos de {sucursal_seleccionada}")
    else:
        st.title("Datos de Todas las Sucursales")
    
    # Análisis por producto
    for producto in datos['Producto'].unique():
        # Mostrar el nombre del producto como subcabecera
        st.subheader(f"Producto: {producto}")
        
        datos_producto = datos[datos['Producto'] == producto]
        
        # Calcular métricas
        datos_producto['Precio_promedio'] = datos_producto['Ingreso_total'] / datos_producto['Unidades_vendidas']
        precio_promedio = datos_producto['Precio_promedio'].mean()
        variacion_precio_promedio_anual = datos_producto.groupby('Año')['Precio_promedio'].mean().pct_change().mean() * 100
        
        datos_producto['Ganancia'] = datos_producto['Ingreso_total'] - datos_producto['Costo_total']
        datos_producto['Margen'] = (datos_producto['Ganancia'] / datos_producto['Ingreso_total']) * 100
        margen_promedio = datos_producto['Margen'].mean()
        variacion_margen_promedio_anual = datos_producto.groupby('Año')['Margen'].mean().pct_change().mean() * 100
        
        unidades_vendidas = datos_producto['Unidades_vendidas'].sum()
        variacion_anual_unidades = datos_producto.groupby('Año')['Unidades_vendidas'].sum().pct_change().mean() * 100
        
        # Visualización
        col1, col2 = st.columns([0.25, 0.75])
        with col1:
            st.metric("Precio Promedio", f"${precio_promedio:,.0f}".replace(",", "."), f"{variacion_precio_promedio_anual:.2f}%")
            st.metric("Margen Promedio", f"{margen_promedio:.0f}%".replace(",", "."), f"{variacion_margen_promedio_anual:.2f}%")
            st.metric("Unidades Vendidas", f"{unidades_vendidas:,.0f}".replace(",", "."), f"{variacion_anual_unidades:.2f}%")
        with col2:
            fig = crear_grafico_ventas(datos_producto, producto)
            st.pyplot(fig)
else:
    st.subheader("Por favor, sube un archivo CSV desde la barra lateral.")
    mostrar_informacion_alumno()
