import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#url = https://tp8-59312-sigma.streamlit.app/

def generar_grafico_evolucion_ventas(datos_filtrados, nombre_producto):
    ventas_agrupadas = datos_filtrados.groupby(['Año', 'Mes'])['Unidades_vendidas'].sum().reset_index()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(len(ventas_agrupadas)), ventas_agrupadas['Unidades_vendidas'], label=nombre_producto)
    
    indices = np.arange(len(ventas_agrupadas))
    unidades_vendidas = ventas_agrupadas['Unidades_vendidas']
    coeficientes_tendencia = np.polyfit(indices, unidades_vendidas, 1)
    linea_tendencia = np.poly1d(coeficientes_tendencia)
    
    ax.plot(indices, linea_tendencia(indices), linestyle='--', color='red', label='Tendencia')
    ax.set_title('Evolución de Ventas Mensual')
    ax.set_xlabel('Año-Mes')
    ax.set_xticks(range(len(ventas_agrupadas)))
    
    etiquetas_xticks = []
    for i, fila in enumerate(ventas_agrupadas.itertuples()):
        if fila.Mes == 1:
            etiquetas_xticks.append(f"{fila.Año}")
        else:
            etiquetas_xticks.append("")
    ax.set_xticklabels(etiquetas_xticks)
    ax.set_ylabel('Unidades Vendidas')
    ax.set_ylim(0, None)
    ax.legend(title='Producto')
    ax.grid(True)
    
    return fig

def mostrar_datos_estudiante():
    with st.container():
        st.markdown('**Legajo:** 59.312')
        st.markdown('**Nombre:** Ignacio Matías Cancino')
        st.markdown('**Comisión:** C7')

st.sidebar.header("Cargar archivo de datos")
archivo_csv = st.sidebar.file_uploader("Subir archivo CSV", type=["csv"])

if archivo_csv is not None:
    datos_ventas = pd.read_csv(archivo_csv)
    lista_sucursales = ["Todas"] + datos_ventas['Sucursal'].unique().tolist()
    sucursal_elegida = st.sidebar.selectbox("Seleccionar Sucursal", lista_sucursales)
    
    if sucursal_elegida != "Todas":
        datos_ventas = datos_ventas[datos_ventas['Sucursal'] == sucursal_elegida]
        st.title(f"Datos de {sucursal_elegida}")
    else:
        st.title("Datos de Todas las Sucursales")
    
    lista_productos = datos_ventas['Producto'].unique()

    for nombre_producto in lista_productos:
        with st.container():
            st.subheader(f"{nombre_producto}")
            datos_producto = datos_ventas[datos_ventas['Producto'] == nombre_producto]
            datos_producto['Precio_promedio'] = datos_producto['Ingreso_total'] / datos_producto['Unidades_vendidas']
            promedio_precio = datos_producto['Precio_promedio'].mean()
            promedio_precio_anual = datos_producto.groupby('Año')['Precio_promedio'].mean()
            variacion_precio_anual = promedio_precio_anual.pct_change().mean() * 100
            
            datos_producto['Ganancia'] = datos_producto['Ingreso_total'] - datos_producto['Costo_total']
            datos_producto['Margen'] = (datos_producto['Ganancia'] / datos_producto['Ingreso_total']) * 100
            promedio_margen = datos_producto['Margen'].mean()
            promedio_margen_anual = datos_producto.groupby('Año')['Margen'].mean()
            variacion_margen_anual = promedio_margen_anual.pct_change().mean() * 100
            
            promedio_unidades = datos_producto['Unidades_vendidas'].mean()
            total_unidades_vendidas = datos_producto['Unidades_vendidas'].sum()
            unidades_agrupadas_anual = datos_producto.groupby('Año')['Unidades_vendidas'].sum()
            variacion_unidades_anual = unidades_agrupadas_anual.pct_change().mean() * 100
            
            columna_metrica, columna_grafico = st.columns([0.25, 0.75])
            
            with columna_metrica:
                st.metric(label="Precio Promedio", value=f"${promedio_precio:,.0f}".replace(",", "."), delta=f"{variacion_precio_anual:.2f}%")
                st.metric(label="Margen Promedio", value=f"{promedio_margen:.0f}%".replace(",", "."), delta=f"{variacion_margen_anual:.2f}%")
                st.metric(label="Unidades Vendidas", value=f"{total_unidades_vendidas:,.0f}".replace(",", "."), delta=f"{variacion_unidades_anual:.2f}%")
            
            with columna_grafico:
                grafico = generar_grafico_evolucion_ventas(datos_producto, nombre_producto)
                st.pyplot(grafico)
else:
    st.subheader("Por favor, sube un archivo CSV desde la barra lateral.")
    mostrar_datos_estudiante()