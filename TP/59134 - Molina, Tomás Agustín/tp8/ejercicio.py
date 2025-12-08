import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## ATENCION: Debe colocar la direccion en la que ha sido publicada la aplicacion en la siguiente linea\
# url = 'https://tp8-59134-c7.streamlit.app/'

st.set_page_config(
    page_title="Análisis de Ventas",                
    layout="wide")                   

def generar_grafico_ventas(ventas_filtradas, nombre_producto):

    ventas_agrupadas = ventas_filtradas.groupby(['Año', 'Mes'])['Unidades_vendidas'].sum().reset_index()

    figura, eje = plt.subplots(figsize=(10, 5))
    eje.plot(range(len(ventas_agrupadas)), ventas_agrupadas['Unidades_vendidas'], label=nombre_producto)
    
    indices = np.arange(len(ventas_agrupadas))
    valores = ventas_agrupadas['Unidades_vendidas']
    coeficientes = np.polyfit(indices, valores, 1)
    tendencia = np.poly1d(coeficientes)
    eje.plot(indices, tendencia(indices), linestyle='--', color='red', label='Tendencia')
    
    eje.set_title('Evolución de Ventas Mensual', fontsize=14)
    eje.set_xlabel('Año-Mes', fontsize=12)
    eje.set_ylabel('Unidades Vendidas', fontsize=12)
    eje.set_ylim(0, None)
    eje.legend(title='Producto')
    eje.grid(True)

    etiquetas_eje = [f"{row.Año}" if row.Mes == 1 else "" for row in ventas_agrupadas.itertuples()]
    eje.set_xticks(indices)
    eje.set_xticklabels(etiquetas_eje)

    return figura

def mostrar_info_usuario():
    with st.container():
        st.markdown('**Legajo del alumno:** 59.134')
        st.markdown('**Nombre completo:** Molina, Tomás Agustín')
        st.markdown('**Comisión:** C7')


st.sidebar.header("Cargar datos")
archivo_subido = st.sidebar.file_uploader("Subir archivo CSV", type=["csv"])

if archivo_subido:
   
    datos = pd.read_csv(archivo_subido)
    
 
    lista_sucursales = ["Todas las sucursales"] + datos['Sucursal'].unique().tolist()
    sucursal_elegida = st.sidebar.selectbox("Elige una sucursal", lista_sucursales)
    
 
    if sucursal_elegida != "Todas las sucursales":
        datos = datos[datos['Sucursal'] == sucursal_elegida]
        st.title(f"Análisis de {sucursal_elegida}")
    else:
        st.title("Análisis De Todas Las Sucursales")
    

    productos_unicos = datos['Producto'].unique()

    for producto in productos_unicos:
        with st.container(border=True):
        
            st.subheader(f"{producto}")
            datos_producto = datos[datos['Producto'] == producto]
            
            datos_producto['Precio_unitario_promedio'] = datos_producto['Ingreso_total'] / datos_producto['Unidades_vendidas']
            precio_promedio = datos_producto['Precio_unitario_promedio'].mean()
            
            datos_producto['Ganancia_neta'] = datos_producto['Ingreso_total'] - datos_producto['Costo_total']
            datos_producto['Margen_ganancia'] = (datos_producto['Ganancia_neta'] / datos_producto['Ingreso_total']) * 100
            margen_promedio = datos_producto['Margen_ganancia'].mean()
            
            promedio_anual = datos_producto.groupby('Año')['Precio_unitario_promedio'].mean()
            variacion_precio_anual = promedio_anual.pct_change().mean() * 100
            
            margen_anual = datos_producto.groupby('Año')['Margen_ganancia'].mean()
            variacion_margen_anual = margen_anual.pct_change().mean() * 100
            
            unidades_totales = datos_producto['Unidades_vendidas'].sum()
            unidades_anuales = datos_producto.groupby('Año')['Unidades_vendidas'].sum()
            variacion_unidades_anual = unidades_anuales.pct_change().mean() * 100
            
            col_metrica, col_grafico = st.columns([0.3, 0.7])
            
            with col_metrica:
                st.metric("Precio Promedio", f"${precio_promedio:,.0f}", f"{variacion_precio_anual:.0f}%")
                st.metric("Margen Promedio", f"{margen_promedio:.0f}%", f"{variacion_margen_anual:.0f}%")
                st.metric("Total Unidades", f"{unidades_totales:,.0f}", f"{variacion_unidades_anual:.2f}%")
            
            with col_grafico:
                grafico = generar_grafico_ventas(datos_producto, producto)
                st.pyplot(grafico)
else:
    st.subheader("Esperando la carga de un archivo CSV.")
    mostrar_info_usuario()

