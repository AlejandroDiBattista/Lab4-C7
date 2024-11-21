import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

## ATENCION: Debe colocar la direccion en la que ha sido publicada la aplicacion en la siguiente linea\
# url = 'https://tp8-parciallab-58700.streamlit.app/'

#layout:'wide' permite mostrar mas info en una sola vista , evitando la necesidad de desplazarse(centra el contenido)
st.set_page_config(layout="wide", page_title="Visualización de Ventas")#define configuraciones globales para la app
def mostrar_informacion_alumno():
    with st.container(border=True):
        st.markdown('**Legajo:** 58.700.')
        st.markdown('**Nombre:** Solana Heredia')
        st.markdown('**Comisión:** C7')

mostrar_informacion_alumno()

st.title('Estadisticas sobre las ventas')
st.markdown("""
            Esta aplicacion permite cargar y visualizar datos de ventas, mostrando informacion clave y graficos interactivos. Puedes ver:
            - Precio promedio y margen promedio de los productos.
            - Graficos de la evolucion de ventas con lineas de tendencia.
            """)
st.sidebar.header('Carge archivo de datos CSV.')
archivo_cvs = st.sidebar.file_uploader('Subir archivo CSV', type= 'csv')
seleccion_sucursal = st.sidebar.selectbox('Seleccione Sucursal',['Todas','Sucursal Norte', 'Sucursal Sur', 'Sucursal Centro'])

if archivo_cvs is not None:
    df = pd.read_csv(archivo_cvs)
    
    if seleccion_sucursal !='Todas':
        df = df[df['Sucursal'] == seleccion_sucursal]
        
    productos = df['Producto'].unique()
    for producto in productos:
        with st.container():
            df_producto = df[df['Producto'] == producto]
            
            venta_anual = df_producto.groupby('Año')['Unidades_vendidas'].sum()
            ingresos_anuales = df_producto.groupby('Año')['Ingreso_total'].sum()
            costo_anual = df_producto.groupby('Año')['Costo_total'].sum()
            
            precio_promedio = (df_producto['Ingreso_total'].sum() / df_producto['Unidades_vendidas'].sum())
            margen_promedio = ((df_producto['Ingreso_total'].sum() - df_producto['Costo_total'].sum()) / df_producto['Ingreso_total'].sum())
            margen_promedio_redondeado = round(margen_promedio * 100)
            margen_anual = (ingresos_anuales - costo_anual) / ingresos_anuales
            
            variacion_porcentual = venta_anual.pct_change().iloc[-1]*100
            precio_promedio_anual = ingresos_anuales / venta_anual
            variacion_precio_promedio = precio_promedio_anual.pct_change().iloc[-1]*100
            variacion_margen_anual = margen_anual.pct_change().iloc[-1]*100
            
            unidades_venidas = df_producto['Unidades_vendidas'].sum()
            

            col_izq, col_der = st.columns([1,2])
            with col_izq:
                st.subheader(f"{producto}")
                st.metric('Precio Promedio', f"${precio_promedio:,.2f}", f"{variacion_precio_promedio:.2f}%", delta_color="normal")
                st.metric('Margen Promedio', f"{round(margen_promedio *100)}%", f"{variacion_margen_anual:.2f}%")
                st.metric('Unidades Vendidas', f"{unidades_venidas:,.2f}", f"{variacion_porcentual:.2f}%")
            
        
            with col_der:
                df_producto['Año'] = df_producto['Año'].astype(int)
                df_producto['Mes'] = df_producto['Mes'].astype(int)
                df_producto['Fecha'] = pd.to_datetime(df_producto['Año'].astype(str) + '-' + df_producto['Mes'].astype(str) + '-01')
                df_producto.sort_values('Fecha', inplace=True)
            
                plt.figure(figsize=((10,4)))
                sns.set_style('whitegrid') #estilos para graficos bonicos
                sns.set_palette("coolwarm")
            
                sns.lineplot(data=df_producto, x='Fecha', y='Unidades_vendidas', label=f"{producto}", color='blue')
                z = np.polyfit(df_producto['Fecha'].map(pd.Timestamp.toordinal), df_producto['Unidades_vendidas'], 1)
                p = np.poly1d(z)
                plt.plot(df_producto['Fecha'], p(df_producto['Fecha'].map(pd.Timestamp.toordinal)), color='red', linestyle='--', label='Tendencia')
                        
                plt.title(f"Evolucion de Ventas Mensual para {producto}", fontsize=16, fontweight='bold')
                plt.xlabel('Año-Mes')
                plt.ylabel('Unidades Vendidas', fontsize=12)
                plt.legend(loc='upper right')
                plt.xticks(rotation=45)
                
                st.pyplot(plt)
            st.write("---")

                
    
    


