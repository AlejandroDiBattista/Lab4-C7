import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

## ATENCION: Debe colocar la direccion en la que ha sido publicada la aplicacion en la siguiente linea\
# url = 'https://tp8-555555.streamlit.app/'

#layout:'wide' permite mostrar mas info en una sola vista , evitando la necesidad de desplazarse
st.set_page_config(layout="wide", page_title="Visualización de Ventas")#define configuraciones globales para la app
def mostrar_informacion_alumno():
    with st.container(border=True):
        st.markdown('**Legajo:** 58.700.')
        st.markdown('**Nombre:** Solana Heredia')
        st.markdown('**Comisión:** C7')

mostrar_informacion_alumno()

st.title('Estadisticas sobre las ventas')
st.sidebar.header('Carge archivo de datos CSV.')
archivo_cvs = st.sidebar.file_uploader('Subir archivo CSV', type= 'csv')
seleccion_sucursal = st.sidebar.selectbox('Seleccione Sucursal',['Todas','Sucursal Norte', 'Sucursal Sur', 'Sucursal Centro'])

if archivo_cvs is not None:
    df = pd.read_csv(archivo_cvs)
    
    if seleccion_sucursal !='Todas':
        df = df[df['Sucursal'] == seleccion_sucursal]
    productos = df['Producto'].unique()
    for producto in productos:
        df_producto = df[df['Producto'] == producto]
        precio_promedio = (df_producto['Ingreso_total'].sum() / df_producto['Unidades_vendidas'].sum())
        margen_promedio = ((df_producto['Ingreso_total'].sum() - df_producto['Costo_total'].sum()) / df_producto['Inegreso_total'].sum())
        unidades_venidas = df_producto['Unidades_vendidas'].sum()

        st.subheader(f"{producto}")
        col1, col2, col3 = st.columns(3)
        col1.metric('Precio Promedio', f"${precio_promedio:.2f}")
        col2.metric('Margen Promedio', f"{margen_promedio:.2f}")
        col3.metric('Unidades Vendidas', f"{unidades_venidas:,.0f}")
        
        df_producto['Fecha'] = pd.to_datetime(df_producto[['Año', 'Mes']].assign(day=1))
        df_producto.sort_values('Fecha', inplace=True)
        
        plt.figure(figsize=((12,6)))
        sns.lineplot(data=df_producto, x='Fecha', y='Unidades_vendidas', label=producto)
        plt.title(f"Evolucion de Ventas Mensual para {producto}")
        plt.xlabel('Año-Mes')
        plt.ylabel('Unidades Vendidas')
        plt.xticks(rotacion=45)
        
        z= np.poly1d(df_producto['Fecha'].map(pd.Timestamp.toordinal), df_producto['Unidades_vendidas'], 1)
        
        p = np.poly1d(z)
        
        plt.plot(df_producto['Fecha'], p(df_producto['Fecha'].map(pd.Timestamp.toordinal)), color='red', linestyle='--',label='Tendencia')
        st.pyplot(plt)
        
    
    


