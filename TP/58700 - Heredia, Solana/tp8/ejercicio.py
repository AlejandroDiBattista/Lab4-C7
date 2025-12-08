import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

## ATENCION: Debe colocar la direccion en la que ha sido publicada la aplicacion en la siguiente linea\
# url = 'https://tp8-parciallab-58700.streamlit.app/'

#layout:'wide' permite mostrar mas info en una sola vista , evitando la necesidad de desplazarse(centra el contenido)
# Configuración de la página
st.set_page_config(layout="wide", page_title="Visualización de Ventas")

# Función para mostrar información del alumno
def mostrar_informacion_alumno():
    with st.container():
        st.markdown('**Legajo:** 58.700.')
        st.markdown('**Nombre:** Solana Heredia')
        st.markdown('**Comisión:** C7')

mostrar_informacion_alumno()

st.title('Estadísticas sobre las Ventas')
st.markdown("""
Esta aplicación permite cargar y visualizar datos de ventas, mostrando información clave y gráficos interactivos. Podrás analizar:
- Precio promedio y margen promedio de los productos.
- Gráficos de la evolución de ventas con líneas de tendencia.
""")

st.sidebar.header('Cargar archivo de datos CSV.')
archivo_csv = st.sidebar.file_uploader('Subir archivo CSV', type='csv')

if archivo_csv is not None:
    datos = pd.read_csv(archivo_csv)

    # Selección de sucursales
    sucursales = ['Todas'] + datos['Sucursal'].unique().tolist()
    seleccion_sucursal = st.sidebar.selectbox('Seleccione Sucursal', sucursales)

    # Filtrar los datos según la sucursal seleccionada
    if seleccion_sucursal != 'Todas':
        datos = datos[datos['Sucursal'] == seleccion_sucursal]
        st.title(f"Datos de {seleccion_sucursal}")
    else:
        st.title("Datos de todas las sucursales")

    productos = datos['Producto'].unique()
    for producto in productos:
        with st.container():
            
            st.subheader(f"{producto}")
            datos_producto = datos[datos['Producto'] == producto]

            
            datos_producto['Precio_promedio'] = datos_producto['Ingreso_total'] / datos_producto['Unidades_vendidas']
            precio_promedio = datos_producto['Precio_promedio'].mean()
            precio_promedio_anual = datos_producto.groupby('Año')['Precio_promedio'].mean()
            variacion_precio_promedio_anual = precio_promedio_anual.pct_change().mean() * 100

            datos_producto['Ganancia'] = datos_producto['Ingreso_total'] - datos_producto['Costo_total']
            datos_producto['Margen'] = (datos_producto['Ganancia'] / datos_producto['Ingreso_total']) * 100
            margen_promedio = datos_producto['Margen'].mean()
            margen_promedio_anual = datos_producto.groupby('Año')['Margen'].mean()
            variacion_margen_promedio_anual = margen_promedio_anual.pct_change().mean() * 100

            unidades_vendidas = datos_producto['Unidades_vendidas'].sum()
            unidades_por_año = datos_producto.groupby('Año')['Unidades_vendidas'].sum()
            variacion_anual_unidades = unidades_por_año.pct_change().mean() * 100

            
            col_izq, col_der = st.columns([0.25, 0.75])

            with col_izq:
                st.metric('Precio Promedio', f"${precio_promedio:,.0f}", f"{variacion_precio_promedio_anual:.2f}%")
                st.metric('Margen Promedio', f"{margen_promedio:.0f}%", f"{variacion_margen_promedio_anual:.2f}%")
                st.metric('Unidades Vendidas', f"{unidades_vendidas:,.0f}", f"{variacion_anual_unidades:.2f}%")

           
            with col_der:
                datos_producto['Fecha'] = pd.to_datetime(datos_producto['Año'].astype(str) + '-' + datos_producto['Mes'].astype(str) + '-01')
                datos_producto.sort_values('Fecha', inplace=True)

                plt.figure(figsize=(10, 4))
                sns.set_style('whitegrid')
                sns.lineplot(data=datos_producto, x='Fecha', y='Unidades_vendidas', label=f"{producto}", color='blue')

                z = np.polyfit(datos_producto['Fecha'].map(pd.Timestamp.toordinal), datos_producto['Unidades_vendidas'], 1)
                p = np.poly1d(z)
                plt.plot(datos_producto['Fecha'], p(datos_producto['Fecha'].map(pd.Timestamp.toordinal)), color='red', linestyle='--', label='Tendencia')

                plt.title(f"Evolución de Ventas Mensual para {producto}", fontsize=16, fontweight='bold')
                plt.xlabel('Año-Mes')
                plt.ylabel('Unidades Vendidas')
                plt.xticks(rotation=45)
                plt.legend(loc='upper right')

                st.pyplot(plt)

            st.write("---")

                
    
    


