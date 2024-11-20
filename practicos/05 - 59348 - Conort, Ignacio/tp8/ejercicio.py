import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

## ATENCION: Debe colocar la direccion en la que ha sido publicada la aplicacion en la siguiente linea\
# url = 'https://tp8-555555.streamlit.app/'

def mostrar_informacion_alumno():
    with st.container(border=True):
        st.markdown('**Legajo:** 59.348')
        st.markdown('**Nombre:** Ignacio Conort')
        st.markdown('**Comisión:** C7')

mostrar_informacion_alumno()

def cargar_datos(archivo):
    try:
        df = pd.read_csv(archivo)
        return df
    except Exception as e:
        st.error(f"Error al cargar el archivo: {e}")
        return None

def calcular_metricas(df):
    df['Precio_promedio'] = df['Ingreso_total'] / df['Unidades_vendidas']
    df['Margen_promedio'] = (df['Ingreso_total'] - df['Costo_total']) / df['Ingreso_total']

    resumen = df.groupby('Producto').agg(
        Unidades_vendidas=('Unidades_vendidas', 'sum'),
        Precio_promedio=('Precio_promedio', 'mean'),
        Margen_promedio=('Margen_promedio', 'mean')
    ).reset_index()
    return resumen

def graficar_evolucion(df, producto):
    df_producto = df[df['Producto'] == producto]

    df_producto['Fecha'] = pd.to_datetime(df_producto['Año'].astype(str) + '-' + df_producto['Mes'].astype(str) + '-01', format='%Y-%m-%d')

    df_sum = df_producto.groupby(['Fecha'])[['Unidades_vendidas']].sum().reset_index()

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(df_sum['Fecha'], df_sum['Unidades_vendidas'], label=producto)

    z = np.polyfit(mdates.date2num(df_sum['Fecha']), df_sum['Unidades_vendidas'], 1)
    p = np.poly1d(z)
    ax.plot(df_sum['Fecha'], p(mdates.date2num(df_sum['Fecha'])), linestyle='--', color='red')

    ax.set_title(f"Evolución de Ventas Mensual")
    ax.set_xlabel("Año-Mes")
    ax.set_ylabel("Unidades Vendidas")
    ax.tick_params(axis='x', rotation=45)
    ax.legend(title="Producto")
    plt.tight_layout()

    return fig

def app():
    st.title("Análisis de Ventas")

    st.sidebar.header("Carga archivo de datos")
    st.sidebar.write("Subir archivo CSV")

    archivo = st.sidebar.file_uploader("Sube tu archivo CSV", type="csv", label_visibility="collapsed", accept_multiple_files=False, key="archivo")
    
    if archivo is not None:
        df = cargar_datos(archivo)
        
        if df is not None:
            sucursal_seleccionada = st.sidebar.selectbox('Seleccionar Sucursal', ['Todas'] + df['Sucursal'].unique().tolist())

            if sucursal_seleccionada != 'Todas':
                df = df[df['Sucursal'] == sucursal_seleccionada]

            resumen = calcular_metricas(df)

            for _, row in resumen.iterrows():
                producto = row['Producto']
                precio_promedio = row['Precio_promedio']
                margen_promedio = row['Margen_promedio']
                unidades_vendidas = row['Unidades_vendidas']

                st.subheader(f"{producto}")
                st.write(f"**Precio Promedio:** ${precio_promedio:.2f}")
                st.write(f"**Margen Promedio:** {margen_promedio * 100:.2f}%")
                st.write(f"**Unidades Vendidas:** {unidades_vendidas}")

                st.pyplot(graficar_evolucion(df, producto))

        else:
            st.error("No se pudo cargar el archivo correctamente.")
    else:
        st.info("Por favor, sube un archivo CSV para comenzar el análisis.")

if __name__ == '__main__':
    app()