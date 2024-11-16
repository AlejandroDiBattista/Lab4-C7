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
        # Leer el archivo CSV
        df = pd.read_csv(archivo)
        return df
    except Exception as e:
        st.error(f"Error al cargar el archivo: {e}")
        return None

# Función para calcular el precio promedio, margen promedio y unidades vendidas por producto
def calcular_metricas(df):
    # Precio promedio = Ingreso total / Unidades vendidas
    df['Precio_promedio'] = df['Ingreso_total'] / df['Unidades_vendidas']

    # Margen promedio = (Ingreso total - Costo total) / Ingreso total
    df['Margen_promedio'] = (df['Ingreso_total'] - df['Costo_total']) / df['Ingreso_total']
    
    # Unidades vendidas = Suma de unidades vendidas por producto
    resumen = df.groupby('Producto').agg(
        Unidades_vendidas=('Unidades_vendidas', 'sum'),
        Precio_promedio=('Precio_promedio', 'mean'),
        Margen_promedio=('Margen_promedio', 'mean')
    ).reset_index()
    
    return resumen

# Función para graficar la evolución mensual de ventas por producto con línea de tendencia
def graficar_evolucion(df, producto):
    # Filtrar los datos para el producto seleccionado
    df_producto = df[df['Producto'] == producto]

    # Crear la columna de fecha con formato "Año-Mes"
    df_producto['Fecha'] = pd.to_datetime(df_producto['Año'].astype(str) + '-' + df_producto['Mes'].astype(str) + '-01', format='%Y-%m-%d')

    # Agrupar por fecha (Año-Mes) y sumar las unidades vendidas
    df_sum = df_producto.groupby(['Fecha'])[['Unidades_vendidas']].sum().reset_index()

    # Configuración del gráfico
    fig, ax = plt.subplots(figsize=(10, 6))

    # Graficar las unidades vendidas por mes
    ax.plot(df_sum['Fecha'], df_sum['Unidades_vendidas'], label=producto)

    # Ajuste de la línea de tendencia (polinomio de grado 1)
    z = np.polyfit(mdates.date2num(df_sum['Fecha']), df_sum['Unidades_vendidas'], 1)
    p = np.poly1d(z)
    ax.plot(df_sum['Fecha'], p(mdates.date2num(df_sum['Fecha'])), linestyle='--', color='red')

    # Título y configuración del gráfico
    ax.set_title(f"Evolución de Ventas Mensual")
    ax.set_xlabel("Año-Mes")
    ax.set_ylabel("Unidades Vendidas")
    ax.tick_params(axis='x', rotation=45)
    ax.legend(title="Producto")
    plt.tight_layout()

    return fig

# Interfaz con Streamlit
def app():
    st.title("Análisis de Ventas")

    # Barra lateral con el título y descripción
    st.sidebar.header("Carga archivo de datos")
    st.sidebar.write("Subir archivo CSV")

    # Subida de archivo CSV
    archivo = st.sidebar.file_uploader("Sube tu archivo CSV", type="csv", label_visibility="collapsed", accept_multiple_files=False, key="archivo")
    
    # Verificar si se subió un archivo
    if archivo is not None:
        # Cargar los datos del archivo CSV subido
        df = cargar_datos(archivo)
        
        if df is not None:
            # Barra lateral para seleccionar la sucursal
            sucursal_seleccionada = st.sidebar.selectbox('Selecciona una Sucursal', ['Todas'] + df['Sucursal'].unique().tolist())

            # Filtrar los datos por la sucursal seleccionada
            if sucursal_seleccionada != 'Todas':
                df = df[df['Sucursal'] == sucursal_seleccionada]

            # Calcular las métricas
            resumen = calcular_metricas(df)

            # Mostrar los gráficos y métricas para cada producto
            for _, row in resumen.iterrows():
                producto = row['Producto']
                precio_promedio = row['Precio_promedio']
                margen_promedio = row['Margen_promedio']
                unidades_vendidas = row['Unidades_vendidas']

                # Mostrar las métricas
                st.subheader(f"{producto}")
                st.write(f"**Precio Promedio:** ${precio_promedio:.2f}")
                st.write(f"**Margen Promedio:** {margen_promedio * 100:.2f}%")
                st.write(f"**Unidades Vendidas:** {unidades_vendidas}")

                # Mostrar el gráfico de evolución de ventas
                st.pyplot(graficar_evolucion(df, producto))

        else:
            st.error("No se pudo cargar el archivo correctamente.")
    else:
        st.info("Por favor, sube un archivo CSV para comenzar el análisis.")

if __name__ == '__main__':
    app()