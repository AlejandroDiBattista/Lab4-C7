import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
#"https://ssz6djgbnm8cxvykmdyng3.streamlit.app/"
# Configuración de la página
st.set_page_config(page_title="Análisis de Ventas", layout="wide")
st.title("Análisis de Ventas")
def mostrar_informacion_alumno():
    with st.container(border=True):
        st.markdown('**Legajo:** 59.162')
        st.markdown('**Nombre:** Nuñez Orquera Luis Alberto')
        st.markdown('**Comisión:** C7')
mostrar_informacion_alumno()

st.sidebar.header("Cargar archivo de datos")
uploaded_file = st.sidebar.file_uploader("Sube un archivo de ventas", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    
    sucursal = st.sidebar.selectbox("Seleccionar Sucursal", ["Todas"] + list(df["Sucursal"].unique()))
    if sucursal != "Todas":
        df = df[df["Sucursal"] == sucursal]

    st.subheader(f"Datos de {'Todas las Sucursales' if sucursal == 'Todas' else sucursal}")

    df['Precio_unitario'] = df['Ingreso_total'] / df['Unidades_vendidas']

    df['Margen'] = (df['Ingreso_total'] - df['Costo_total']) / df['Ingreso_total'] * 100

    sumas = df.groupby('Producto').agg({
        'Unidades_vendidas': 'sum',
        'Ingreso_total': 'sum',
        'Costo_total': 'sum',
        'Precio_unitario': 'sum'  
    })

    conteo = df.groupby('Producto').size()

    df_grouped = sumas.copy()
    df_grouped['Precio_promedio'] = df_grouped['Precio_unitario'] / conteo
    df_grouped = df_grouped.drop('Precio_unitario', axis=1)
    df_grouped = df_grouped.reset_index()

    ganancia = df_grouped['Ingreso_total'] - df_grouped['Costo_total']
    df_grouped['Margen_promedio'] = (ganancia / df_grouped['Ingreso_total']) * 100
    df_grouped['Margen_promedio'] = df_grouped['Margen_promedio'].round(2)

    st.sidebar.subheader("Personalización del gráfico")
    color_line = st.sidebar.color_picker("Elige un color para la línea", "#1f77b4")

    for i, row in df_grouped.iterrows():
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.markdown(f"### {row['Producto']}")
            
            datos_producto = df[df["Producto"] == row["Producto"]]
            margen_promedio_anual = datos_producto.groupby('Año')['Margen'].mean()
            variacion_margen_promedio_anual = margen_promedio_anual.pct_change().mean() * 100
            
            unidades_promedio = datos_producto['Unidades_vendidas'].mean()
            unidades_vendidas = datos_producto['Unidades_vendidas'].sum()
            
            unidades_por_año = datos_producto.groupby('Año')['Unidades_vendidas'].sum()
            variacion_anual_unidades = unidades_por_año.pct_change().mean() * 100
            
            datos_producto['Precio_promedio'] = datos_producto['Ingreso_total'] / datos_producto['Unidades_vendidas']
            precio_promedio_anual = datos_producto.groupby('Año')['Precio_promedio'].mean()
            variacion_precio_promedio_anual = precio_promedio_anual.pct_change().mean() * 100
            
            st.metric("Precio Promedio", f"${row['Precio_promedio']:.2f}", f"{variacion_precio_promedio_anual:.2f}%")
            st.metric("Margen Promedio", f"{row['Margen_promedio']:.2f}%", f"{variacion_margen_promedio_anual:.2f}%")
            st.metric("Unidades Vendidas", f"{row['Unidades_vendidas']:,}", f"{variacion_anual_unidades:.2f}%")

        with col2:
            product_data = df[df["Producto"] == row["Producto"]]
            product_data['Fecha'] = pd.to_datetime(product_data['Año'].astype(str) + '-' + 
                                         product_data['Mes'].astype(str).str.zfill(2))
            ventas_por_periodo = product_data.groupby('Fecha')['Unidades_vendidas'].sum().reset_index()
            ventas_por_periodo = ventas_por_periodo.sort_values('Fecha')

            plt.figure(figsize=(10, 6))
            
            plt.plot(
                ventas_por_periodo['Fecha'],
                ventas_por_periodo['Unidades_vendidas'],
                color=color_line,
                marker='o',
                label=row["Producto"],
                linestyle='-'
            )

            if len(ventas_por_periodo) > 1:
                x = np.arange(len(ventas_por_periodo))
                y = ventas_por_periodo['Unidades_vendidas']
                tendencia = np.polyfit(x, y, 1)
                linea_tendencia = np.poly1d(tendencia)
                plt.plot(
                    ventas_por_periodo['Fecha'],
                    linea_tendencia(x),
                    linestyle='--',
                    color='red',
                    label='Tendencia'
                )

            plt.grid(True, linestyle='--', alpha=0.6)
            plt.xticks(rotation=45)
            plt.title(f'Evolución de Ventas - {row["Producto"]}')
            plt.xlabel('Fecha')
            plt.ylabel('Unidades vendidas')
            plt.legend()
            plt.tight_layout()
            st.pyplot(plt)
            plt.close()

        st.markdown("---")

else:
    st.info("Sube un archivo CSV para comenzar el análisis.")