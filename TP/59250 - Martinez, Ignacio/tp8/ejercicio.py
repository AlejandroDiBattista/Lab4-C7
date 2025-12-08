import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## ATENCION: Debe colocar la direccion en la que ha sido publicada la aplicacion en la siguiente linea\
# url = 'https://tp8-59250.streamlit.app/'



def mostrar_informacion_alumno():
    with st.container():
        st.markdown('**📋 Legajo:** 59.250')
        st.markdown('**👤 Nombre:** Ignacio Martinez')
        st.markdown('**🏫 Comisión:** C7')

def crear_grafico_ventas(datos_producto, producto):
    ventas_por_producto = datos_producto.groupby(['Año', 'Mes'])['Unidades_vendidas'].sum().reset_index()
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.plot(range(len(ventas_por_producto)), ventas_por_producto['Unidades_vendidas'], label=producto, marker='o', color='blue', linewidth=2)
    x = np.arange(len(ventas_por_producto))
    y = ventas_por_producto['Unidades_vendidas']
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    ax.plot(x, p(x), linestyle='--', color='red', label='Tendencia 📈', linewidth=1.5)
    ax.set_title('📊 Evolución de Ventas Mensual', fontsize=16, fontweight='bold')
    ax.set_xlabel('Año-Mes', fontsize=12)
    ax.set_ylabel('Unidades Vendidas', fontsize=12)
    ax.legend(title='Marca', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.tick_params(axis='x', rotation=45)
    ax.set_ylim(0, None)
    etiquetas = [f"{row.Año}" if row.Mes == 1 else "" for row in ventas_por_producto.itertuples()]
    ax.set_xticks(range(len(ventas_por_producto)))
    ax.set_xticklabels(etiquetas)
    return fig

st.sidebar.header("📂 Cargar archivo de datos")
archivo_cargado = st.sidebar.file_uploader("⬆️ Subir archivo CSV", type=["csv"])

if archivo_cargado is not None:
    datos = pd.read_csv(archivo_cargado)
    sucursales = ["🌍 Todas"] + datos['Sucursal'].unique().tolist()
    sucursal_seleccionada = st.sidebar.selectbox("📍 Seleccionar Sucursal", sucursales)

    if sucursal_seleccionada != "🌍 Todas":
        datos = datos[datos['Sucursal'] == sucursal_seleccionada]
        st.title(f"📊 Datos de {sucursal_seleccionada}")
    else:
        st.title("📊 Análisis de las sucursales")

    productos = datos['Producto'].unique()
    for producto in productos:
        with st.container():
            st.subheader(f"🛒 Producto: {producto}")
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
            unidades_promedio = datos_producto['Unidades_vendidas'].mean()
            unidades_vendidas = datos_producto['Unidades_vendidas'].sum()
            unidades_por_año = datos_producto.groupby('Año')['Unidades_vendidas'].sum()
            variacion_anual_unidades = unidades_por_año.pct_change().mean() * 100
            col1, col2 = st.columns([0.3, 0.7])
            with col1:
                st.metric(label="💲 Precio Promedio", value=f"${precio_promedio:,.0f}".replace(",", "."), delta=f"{variacion_precio_promedio_anual:.2f}% 📉")
                st.metric(label="📈 Margen Promedio", value=f"{margen_promedio:.0f}%".replace(",", "."), delta=f"{variacion_margen_promedio_anual:.2f}% 📉")
                st.metric(label="📦 Unidades Vendidas", value=f"{unidades_vendidas:,.0f}".replace(",", "."), delta=f"{variacion_anual_unidades:.2f}% 📉")
            with col2:
                fig = crear_grafico_ventas(datos_producto, producto)
                st.pyplot(fig)
else:
    st.subheader("⚠️ Cargar tus datos desde la barra lateral para analizar.")
    mostrar_informacion_alumno()
