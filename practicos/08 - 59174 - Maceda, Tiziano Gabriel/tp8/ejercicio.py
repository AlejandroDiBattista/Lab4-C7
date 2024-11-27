import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## ATENCION: Debe colocar la direccion en la que ha sido publicada la aplicacion en la siguiente linea\

#url: https://59174-tiziano-maceda.streamlit.app/

def grafico_ventas(datos_filtro, nombre_producto):
    resumen_ventas = datos_filtro.groupby(['Año', 'Mes'])['Unidades_vendidas'].sum().reset_index()

    figura, eje = plt.subplots(figsize=(8, 4))
    eje.plot(range(len(resumen_ventas)), resumen_ventas['Unidades_vendidas'], linewidth=2.5, label=nombre_producto)

    x_valores = np.arange(len(resumen_ventas))
    y_valores = resumen_ventas['Unidades_vendidas']
    coeficientes = np.polyfit(x_valores, y_valores, 1)
    tendencia = np.poly1d(coeficientes)

    eje.plot(x_valores, tendencia(x_valores), linestyle='--', linewidth=2, color='red', label='Tendencia')

    eje.set_title('Ventas Mensuales por Producto')
    eje.set_xlabel('Fecha')
    eje.set_xticks(range(len(resumen_ventas)))

    etiquetas_eje = []
    for i, fila in resumen_ventas.iterrows():
        etiquetas_eje.append(f"{fila['Año']}" if fila['Mes'] == 1 else "")
    eje.set_xticklabels(etiquetas_eje)
    eje.set_ylabel('Total Unidades Vendidas')
    eje.set_ylim(None, None)
    eje.legend(title='Producto', frameon=True, facecolor='white', edgecolor='none')
    eje.grid(True, linestyle='--', alpha=0.5, color='#cccccc')

    return figura

def info_alumno():
    with st.container():
         st.markdown("Legajo: 59.174")
         st.markdown("Nombre: Tiziano Maceda")
         st.markdown("Comisión: C7")

st.sidebar.header("Cargar Datos")
archivo = st.sidebar.file_uploader("Subir archivo CSV", type=["csv"])

if archivo is not None:
    datos = pd.read_csv(archivo)

    lista_sucursales = ["Todas"] + datos['Sucursal'].unique().tolist()
    sucursal_elegida = st.sidebar.selectbox("Elige una Sucursal", lista_sucursales)

    if sucursal_elegida != "Todas":
        datos = datos[datos['Sucursal'] == sucursal_elegida]
        st.title(f"Análisis de {sucursal_elegida}")
    else:
        st.title("Análisis de Todas las Sucursales")

    productos = datos['Producto'].unique()

    for producto in productos:
        with st.container():
            st.subheader(f"{producto}")
            datos_filtrados = datos[datos['Producto'] == producto].copy()

            datos_filtrados['Promedio_Precio'] = datos_filtrados['Ingreso_total'] / datos_filtrados['Unidades_vendidas']
            promedio_precio = datos_filtrados['Promedio_Precio'].mean()

            promedio_anual_precio = datos_filtrados.groupby('Año')['Promedio_Precio'].mean()
            variacion_precio = promedio_anual_precio.pct_change().mean(skipna=True) * 100

            datos_filtrados['Ganancia'] = datos_filtrados['Ingreso_total'] - datos_filtrados['Costo_total']
            datos_filtrados['Margen'] = (datos_filtrados['Ganancia'] / datos_filtrados['Ingreso_total']) * 100
            promedio_margen = datos_filtrados['Margen'].mean()

            promedio_anual_margen = datos_filtrados.groupby('Año')['Margen'].mean()
            variacion_margen = promedio_anual_margen.pct_change().mean(skipna=True) * 100

            total_unidades = datos_filtrados['Unidades_vendidas'].sum()
            unidades_anuales = datos_filtrados.groupby('Año')['Unidades_vendidas'].sum()
            variacion_unidades = unidades_anuales.pct_change().mean(skipna=True) * 100

            col1, col2 = st.columns([0.25, 0.75])

            with st.container():
                col1, col2, col3 = st.columns(3)
                col1.metric(label="Precio Promedio", value=f"${promedio_precio:,.0f}".replace(",", "."), delta=f"{variacion_precio:.2f}%")
                col2.metric(label="Margen Unidades", value=f"{promedio_margen:.0f}%".replace(",", "."), delta=f"{variacion_margen:.2f}%")
                col3.metric(label="Unidades Totales", value=f"{total_unidades:,.0f}".replace(",", "."), delta=f"{variacion_unidades:.2f}%")

            grafico = grafico_ventas(datos_filtrados, producto)
            st.pyplot(grafico, clear_figure=True)
else:
    st.subheader("Por favor, carga un archivo CSV desde la barra lateral.")
    info_alumno()