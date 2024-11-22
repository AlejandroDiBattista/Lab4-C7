import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## ATENCION: Debe colocar la direccion en la que ha sido publicada la aplicacion en la siguiente linea\
# url = 'https://tp8-58728.streamlit.app/'

def mostrar_informacion_alumno():
    with st.container():
        st.markdown('**Legajo:** 58728')
        st.markdown('**Nombre:** Martino Montivero Lizondo')
        st.markdown('**Comisión:** C7')

mostrar_informacion_alumno()

if 'paraelboton' not in st.session_state:
    st.session_state.paraelboton = False

st.title(" Informe de Ventas")


datos = st.sidebar.file_uploader("Subir informe", type=["csv"])


if datos is not None:
    df = pd.read_csv(datos)
    
    if st.button("Mostrar informe"):
        st.session_state.paraelboton = not st.session_state.paraelboton
    
    if st.session_state.paraelboton:
        st.write(df)

    opciones = df["Sucursal"].unique()
    opcion_todas = "Todas"
    sucursal_elegida = st.sidebar.selectbox("Seleccione una sucursal", options=[opcion_todas] + list(opciones))

    
    if sucursal_elegida != opcion_todas:
        df = df[df["Sucursal"] == sucursal_elegida]
    productos = df["Producto"].unique()

    st.header(f"Analisis de ventas para la sucursal: {sucursal_elegida}")

    
    
    def calcular_metricas(df, nombre_producto):
        # Filtrar datos del producto específico
        dfproducto = df[df['Producto'] == nombre_producto]
        
        # Calcular columnas adicionales necesarias
        dfproducto["Ingreso_por_unidad"] = dfproducto["Ingreso_total"] / dfproducto["Unidades_vendidas"]
        dfproducto["Porcentaje_margen"] = ((dfproducto["Ingreso_total"] - dfproducto["Costo_total"]) / dfproducto["Ingreso_total"]) * 100
        dfproducto["Beneficio"] = dfproducto["Ingreso_total"] - dfproducto["Costo_total"]
        
        # datos por año 
        precioxaño = dfproducto.groupby("Año")["Ingreso_por_unidad"].mean()
        ventasxaño = dfproducto.groupby("Año")["Unidades_vendidas"].sum()
        margenxaño = dfproducto.groupby("Año")["Porcentaje_margen"].mean()

        # datos promedio
        preciopromedio = dfproducto["Ingreso_por_unidad"].mean()
        margenpromedio = dfproducto["Porcentaje_margen"].mean()
        totalunidades = dfproducto["Unidades_vendidas"].sum()
        
       
        # Calcular las metricas
        cambio_precio = precioxaño.pct_change().mean() * 100
        cambio_margen = margenxaño.pct_change().mean() * 100
        cambio_ventas = ventasxaño.pct_change().mean() * 100
        
    
        return pd.Series({
            'preciopromedio': preciopromedio,
            'margenpromedio': margenpromedio,
            'totalunidades': int(totalunidades),
            'cambio_precio': cambio_precio,
            'cambio_margen': cambio_margen,
            'cambio_ventas': cambio_ventas
        })

   
    def grafico(data, producto):
        ventas = data.groupby(["Año", "Mes"])["Unidades_vendidas"].sum().reset_index()
        fechas = range(len(ventas))
        
        # Crear gráfico de evolución
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(fechas, ventas["Unidades_vendidas"], label=f"{producto}", marker='o', linestyle='-', color='blue')
        
        # Calcular línea de tendencia
        x = np.arange(len(ventas))
        y = ventas["Unidades_vendidas"]
        coeficientes = np.polyfit(x, y, 1)
        tendencia = np.poly1d(coeficientes)
        ax.plot(x, tendencia(x), linestyle="--", color="red", label="Tendencia")
        
        # Configuración del gráfico
        ax.set_title(f"Evolución de Ventas Mensuales")
        ax.set_xlabel("Tiempo (Año)")
        ax.set_ylabel("Unidades Vendidas")
        ax.grid(True)
        ax.legend()
        
        
        # Establecer solo un punto por año
        años = sorted(set([fila.Año for fila in ventas.itertuples()]))
        posiciones = [ventas[ventas["Año"] == año].index[0] for año in años]
        
        ax.set_xticks(posiciones)
        ax.set_xticklabels(años)
        
        plt.tight_layout()
        
        return fig


    
    for producto in productos:
        st.subheader(producto)
        st.divider()
        
        indicadores = calcular_metricas(df, producto)
        
        # Mostrar métricas
        col1, col2, col3 = st.columns(3)
        col1.metric("Precio Promedio", f"${indicadores['preciopromedio']:.2f}", f"{indicadores['cambio_precio']:+.2f}%")
        col2.metric("Margen Promedio", f"{indicadores['margenpromedio']:.2f}%", f"{indicadores['cambio_margen']:+.2f}%")
        col3.metric("Unidades Vendidas", f"{int(indicadores['totalunidades']):,}", f"{indicadores['cambio_ventas']:+.2f}%")
        
        # Mostrar gráfico
        fig = grafico(df[df["Producto"] == producto], producto)
        st.pyplot(fig)

