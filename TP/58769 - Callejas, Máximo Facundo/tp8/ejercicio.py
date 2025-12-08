import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## ATENCION: Debe colocar la direccion en la que ha sido publicada la aplicacion en la siguiente linea\
# url = 'https://tp8-58769.streamlit.app/'


st.set_page_config(page_title="Análisis de Ventas", layout="wide")



def mostrar_informacion_alumno():
    with st.container(border=True):
        st.markdown("**Legajo:** 58.769")
        st.markdown("**Nombre:** Máximo Callejas")
        st.markdown("**Comisión:** C7")



def cargarDatos():
    st.sidebar.markdown("## Cargar archivo de datos")
    uploaded_file = st.sidebar.file_uploader("**Subir archivo CSV**", type=["csv"])
    if uploaded_file:
        try:
            data = pd.read_csv(uploaded_file)
            data['Orden_original'] = data.index
            st.sidebar.success("Datos cargados correctamente")
            return data
        except Exception as e:
            st.sidebar.error(f"Error al cargar los datos: {e}")
    return None



def calcularMetricas(data):
    data['Fecha'] = pd.to_datetime(data['Año'].astype(str) + "-" + data['Mes'].astype(str), format="%Y-%m")
    data['Precio_promedio'] = np.round(data['Ingreso_total'] / data['Unidades_vendidas'])
    data['Margen_promedio'] = (data['Ingreso_total'] - data['Costo_total']) / data['Ingreso_total']

    data = data.sort_values('Orden_original')

    resumen_anual = data.groupby(['Producto', 'Año'], group_keys=False).agg({
        'Unidades_vendidas': 'sum',
        'Ingreso_total': 'sum',
        'Costo_total': 'sum',
        'Precio_promedio': 'mean',
        'Margen_promedio': 'mean'
    }).reset_index()

    variaciones = []

    for producto in data['Producto'].unique():
        producto_data = resumen_anual[resumen_anual['Producto'] == producto].sort_values('Año')
        if len(producto_data['Año'].unique()) < 2:
            variaciones.append({
                'Producto': producto,
                'Variacion_precio': None,
                'Variacion_margen': None,
                'Variacion_unidades': None
            })
            continue
        
        incrementos_precio = []
        incrementos_margen = []
        incrementos_unidades = []
        
        for i in range(1, len(producto_data)):
            unidades_anterior = producto_data.iloc[i - 1]['Unidades_vendidas']
            unidades_actual = producto_data.iloc[i]['Unidades_vendidas']
            incrementos_unidades.append((unidades_actual - unidades_anterior) / unidades_anterior * 100)
            precio_anterior = producto_data.iloc[i - 1]['Precio_promedio']
            precio_actual = producto_data.iloc[i]['Precio_promedio']
            incrementos_precio.append((precio_actual - precio_anterior) / precio_anterior * 100)
            margen_anterior = producto_data.iloc[i - 1]['Margen_promedio']
            margen_actual = producto_data.iloc[i]['Margen_promedio']
            incrementos_margen.append((margen_actual - margen_anterior) / margen_anterior * 100)

        promedio_precio = np.mean(incrementos_precio)
        promedio_margen = np.mean(incrementos_margen)
        promedio_unidades = np.mean(incrementos_unidades)
        
        variaciones.append({
            'Producto': producto,
            'Variacion_precio': promedio_precio,
            'Variacion_margen': promedio_margen,
            'Variacion_unidades': promedio_unidades
        })
    variaciones_df = pd.DataFrame(variaciones)

    resumen = resumen_anual.groupby('Producto', group_keys=False).agg({
        'Unidades_vendidas': 'sum',
        'Ingreso_total': 'sum',
        'Costo_total': 'sum',
        'Precio_promedio': lambda x: int(np.round(np.mean(x))),
        'Margen_promedio': 'mean'
    }).reset_index()
    resumen = resumen.merge(variaciones_df, on='Producto', how='left')
    resumen = resumen.set_index('Producto')
    resumen = resumen.loc[data['Producto'].unique()].reset_index()

    return resumen



def graficarVentas(data, producto):
    data['Fecha'] = pd.to_datetime(data['Año'].astype(str) + "-" + data['Mes'].astype(str), format="%Y-%m")
    producto_data = data[data['Producto'] == producto].groupby('Fecha')['Unidades_vendidas'].sum().reset_index()
    plt.figure(figsize=(8, 4))
    plt.plot(producto_data['Fecha'], producto_data['Unidades_vendidas'], label=producto, linestyle='-', linewidth=2)

    z = np.polyfit(np.arange(len(producto_data)), producto_data['Unidades_vendidas'], 1)
    p = np.poly1d(z)

    plt.plot(producto_data['Fecha'], p(np.arange(len(producto_data))), "r--", label="Tendencia", linewidth=2)
    plt.ylim(bottom=0)
    plt.title(f"Evolución de Ventas Mensual - {producto}", fontsize=14)
    plt.xlabel("Año - Mes", fontsize=12)
    plt.ylabel("Unidades Vendidas", fontsize=12)
    plt.grid(alpha=0.3)
    plt.legend()
    st.pyplot(plt)



def main():
    datos = cargarDatos()

    if datos is None:
        st.markdown("### Por favor, sube un archivo CSV desde la barra lateral.")
        mostrar_informacion_alumno()
    else:
        sucursales = ["Todas"] + datos["Sucursal"].unique().tolist()
        sucursal_seleccionada = st.sidebar.selectbox("Seleccionar Sucursal", sucursales)

        if sucursal_seleccionada != "Todas":
            datos = datos[datos["Sucursal"] == sucursal_seleccionada]

        resumen = calcularMetricas(datos)

        st.title(f"Datos de {sucursal_seleccionada if sucursal_seleccionada != 'Todas' else 'todas las Sucursales'}")
        
        for _, row in resumen.iterrows():
            with st.container(border=True):
                st.subheader(f"{row['Producto']}")
                col1, col2 = st.columns([1, 3])
                with col1:
                    # Precio promedio
                    st.metric(
                        label="Precio Promedio", 
                        value=f"${int(row['Precio_promedio']):,}", 
                        delta=f"{row['Variacion_precio']:+.2f}%"
                    )
                    # Margen promedio
                    st.metric(
                        label="Margen Promedio", 
                        value=f"{np.round(row['Margen_promedio'] * 100)}%",
                        delta=f"{row['Variacion_margen']:+.2f}%"
                    )
                    # Unidades vendidas
                    st.metric(
                        label="Unidades Vendidas", 
                        value=f"{row['Unidades_vendidas']:,}",
                        delta=f"{row['Variacion_unidades']:+.2f}%"
                    )

                with col2:
                    graficarVentas(datos, row['Producto'])

if __name__ == "__main__":
    main()