import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## ATENCION: Debe colocar la direccion en la que ha sido publicada la aplicacion en la siguiente linea\
# url = 'https://tp8-58769.streamlit.app/'

st.set_page_config(page_title="Análisis de Ventas", layout="wide")

def mostrar_informacion_alumno():
    st.markdown(
        """
        <div style='
            border: 1px solid #ccc;
            border-radius: 5px;
            padding: 15px;
            background-color: #f9f9f9;
            margin: 10px auto;
            width: 50%;
        '>
            <strong>Legajo:</strong> 58.769<br>
            <strong>Nombre:</strong> Máximo Callejas<br>
            <strong>Comisión:</strong> C7
        </div>
        """,
        unsafe_allow_html=True,
    )



# Cargar archivo csv
def cargarDatos():
    uploaded_file = st.sidebar.file_uploader("**Subir archivo CSV**", type=["csv"])
    if uploaded_file:
        try:
            data = pd.read_csv(uploaded_file)
            st.sidebar.success("Datos cargados correctamente")
            return data
        except Exception as e:
            st.sidebar.error(f"Error al cargar los datos: {e}")
    return None



def calcularMetricas(data):
    data['Fecha'] = data['Año'].astype(str) + "-" + data['Mes'].astype(str)
    data['Fecha'] = pd.to_datetime(data['Fecha'], format="%Y-%m")
    data['Precio_promedio'] = data['Ingreso_total'] / data['Unidades_vendidas']
    data['Margen_promedio'] = (data['Ingreso_total'] - data['Costo_total']) / data['Ingreso_total']
    data['Año'] = data['Fecha'].dt.year
    resumen_anual = data.groupby(['Producto', 'Año']).agg({
        'Unidades_vendidas': 'sum',
        'Ingreso_total': 'sum',
        'Costo_total': 'sum',
        'Precio_promedio': 'mean',
        'Margen_promedio': 'mean'
    }).reset_index()
    
    variaciones = []
    for producto in data['Producto'].unique():
        producto_data = resumen_anual[resumen_anual['Producto'] == producto]
        producto_data = producto_data.sort_values('Año')

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
    resumen = resumen_anual.groupby('Producto').agg({
        'Unidades_vendidas': 'sum',
        'Ingreso_total': 'sum',
        'Costo_total': 'sum',
        'Precio_promedio': 'mean',
        'Margen_promedio': 'mean'
    }).reset_index()
    resumen = resumen.merge(variaciones_df, on='Producto', how='left')
    
    return resumen



def graficarVentas(data, producto):
    data['Fecha'] = data['Año'].astype(str) + "-" + data['Mes'].astype(str)
    data['Fecha'] = pd.to_datetime(data['Fecha'], format="%Y-%m")
    producto_data = data[data['Producto'] == producto].groupby('Fecha')['Unidades_vendidas'].sum().reset_index()

    plt.figure(figsize=(10, 6))
    plt.plot(producto_data['Fecha'], producto_data['Unidades_vendidas'], label=producto)
    z = np.polyfit(producto_data.index, producto_data['Unidades_vendidas'], 1)
    p = np.poly1d(z)
    plt.plot(producto_data['Fecha'], p(producto_data.index), "r--", label='Tendencia')
    plt.title(f'Evolución de Ventas Mensual - {producto}')
    plt.xlabel('Año-Mes')
    plt.ylabel('Unidades Vendidas')
    plt.legend()
    plt.ylim(bottom=0)
    
    st.pyplot(plt)



def mostrarMetrica(titulo, valor_principal, variacion, es_porcentaje=False):
    flecha = "▲" if variacion > 0 else "▼"
    color = "green" if variacion > 0 else "red"
    variacion_texto = f"<span style='color:{color}; font-size:14px;'>{flecha} {variacion:+.2f}%</span>"
    unidad = "%" if es_porcentaje else ""
    
    return (
        f"<div style='text-align:center;'>"
        f"<span style='font-size:16px;'>{titulo}</span><br>"
        f"<span style='font-size:24px; font-weight:bold;'>{valor_principal}{unidad}</span><br>"
        f"{variacion_texto}"
        f"</div>"
    )



def main():
    uploaded_file = st.sidebar.file_uploader("**Subir archivo CSV**", type=["csv"])
    if not uploaded_file:
        st.markdown(
            "<div style='text-align:center; color:gray; font-size:25px; margin-top:10px;'>"
            "Por favor, suba un archivo .csv desde la barra lateral"
            "</div>",
            unsafe_allow_html=True,
        )
        mostrar_informacion_alumno()
    
    datos = None
    if uploaded_file:
        try:
            datos = pd.read_csv(uploaded_file)
            st.sidebar.success("Datos cargados correctamente")
        except Exception as e:
            st.sidebar.error(f"Error al cargar los datos: {e}")
    
    if datos is not None:
        sucursales = ['Todas'] + datos['Sucursal'].unique().tolist()
        sucursal_seleccionada = st.sidebar.selectbox("Seleccionar Sucursal", sucursales)
        
        if sucursal_seleccionada != 'Todas':
            datos = datos[datos['Sucursal'] == sucursal_seleccionada]

        resumen = calcularMetricas(datos)

        for _, row in resumen.iterrows():
            with st.container():
                st.markdown(
                    f"""
                    <div style='border: 1px solid black; padding: 20px; margin: 10px 0; border-radius: 5px;'>
                        <h3 style="text-align:center;">{row['Producto']} - {sucursal_seleccionada if sucursal_seleccionada != 'Todas' else 'Todas las Sucursales'}</h3>
                    """,
                    unsafe_allow_html=True
                )
                
                col1, col2 = st.columns([1, 3])
                
                with col1:
                    #Precio Promedio
                    st.markdown(
                        mostrarMetrica(
                            "Precio Promedio",
                            f"${row['Precio_promedio']:.2f}",
                            row['Variacion_precio']
                        ),
                        unsafe_allow_html=True
                    )
                    
                    #Margen Promedio
                    st.markdown(
                        mostrarMetrica(
                            "Margen Promedio",
                            f"{row['Margen_promedio']*100:.2f}",
                            row['Variacion_margen'],
                            es_porcentaje=True
                        ),
                        unsafe_allow_html=True
                    )
                    
                    #Unidades Vendidas
                    st.markdown(
                        mostrarMetrica(
                            "Unidades Vendidas",
                            f"{int(row['Unidades_vendidas']):,}",
                            row['Variacion_unidades']
                        ),
                        unsafe_allow_html=True
                    )
                
                with col2:
                    #Gráfico
                    graficarVentas(datos, row['Producto'])

                st.markdown("</div>", unsafe_allow_html=True)
                
if __name__ == "__main__":
    main()