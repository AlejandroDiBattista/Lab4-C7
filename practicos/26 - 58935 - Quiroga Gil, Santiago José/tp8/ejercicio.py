import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# url = 'https://tp8-58935.streamlit.app'
st.set_page_config(
    page_title="2do Parcial",
    page_icon="📊",
    layout="wide",
    menu_items={
        "About":":fire: Kiro :fire:"      
        }
)


def mostrar_informacion_alumno():
    
    with st.container(border=True):
        cols = st.columns([1, 2])
        with cols[0]:
            st.write("", width=100)
        with cols[1]:
            st.markdown("""
            ### Información
            - **Legajo:** 58935
            - **Nombre:** Quiroga Gil Santiago José
            - **Comisión:** 7
            """)

def grafica_stonks(datos_prod, nom_prod):
  
    ventas_mes = datos_prod.groupby(['Año', 'Mes'])['Unidades_vendidas'].sum().reset_index()

    fig, eje = plt.subplots(figsize=(12, 6))

    x = range(len(ventas_mes))
    eje.plot(x, ventas_mes['Unidades_vendidas'], 
            marker='o', linewidth=2, markersize=4, 
            label=nom_prod, color='blue')

    z = np.polyfit(x, ventas_mes['Unidades_vendidas'], 1)
    p = np.poly1d(z)
    eje.plot(x, p(x), linestyle='--', color='red', 
            label='Tendencia', linewidth=1.5)
    
    
    eje.set_title('Evolución de Ventas Mensual', pad=20, fontsize=14)
    eje.set_xlabel('Período', fontsize=12)
    eje.set_ylabel('Unidades Vendidas', fontsize=12)
    
    eje.set_xticks(x)
    labels = [f"{row.Año}" if row.Mes == 1 else "" 
              for row in ventas_mes.itertuples()]
    eje.set_xticklabels(labels, rotation=45)
    

    eje.set_ylim(0, None)
    eje.legend(title='Producto', bbox_to_anchor=(1.05, 1), loc='upper left')
    eje.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    return fig


def calcular_prod(datos_prod):

    datos_prod['Precio_promedio'] = datos_prod['Ingreso_total'] / datos_prod['Unidades_vendidas']
    precio_prom = datos_prod['Precio_promedio'].mean()
    precio_anual = datos_prod.groupby('Año')['Precio_promedio'].mean()
    var_precio = precio_anual.pct_change().mean() * 100

    datos_prod['Ganancia'] = datos_prod['Ingreso_total'] - datos_prod['Costo_total']
    datos_prod['Margen'] = (datos_prod['Ganancia'] / datos_prod['Ingreso_total']) * 100
    margen_prom = datos_prod['Margen'].mean()
    margen_anual = datos_prod.groupby('Año')['Margen'].mean()
    var_margen = margen_anual.pct_change().mean() * 100

    u_total = datos_prod['Unidades_vendidas'].sum()
    u_anual = datos_prod.groupby('Año')['Unidades_vendidas'].sum()
    var_u = u_anual.pct_change().mean() * 100
    
    return (precio_prom, var_precio, margen_prom, var_margen, 
            u_total, var_u)

def main():

  
    st.sidebar.title("Configuración")
    
    archivo = st.sidebar.file_uploader(
        "Cargar archivo CSV",
        type=["csv"],
        help="Seleccione un archivo CSV con los datos de ventas"
    )    
    if archivo:
        datos_csv = pd.read_csv(archivo) 
        opciones = ["Todas"] + (datos_csv['Sucursal'].unique().tolist())
        opcion_selec = st.sidebar.selectbox(
            "Seleccionar Sucursal",
            opciones,
            help="Filtre los datos por sucursal"
        )
        
        if opcion_selec != "Todas":
            datos_csv = datos_csv[datos_csv['Sucursal'] == opcion_selec]
            st.title(f"📊  Ventas - {opcion_selec}")
        else:
            st.title("📊  Ventas - Todas las Sucursales")
            
   
        for producto in (datos_csv['Producto'].unique()):
            with st.container(border=True):
                st.subheader(f"🏷️ {producto}")
                
                datos_prod = datos_csv[datos_csv['Producto'] == producto]
                result = calcular_prod(datos_prod)
           
                col1, col2 = st.columns([0.25, 0.75])
                
                with col1:
                    st.metric(
                        "💰 Precio Promedio",
                        f"${result[0]:,.0f}",
                        f"{result[1]:.2f}%"
                    )
                    st.metric(
                        "📈 Margen Promedio",
                        f"{result[2]:.0f}%",
                        f"{result[3]:.2f}%"
                    )
                    st.metric( 
                        "📦 Unidades Vendidas",
                        f"{result[4]:,.0f}",
                        f"{result[5]:.2f}%"
                        
                    )                
                with col2:
                    fig = grafica_stonks(datos_prod, producto)
                    st.pyplot(fig)
    else:
        st.info("TRABAJO PRACTICO 8 y 2do PARCIAL LABORATORIO DE COMPUTACION IV")
        mostrar_informacion_alumno()

main()