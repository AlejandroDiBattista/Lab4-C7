import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# URL de la aplicación
url = 'https://parcialjulieta8.streamlit.app/'

def mostrar_informacion_alumno():
    with st.container():
        st.markdown("**Legajo:** 55.606")
        st.markdown("**Nombre:** Julieta Meza")
        st.markdown("**Comisión:** C7")

def mostrarAnalisisVentas():
    
    st.sidebar.header("¡Cargar todos los archivos csv!")
    
    uploaded_file = st.sidebar.file_uploader("Subir archivos", type="csv")

    if uploaded_file is not None:
        
        df = pd.read_csv(uploaded_file)
        
        required_columns = ["Sucursal", "Producto", "Año", "Mes", "Unidades_vendidas", "Ingreso_total", "Costo_total"]

        if all(column in df.columns for column in required_columns):
            
            df['Precio_promedio'] = df['Ingreso_total'] / df['Unidades_vendidas']
            
            df['Margen_promedio'] = (df['Ingreso_total'] - df['Costo_total']) / df['Ingreso_total']
            
            resumen = df.groupby('Producto').agg({
                
                'Precio_promedio': 'mean',
                
                'Margen_promedio': 'mean',
                
                'Unidades_vendidas': 'sum'
                
            }).reset_index()

            
            sucursales = ['Todas'] + df['Sucursal'].unique().tolist()
            
            sucursalSeleccionada = st.sidebar.selectbox("Selecciona alguna sucursal", sucursales)

            if sucursalSeleccionada != 'Todas':
                df = df[df['Sucursal'] == sucursalSeleccionada]

            st.title("Análisis de Ventas de Gaseosas y Vinos :)")
            st.header(f"¡Datos de {'Todas las Sucursales disponibles!' if sucursalSeleccionada == 'Todas' else sucursalSeleccionada}")

            for _, row in resumen.iterrows():
                
                producto = row['Producto']
                
                precio_promedio = row['Precio_promedio']
                
                margen_promedio = row['Margen_promedio']
                
                unidades_vendidas = row['Unidades_vendidas']

                columnaPrincipal, columna2 = st.columns([1, 3])
                with columnaPrincipal:
                    st.markdown(f"### {producto}")
                    st.metric("Precios completos", f"${precio_promedio:,.2f}")
                    
                    st.metric("Margen Completo", f"{margen_promedio * 100:.2f}%")
                    
                    st.metric("Unidades Vendidas", f"{int(unidades_vendidas):,}")
                
                with columna2:
                    df_producto = df[df['Producto'] == producto]
                    
                    df_producto['Fecha'] = pd.to_datetime(df_producto['Año'].astype(str) + '-' + df_producto['Mes'].astype(str) + '-01')
                    
                    ventasMensuales = df_producto.groupby('Fecha')['Unidades_vendidas'].sum().reset_index()
                    
                    plt.figure(figsize=(6, 4))
                    
                    plt.plot(ventasMensuales['Fecha'], ventasMensuales['Unidades_vendidas'], marker="o", label=producto)
                    
                    z = np.polyfit(ventasMensuales.index, ventasMensuales['Unidades_vendidas'], 1)
                    
                    polinomio = np.poly1d(z)
                    
                    plt.plot(ventasMensuales['Fecha'], polinomio(ventasMensuales.index), "r--", label="Tendencia")
                    
                    plt.xlabel("Año/Mes")
                    
                    plt.ylabel("¡Unidades Totalemnte vendidas!")
                    
                    plt.title("¡Evolución de todas las ventas mensuales!")
                    
                    plt.legend()
                    
                    st.pyplot(plt)
        else:
            st.error("El archivo CSV no esta correctamente subido(sube un archivo csv)")
    else:
        st.write("¡¡Cargar un archivo CSV!!")
        

mostrar_informacion_alumno()

mostrarAnalisisVentas()
