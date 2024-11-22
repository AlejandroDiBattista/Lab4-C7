import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

"url:https://ssz6djgbnm8cxvykmdyng3.streamlit.app/"

st.set_page_config(page_title="Análisis de Ventas", layout="wide")
st.title("Análisis de Ventas")

st.sidebar.header("Cargar archivo de datos")

uploaded_file = st.sidebar.file_uploader("Sube un archivo de ventas", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    sucursal = st.sidebar.selectbox("Seleccionar Sucursal", ["Todas"] + list(df["Sucursal"].unique()))
    if sucursal != "Todas":
        df = df[df["Sucursal"] == sucursal]

    st.subheader(f"Datos de {'Todas las Sucursales' if sucursal == 'Todas' else sucursal}")

    df_grouped = df.groupby("Producto").agg({
        "Ingreso_total": "sum",
        "Costo_total": "sum",
        "Unidades_vendidas": "sum"
    }).reset_index()
    df_grouped["Precio_promedio"] = df_grouped["Ingreso_total"] / df_grouped["Unidades_vendidas"]
    df_grouped["Margen_promedio"] = (df_grouped["Ingreso_total"] - df_grouped["Costo_total"]) / df_grouped["Ingreso_total"]

    df_grouped["Variación_precio"] = (df_grouped["Precio_promedio"] / df_grouped["Precio_promedio"].shift(1) - 1) * 100
    df_grouped["Variación_unidades"] = (df_grouped["Unidades_vendidas"] / df_grouped["Unidades_vendidas"].shift(1) - 1) * 100

    st.sidebar.subheader("Personalización del gráfico")
    color_line = st.sidebar.color_picker("Elige un color para la línea", "#1f77b4")

    for i, row in df_grouped.iterrows():
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.markdown(f"### {row['Producto']}")
            st.metric("Precio Promedio", f"${row['Precio_promedio']:.2f}", f"{row['Variación_precio']:.2f}%")
            st.metric("Margen Promedio", f"{row['Margen_promedio']:.2%}")
            st.metric("Unidades Vendidas", f"{row['Unidades_vendidas']:,}", f"{row['Variación_unidades']:.2f}%")

        with col2:
            product_data = df[df["Producto"] == row["Producto"]]
            product_data["Periodo"] = product_data["Año"].astype(str) + "-" + product_data["Mes"].astype(str).str.zfill(2)
            ventas_por_periodo = product_data.groupby("Periodo")["Unidades_vendidas"].sum().reset_index()

            ventas_por_periodo["Año"] = ventas_por_periodo["Periodo"].str[:4]

            plt.figure(figsize=(10, 6))
            # Línea principal
            plt.plot(
                ventas_por_periodo["Año"],
                ventas_por_periodo["Unidades_vendidas"],
                color=color_line,
                marker="o",
                label=row["Producto"],
                linestyle="-"
            )
            
            # Línea de tendencia
            if len(ventas_por_periodo) > 1:
                z = np.polyfit(range(len(ventas_por_periodo)), ventas_por_periodo["Unidades_vendidas"], 1)
                p = np.poly1d(z)
                plt.plot(
                    ventas_por_periodo["Año"],
                    p(range(len(ventas_por_periodo))),
                    linestyle="--",
                    color="gray",
                    label="Tendencia"
                )

            plt.grid(True, linestyle="--", alpha=0.6)
            plt.xticks(rotation=0)
            plt.title(f"Evolución de Ventas - {row['Producto']}")
            plt.xlabel("Año")
            plt.ylabel("Unidades vendidas")
            plt.legend()
            st.pyplot(plt)

        st.markdown("---")

else:
    st.info("Sube un archivo CSV para comenzar el análisis.")
