import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

st.markdown('''
            ## Demo Streamlit
            Esta es una demostración de Streamlit
            * Editar parámetros
            * Graficar $\sin(x)$ y $\cos(x)$
            * Guardar en sesión
            ''')

with st.sidebar:
    c1, c2 = st.columns(2)
    a = c1.slider('Rango de X',-10,10,(-5,5))
    b = c2.number_input('Cantidad de puntos',0.0,50.0,20.0, step=5.0)

    x = np.linspace(*a, int(b))
    ys = np.sin(x)
    yc = np.cos(x)
    plt.plot(x, ys, label='y = sin(x)')
    plt.plot(x, yc, label='y = cos(x)')
    plt.grid()
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.axhline(0, color='black', linewidth=2)
    plt.axvline(0, color='black', linewidth=2)
    plt.legend()
    st.sidebar.pyplot(plt)
    st.header(f"a: {a}, b: {b}")

c = st.checkbox('Marcame!')
if c:
    st.success('Estoy marcado!')
else:
    st.warning('No estoy marcado!')

if 'contador' not in st.session_state:
    st.session_state['contador'] = 0

if st.button('Incrementar!'):
    c = st.session_state['contador']
    st.session_state['contador'] = c + 1


st.header(st.session_state['contador'])