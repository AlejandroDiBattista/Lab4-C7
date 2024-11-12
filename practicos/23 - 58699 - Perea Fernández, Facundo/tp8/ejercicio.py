import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## ATENCION: Debe colocar la direccion en la que ha sido publicada la aplicacion en la siguiente linea\
# url = 'https://tp8-555555.streamlit.app/'

def mostrar_informacion_alumno():
    with st.container(border=True):
        st.markdown('**Legajo:** 58.699')
        st.markdown('**Nombre:** Facundo Perea Fernandez')
        st.markdown('**Comisión:** C7')

mostrar_informacion_alumno()