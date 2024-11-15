from fasthtml.common import *

app, rt = fast_app()

def Saludar(nombre):
    return Main(
        H4(nombre),
        id="saludo"
    )

@rt('/')
def get():
    nombres= ["Juan", "Pedro", "Lucas"]
    return Titled(
        "Agenda",
        Ol(
        *[Li(Saludar(n)) for n in nombres],
        id="saludos"
    ))

serve()

