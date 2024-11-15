from fasthtml.common import *

app, rt, contactos, Contacto = fast_app('agenda.db',
                   id=int, nombre=str, apellido=str, telefono=str,
                   pk='id')

if contactos.count == 0:
    contactos.insert(Contacto(id=1,nombre="Juan",  apellido="Perez",  telefono="123456"))
    contactos.insert(Contacto(id=2,nombre="Pedro", apellido="Gomez",  telefono="654321"))
    contactos.insert(Contacto(id=3,nombre="Lucas", apellido="Garcia", telefono="456789"))

def MostrarContacto(contacto):
    return Li(
        H5(f"{contacto.nombre} {contacto.apellido}"),
        P(contacto.telefono),
        A("Borrar", 
          hx_delete=f"/borrar/{contacto.id}",
          hx_target=f"#contacto-{contacto.id}"),
        id=f"contacto-{contacto.id}"
    )
    
def ListaContactos():
    return Ul(
        *[MostrarContacto(c) for c in contactos()],
        id="contactos"
    )
@rt('/')
def get():
    return Titled(
        "Mi Agenda",
        Button("Agregar", 
               hx_get="/agregar", 
               hx_target="#contactos",
               hx_swap="outerHTML",
               id="agregar"),
            ListaContactos()
        )

@rt('/agregar')
def get():
    return Form(
            Input(placeholder="Nombre",   name="nombre"),
            Input(placeholder="Apellido", name="apellido"),
            Input(placeholder="Telefono", name="telefono"),
            Button("Guardar"),
            action="/agregar"

        )
@rt('/agregar')
def post(nuevo:Contacto):
    print(nuevo)
    contactos.insert(nuevo)
    return ListaContactos()

@rt('/borrar/{id}')
def delete(id:int):
    contactos.delete(id)

serve()