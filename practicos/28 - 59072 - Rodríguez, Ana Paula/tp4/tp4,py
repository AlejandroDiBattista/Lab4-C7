def obtener_puntaje(valor_numero, tipo_palo):
    tabla_puntajes = {
        (1, 'espada'): 14,
        (1, 'basto'): 13,
        (7, 'oro'): 12,
        (7, 'copa'): 11,
        3: 10,
        2: 9,
        1: 8,
        12: 7,
        11: 6,
        10: 5,
        7: 4,
        6: 3,
        5: 2,
        4: 1
    }
    return tabla_puntajes.get((valor_numero, tipo_palo), tabla_puntajes.get(valor_numero, 0))

palo_descripcion = {'oro': 'oro', 'copa': 'copa', 'espada': 'espada', 'basto': 'basto'}


class Naipes:
    def __init__(self, valor_numero, tipo_palo):
        self.valor_numero = valor_numero
        self.tipo_palo = tipo_palo

    @property
    def puntaje(self):
        return obtener_puntaje(self.valor_numero, self.tipo_palo)

    def __str__(self):
        return f"{self.valor_numero:2}{palo_descripcion[self.tipo_palo]}"

    def __repr__(self):
        return self.__str__()

    def __gt__(self, otro_naipe):
        return self.puntaje > otro_naipe.puntaje


carta_prueba = Naipes(1, "oro")
print(carta_prueba.valor_numero)  # 1
print(carta_prueba.tipo_palo)  # oro
print(carta_prueba.puntaje)  # 8
print(str(carta_prueba))  # 1 de oro
print(repr(carta_prueba))
naipe_a = Naipes(7, "copa")
naipe_b = Naipes(7, "oro")
print(naipe_a, naipe_b)  # False

import random

lista_numeros = [1, 2, 3, 4, 5, 6, 7, 10, 11, 12]
lista_palos = ['oro', 'copa', 'espada', 'basto']


class Baraja:
    def __init__(self):
        self.naipe_baraja = [Naipes(valor_numero, tipo_palo) for valor_numero in lista_numeros for tipo_palo in lista_palos]
        self.barajar()

    def barajar(self):
        random.shuffle(self.naipe_baraja)

    def repartir(self, cantidad_naipe):
        return [self.naipe_baraja.pop() for _ in range(cantidad_naipe)]


baraja_prueba = Baraja()
baraja_prueba.barajar()
print(baraja_prueba.naipe_baraja)
print(baraja_prueba.repartir(3))


class Participante:
    def __init__(self, alias):
        self.alias = alias
        self.naipes_participante = []
        self.puntos_totales = 0

    def tomar_naipes(self, naipes_recibidos):
        self.naipes_participante = naipes_recibidos

    def jugar_naipes(self):
        random.shuffle(self.naipes_participante)
        return self.naipes_participante.pop()

    def __str__(self):
        return self.alias

    def __repr__(self):
        return self.__str__()


participante_prueba = Participante("Juan")
participante_prueba.tomar_naipes(baraja_prueba.repartir(3))

print(participante_prueba.naipes_participante)
print(participante_prueba.jugar_naipes())
print(participante_prueba.naipes_participante)


class Juego:
    def __init__(self, jugador_uno, jugador_dos):
        self.lista_jugadores = [jugador_uno, jugador_dos]
        self.baraja_juego = Baraja()

    def ronda_juego(self):
        return [[jugador.jugar_naipes() for jugador in self.lista_jugadores] for _ in range(3)]

    def verificar_ganador_ronda(self, lista_jugadas):
        resultado_ronda = [sum(1 for carta1, carta2 in lista_jugadas if carta1 > carta2),
                           sum(1 for carta1, carta2 in lista_jugadas if carta2 > carta1)]

        for puntos_ronda, jugador in zip(resultado_ronda, self.lista_jugadores):
            if puntos_ronda >= 2:
                jugador.puntos_totales += 1
                print(f"{jugador} gana la ronda")

    def mostrar_puntajes(self):
        print(f"\nPuntajes:")
        for jugador in self.lista_jugadores:
            print(f" - {jugador}: {jugador.puntos_totales}")
        print('-' * 20)

    def iniciar_juego(self):
        contador_ronda = 0
        while all(jugador.puntos_totales < 15 for jugador in self.lista_jugadores):
            self.baraja_juego = Baraja()
            contador_ronda += 1
            print(f"Ronda {contador_ronda} {[jugador for jugador in self.lista_jugadores]}")

            for jugador in self.lista_jugadores:
                jugador.tomar_naipes(self.baraja_juego.repartir(3))

            jugadas_ronda = self.ronda_juego()
            for jugada in jugadas_ronda:
                print(' ', jugada)
            self.verificar_ganador_ronda(jugadas_ronda)

            self.mostrar_puntajes()

        for jugador in self.lista_jugadores:
            if jugador.puntos_totales >= 15:
                print(f"{jugador} gana la partida")


partida_prueba = Juego(Participante("Mauricio"), Participante("Lujan"))
partida_prueba.iniciar_juego()
