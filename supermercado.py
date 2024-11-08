import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import deque
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches

class SimulacionSupermercado:
    def __init__(self, lambda_rate, mu_rate, num_cajas, simulation_time):
        self.lambda_rate = lambda_rate / 60
        self.mu_rate = mu_rate / 60
        self.num_cajas = num_cajas
        self.simulation_time = simulation_time
        
    def simular(self):
        # Generar llegadas de clientes
        num_clientes_esperados = int(self.lambda_rate * self.simulation_time * 1.5)
        llegadas = np.sort(np.random.exponential(1/self.lambda_rate, num_clientes_esperados))
        llegadas = llegadas[llegadas < self.simulation_time]
        
        # Inicializar variables
        tiempos_cola = []
        tiempos_sistema = []
        longitudes_cola = []
        tiempos_servicio = []
        utilizacion_cajas = []
        cajas_ocupadas = np.zeros(self.num_cajas)
        cola = deque()
        
        # Guardar estados para animación
        self.estados_cajas = []
        self.tiempos = []
        self.longitudes_cola_tiempo = []
        tiempo_actual = 0
        
        # Procesar cada llegada
        for tiempo_llegada in llegadas:
            # Guardar estado actual para animación
            self.estados_cajas.append(cajas_ocupadas.copy())
            self.tiempos.append(tiempo_llegada)
            self.longitudes_cola_tiempo.append(len(cola))
            
            # Liberar cajas que ya terminaron
            cajas_disponibles = cajas_ocupadas <= tiempo_llegada
            cajas_ocupadas[cajas_disponibles] = tiempo_llegada
            
            # Calcular utilización actual
            utilizacion_cajas.append(np.mean(cajas_ocupadas > tiempo_llegada))
            
            # Atender clientes en cola si hay cajas disponibles
            while cola and np.any(cajas_disponibles):
                cliente_cola = cola.popleft()
                caja = np.argmin(cajas_ocupadas)
                tiempo_espera = tiempo_llegada - cliente_cola
                tiempo_servicio = min(max(np.random.exponential(1/self.mu_rate), 1), 5)
                
                tiempos_cola.append(tiempo_espera)
                tiempos_sistema.append(tiempo_espera + tiempo_servicio)
                tiempos_servicio.append(tiempo_servicio)
                cajas_ocupadas[caja] = tiempo_llegada + tiempo_servicio
                cajas_disponibles = cajas_ocupadas <= tiempo_llegada
            
            # Atender cliente actual si hay caja disponible
            if np.any(cajas_ocupadas <= tiempo_llegada):
                caja = np.argmin(cajas_ocupadas)
                tiempo_servicio = min(max(np.random.exponential(1/self.mu_rate), 1), 5)
                tiempos_cola.append(0)
                tiempos_sistema.append(tiempo_servicio)
                tiempos_servicio.append(tiempo_servicio)
                cajas_ocupadas[caja] = tiempo_llegada + tiempo_servicio
            else:
                cola.append(tiempo_llegada)
            
            longitudes_cola.append(len(cola))
        
        # Guardar último estado
        self.estados_cajas.append(cajas_ocupadas.copy())
        self.tiempos.append(tiempo_llegada)
        self.longitudes_cola_tiempo.append(len(cola))
        
        return (np.array(tiempos_cola), np.array(tiempos_sistema), 
                np.array(longitudes_cola), np.array(tiempos_servicio),
                np.array(utilizacion_cajas))

    def animar_cajas(self):
        # Crear figura con dos subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[3, 1])
        fig.suptitle('Simulación del Supermercado', fontsize=14)
        
        # Inicializar barras para las cajas
        bar_colors = ['lightgrey'] * self.num_cajas
        bars = ax1.bar(range(self.num_cajas), [0] * self.num_cajas, color=bar_colors)
        
        # Ajustar el límite superior del eje y del gráfico de cajas
        max_tiempo_restante = np.max(self.estados_cajas)  # Tomar el valor máximo de tiempos restantes de las cajas
        ax1.set_ylim(0, max(10, max_tiempo_restante))  # Limitar el eje y a 10 si es menor que max_tiempo_restante
        ax1.set_title('Estado de las Cajas')
        ax1.set_xlabel('Número de Caja')
        ax1.set_ylabel('Tiempo Restante de Atención (minutos)')
        
        # Inicializar línea para la cola
        line, = ax2.plot([], [], 'b-', label='Longitud de la Cola')
        ax2.set_xlim(0, min(self.simulation_time, np.max(self.longitudes_cola_tiempo) + 1))  # Ajustar el límite superior del eje x
        ax2.set_ylim(0, max(self.longitudes_cola_tiempo) + 1)
        ax2.set_title('Longitud de la Cola')
        ax2.set_xlabel('Tiempo (minutos)')
        ax2.set_ylabel('Personas en Cola')
        ax2.grid(True)
        
        tiempo_texto = ax1.text(0.02, 0.95, '', transform=ax1.transAxes)
        
        def init():
            line.set_data([], [])
            return list(bars) + [line, tiempo_texto]
        
        def update(frame):
            tiempo_actual = self.tiempos[frame]
            estado_actual = self.estados_cajas[frame]
            
            # Actualizar barras
            for i, bar in enumerate(bars):
                tiempo_restante = max(0, estado_actual[i] - tiempo_actual)
                bar.set_height(tiempo_restante)
                bar.set_color('red' if tiempo_restante > 0 else 'green')
            
            # Actualizar línea de cola
            line.set_data(self.tiempos[:frame+1], self.longitudes_cola_tiempo[:frame+1])
            
            tiempo_texto.set_text(f'Tiempo: {tiempo_actual:.1f} min')
            return list(bars) + [line, tiempo_texto]
        
        anim = FuncAnimation(
            fig, 
            update, 
            init_func=init,
            frames=len(self.tiempos),
            interval=100,
            repeat=True
        )
        
        plt.tight_layout()
        plt.show()
        
    def mostrar_resultados(self, tiempos_cola, tiempos_sistema, longitudes_cola, tiempos_servicio, utilizacion_cajas):
        # Calcular estadísticas
        print("\nEstadísticas de la simulación:")
        print(f"Tiempo promedio en cola: {np.mean(tiempos_cola):.2f} minutos")
        print(f"Tiempo promedio de servicio: {np.mean(tiempos_servicio):.2f} minutos")
        print(f"Tiempo promedio en sistema: {np.mean(tiempos_sistema):.2f} minutos")
        print(f"Longitud promedio de cola: {np.mean(longitudes_cola):.2f}")
        print(f"Máxima longitud de cola: {np.max(longitudes_cola)}")
        print(f"Utilización promedio de cajas: {np.mean(utilizacion_cajas):.2%}")
        
        # Visualizaciones
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Tiempos de espera
        sns.histplot(data=tiempos_cola, bins=30, kde=True, ax=axes[0,0])
        axes[0,0].set_title('Distribución de tiempos de espera en cola')
        axes[0,0].set_xlabel('Minutos')
        
        # Tiempos en sistema
        sns.histplot(data=tiempos_sistema, bins=30, kde=True, ax=axes[0,1])
        axes[0,1].set_title('Distribución de tiempos en sistema')
        axes[0,1].set_xlabel('Minutos')
        
        # Evolución de la cola
        axes[1,0].plot(longitudes_cola)
        axes[1,0].set_title('Evolución de la longitud de la cola')
        axes[1,0].set_xlabel('Número de cliente')
        axes[1,0].set_ylabel('Clientes en cola')
        
        # Utilización de cajas
        axes[1,1].plot(utilizacion_cajas)
        axes[1,1].set_title('Utilización de cajas a lo largo del tiempo')
        axes[1,1].set_xlabel('Número de cliente')
        axes[1,1].set_ylabel('Proporción de cajas ocupadas')
        
        plt.tight_layout()
        plt.show()

# Parámetros de la simulación
lambda_rate = 20 # Tasa de llegada de clientes por hora
mu_rate = 20 # Tasa de servicio de cajas por hora
num_cajas = 4 # Número de cajas en el supermercado
tiempo_sim = 60 # Tiempo de simulación en minutos
np.random.seed(42) # Semilla para reproducibilidad

# Crear simulación y ejecutar
sim = SimulacionSupermercado(lambda_rate, mu_rate, num_cajas, tiempo_sim)
resultados = sim.simular()
#sim.mostrar_resultados(*resultados)
sim.animar_cajas()
