# ------------------------ Importaciones

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

# ------------------------ Fin de las Importaciones

# 1. Descarga y prepara los datos. Explica el procedimiento.
#para la entrega en tripleten
#region0 = pd.read_csv("/datasets/geo_data_0.csv")
#region1 = pd.read_csv("/datasets/geo_data_1.csv")
#region2 = pd.read_csv("/datasets/geo_data_2.csv")

region0 = pd.read_csv(r"C:\Users\sasor\Desktop\Tripleten\Sprint 12\Proyecto\geo_data_0.csv")
region1 = pd.read_csv(r"C:\Users\sasor\Desktop\Tripleten\Sprint 12\Proyecto\geo_data_1.csv")
region2 = pd.read_csv(r"C:\Users\sasor\Desktop\Tripleten\Sprint 12\Proyecto\geo_data_2.csv")

"""se usa la libreria pandas para pooder usar read_csv y poder tener la base de cada uno de los dfs para trabajar con ellos"""

# para verificar la estructura de cada df
region0.shape
region1.shape
region2.shape

region0.info()
region1.info()
region2.info()

region0.dtypes
region1.dtypes
region2.dtypes

region0.isnull().sum()
region1.isnull().sum()
region2.isnull().sum()

"""no se enontraron inconsistencias en cada df individial"""

# 2. Entrena y prueba el modelo para cada región en geo_data_0.csv:
    # Divide los datos en un conjunto de entrenamiento y un conjunto de validación en una proporción de 75:25

"""se opta por juntar todo en uno mismo para no repetir todo 3 veces"""

regiones = [("region0", region0), ("region1", region1), ("region2", region2)]
 
for nombre, data in regiones:
    print(nombre)
    print(data.head())
    print(data.info())
    print("nulos:", data.isna().sum().sum())
    print("filas duplicadas completas:", data.duplicated().sum())
    print("ids repetidos:", data["id"].duplicated().sum())
    print()

#Entrena el modelo y haz predicciones para el conjunto de validación.
def entrenar_modelo(data):
    features = data.drop(["id", "product"], axis=1)
    #se quita ya que son mas claves de identificacion y no aporta al entrenamiento
    target = data["product"]
    features_train, features_valid, target_train, target_valid = train_test_split(features, target, test_size=0.25, random_state=12345)
    #Entrena el modelo y haz predicciones para el conjunto de validación. Restriccion: solo se puede usar regresion logistica
    modelo = LinearRegression()
    modelo.fit(features_train, target_train) 
    predicciones_valid = modelo.predict(features_valid)
    predicciones_valid = pd.Series(predicciones_valid, index=target_valid.index)
    rmse = mean_squared_error(target_valid, predicciones_valid) ** 0.5
    # Muestra el volumen medio de reservas predicho y RMSE del modelo.
    print("reservas predichas:", predicciones_valid.mean())
    print("rmse del modelo:", rmse)
    return target_valid, predicciones_valid

resultados = {}
for nombre, data in regiones:
    print(nombre)
    target_valid, predicciones_valid = entrenar_modelo(data)
    resultados[nombre] = (target_valid, predicciones_valid)
    print()
#
    # Analiza los resultados.
"""los resultados por region indican que la r1 tiene el rmse mas bajo de las 3, es el mas indicado de todos ya que se equivoca menos al cambio de volumen. las reservas preduchas nos indican que igualmente la r1 es la menor de las 3, esto quiere decir que aunque sea mas facil de predecir, tambien tiene menos promedio de reservas"""

# 3. Prepárate para el cálculo de ganancias:
    #Almacena todos los valores necesarios para los cálculos en variables separadas.
presupuesto = 100000000
pozos_a_desarrollar = 200
pozos_explorados = 500
precio_por_unidad = 4500

    #Dada la inversión de 100 millones por 200 pozos petrolíferos, de media un pozo petrolífero debe producir al menos un valor de 500,000 dólares en unidades para evitar pérdidas (esto es equivalente a 111.1 unidades). Compara esta cantidad con la cantidad media de reservas en cada región.

punto_equilibrio = presupuesto / pozos_a_desarrollar / precio_por_unidad
print("punto de equilibrio por pozo:", punto_equilibrio)
 
for nombre, data in regiones:
    print(nombre, "media real de reservas:", data["product"].mean())
#    
    #Presenta conclusiones sobre cómo preparar el paso para calcular el beneficio.

"""el PE es de 111.1 unidades por pozo (111 unidades redondeado) y la reserva media real, por cada una de las regiones, es menor al PE. Por el momento podriamos decir que tenemos la seguridad de que el modelo escogera los 200 pozos que se piden, siendo los mejores resultados aquellas predicciones mas altas. Pero hasta ahora el entorno no se ve tan positivo, ya que las reservas no llegan al punto de equilibrio"""

# 4. Escribe una función para calcular la ganancia de un conjunto de pozos de petróleo seleccionados y modela las predicciones:

def calcular_ganancia(target, predicciones, cantidad):
    predicciones_ordenadas = predicciones.sort_values(ascending=False)
    seleccionados = target[predicciones_ordenadas.index][:cantidad]
    ingresos = seleccionados.sum() * precio_por_unidad
    return ingresos - presupuesto

for nombre in resultados:
    target_valid, predicciones_valid = resultados[nombre]
    ganancia = calcular_ganancia(target_valid, predicciones_valid, pozos_a_desarrollar)
    print(nombre, "ganancia con los 200 mejores pozos:", ganancia)

"""tomando en cuenta los 200 mejores pozoso hay una ganancia positiva en cada region, sienfo la r0 la mayor de todas. esta es solo una instancia del resultado que se va a calcular ya que hacen falta mas cosas antes de decidir que hare a continuacion"""

# 5. Calcula riesgos y ganancias para cada región:

def bootstrap_ganancia(target, predicciones):
    valores = []
    state = np.random.RandomState(12345)
    for i in range(1000):
        # simulo un estudio de 500 pozos sacando una submuestra con reemplazo
        target_sub = target.sample(n=pozos_explorados, replace=True, random_state=state)
        pred_sub = predicciones[target_sub.index]
        valores.append(calcular_ganancia(target_sub, pred_sub, pozos_a_desarrollar))
 
    valores = pd.Series(valores)
    # los paso a float de python, si no quedan como np.float64 y al
    # imprimirlos dentro de una tupla sale feo tipo "np.float64(123.45)"
    media = float(valores.mean())
    inferior = float(valores.quantile(0.025))
    superior = float(valores.quantile(0.975))
    # el riesgo de perdida es simplemente el porcentaje de veces que la
    # ganancia dio negativa en las 1000 simulaciones
    riesgo = float((valores < 0).mean() * 100)
 
    return media, inferior, superior, riesgo
 
resumen = {}
for nombre in resultados:
    target_valid, predicciones_valid = resultados[nombre]
    media, inferior, superior, riesgo = bootstrap_ganancia(target_valid, predicciones_valid)
    resumen[nombre] = (media, inferior, superior, riesgo)
    print(nombre)
    print("ganancia media:", media)
    print("int. confianza  - 95%:", (inferior, superior))
    print("riesgo de perdida:", riesgo, "%")
    print()
    
"""dados los resultados de este analisis se puede comprobar que la r1 es la unica con un riesgo muy pequeño de perdida, del 1% lo que la hace la mejor opcion y la unica ya que tanto como r0 y r2 quedan descartadas; presentan un riesgo de perdida igual o mayor al 6%. Cuando se añadio el bootstrapping la r1 fue la que generaba la mayor ganancia media a comparacion de las otras y eso es otro punto a favor para dicha region. Este resultado choca con el analisis anterior donde r0 nos arrojaba una ganancia estimada mayor que cualquiera de las demas. Esto se explica desde la logica, ya que anteriormente en el paso 4 unciamente estabamos analizando una instancia, o sea, un escenario posible y particular. Con el bootstrapping estamos haciendo mas instancias y promediandolas. Se concluye que r1 es la mejor opcion entonces"""