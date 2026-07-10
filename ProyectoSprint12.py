# ------------------------ Importaciones

import pandas as pd

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

