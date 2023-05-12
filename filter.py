import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Creamos un DataFrame con los datos
df = pd.DataFrame({'Grupo': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'A', 'B', 'A'],
                   'Columna': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                   'Valor': [20, 30, 25, 35, 45, 40, 30, 50, 55, 60]})

# Creamos una matriz de características utilizando la columna 'Valor'
X = df[['Valor']].values

# Creamos un objeto KMeans y lo ajustamos a los datos
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)

# Añadimos una nueva columna al DataFrame con las etiquetas de cluster
df['Cluster'] = kmeans.labels_

# Agrupamos los registros por 'Grupo' y 'Cluster' y obtenemos la suma de 'Valor'
grouped = df.groupby(['Grupo', 'Cluster'])['Valor'].sum().reset_index()

# Creamos una figura y un conjunto de ejes
fig, ax = plt.subplots()

# Creamos un gráfico de barras para cada grupo y cluster
for name, group in grouped.groupby('Grupo'):
    ax.bar(group['Cluster'], group['Valor'], label=name)

# Añadimos etiquetas y título al gráfico
ax.set_xlabel('Cluster')
ax.set_xticks([0, 1])
ax.set_xticklabels(['Cluster 0', 'Cluster 1'])
ax.set_ylabel('Valor')
ax.set_title('Gráfico de barras por grupo y cluster')

# Añadimos una leyenda
ax.legend()

# Mostramos el gráfico
plt.show()
