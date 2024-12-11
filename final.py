import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, classification_report
from wordcloud import WordCloud
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

# Ruta del archivo de datos
ruta_archivo = r"C:\Users\DELL\Desktop\ITI 9\INTELIGENCIA DE NEGOCIOS\PROYECTO_FINAL\tiktokencuesta.xlsx"

# =============================================
# 1. Preparación inicial de los datos
# =============================================

# 1.1 Cargar datos
datos = pd.read_excel(ruta_archivo)
datos.columns = datos.columns.str.strip()  # Quitar espacios de los nombres de columnas

# Identificar columnas numéricas y textuales
numeric_cols = [
    "Cómo calificarías el funcionamiento de TikTok?",
    "¿Cuántas veces al día abres la aplicación de TikTok?",
    "¿Cuántos videos compartes diariamente desde TikTok?",
    "Con cuántos amigos te compartes TikTok?",
    "Del 1 - 10 que tanto te gusta TikTok",
]

columnas_textuales = [
    "¿Qué te motiva a usar TikTok diariamente?",
    "¿Qué tipo de contenido prefieres consumir en TikTok (educativo, entretenimiento, humor, tendencias, etc.)?",
    "¿Cómo describirías la influencia de TikTok en tu estado de ánimo diario?",
    "¿Qué cambios has notado en tu rutina diaria desde que empezaste a usar TikTok?",
    "¿Qué opinas del impacto que tiene TikTok en las relaciones sociales o personales?",
]

datos_numericos = datos[columnas_numericas]
datos_textuales = datos[columnas_textuales]

# Revisar valores nulos
if datos_numericos.isnull().sum().sum() > 0 or datos_textuales.isnull().sum().sum() > 0:
    print("Existen valores nulos en los datos. Limpiando valores nulos.")
    datos_numericos = datos_numericos.dropna()
    datos_textuales = datos_textuales.fillna("")
else:
    print("Datos cargados y verificados correctamente.")

# Convertir todos los valores textuales a cadenas
datos_textuales = datos_textuales.astype(str)

# =============================================
# 2. Análisis Exploratorio de Datos (AED)
# =============================================

# 2.1 Estadísticas descriptivas
descriptivas_numericas = datos_numericos.describe()
print("Estadísticas descriptivas:\n", descriptivas_numericas)

# 2.2 Histogramas y diagramas de dispersión
datos_numericos.hist(figsize=(8, 6), bins=10, edgecolor='black')
plt.suptitle("Distribuciones de columnas numéricas", fontsize=12)
plt.show()

sns.pairplot(datos_numericos, height=1.5, diag_kind="kde")
plt.suptitle("Diagramas de dispersión entre columnas", y=1.02, fontsize=12)
plt.show()

# 2.3 Nube de palabras
texto_combinado = " ".join(datos_textuales.values.flatten())
nube_palabras = WordCloud(width=800, height=400, background_color="white", colormap="viridis").generate(texto_combinado)

plt.figure(figsize=(8, 4))
plt.imshow(nube_palabras, interpolation="bilinear")
plt.axis("off")
plt.title("Nube de Palabras: Respuestas Textuales", fontsize=12)
plt.show()

# =============================================
# 3. Aplicación de Modelos de Agrupación
# =============================================

# Estandarizar datos numéricos
escalador = StandardScaler()
datos_escalados = escalador.fit_transform(datos_numericos)

# Método del codo
inercia = []
rango_k = range(2, 10)
for k in rango_k:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(datos_escalados)
    inercia.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(rango_k, inercia, marker='o')
plt.title("Método del Codo", fontsize=12)
plt.xlabel("Número de Clústeres (k)", fontsize=10)
plt.ylabel("Inercia", fontsize=10)
plt.grid(True)
plt.show()

# Elegir k óptimo y ajustar K-means
k_optimo = 4
kmeans = KMeans(n_clusters=k_optimo, random_state=42, n_init=10)
kmeans.fit(datos_escalados)
datos_numericos = datos_numericos.copy()
datos_numericos.loc[:, 'Cluster'] = kmeans.labels_

# Visualización de clústeres
sns.pairplot(datos_numericos, hue="Cluster", palette="Set2", height=1.8, diag_kind="kde")
plt.suptitle("Visualización de Clústeres", y=1.02, fontsize=12)
plt.show()

# Análisis de clústeres
for cluster in range(k_optimo):
    datos_cluster = datos_numericos[datos_numericos['Cluster'] == cluster]
    print(f"Características del Clúster {cluster}:\n", datos_cluster.mean())

# =============================================
# 4. Aplicación de Modelos de Clasificación
# =============================================

# Naive Bayes para clasificación
datos_numericos.loc[:, 'Satisfacción'] = datos_numericos["Del 1 - 10 que tanto te gusta TikTok"] > 5
X = datos_escalados
y = datos_numericos['Satisfacción']

# Dividir datos en entrenamiento y prueba
X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar modelo Naive Bayes
modelo_nb = GaussianNB()
modelo_nb.fit(X_entrenamiento, y_entrenamiento)

# Evaluar modelo
y_pred = modelo_nb.predict(X_prueba)
print("Reporte de Clasificación para Naive Bayes:\n", classification_report(y_prueba, y_pred))

# =============================================
# Código terminado
# =============================================
print("Análisis completado.")
