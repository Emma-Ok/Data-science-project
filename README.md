# ⚽ Análisis Exploratorio de Datos - UEFA Champions League 2025

<div align="center">

![UEFA Champions League](https://img.shields.io/badge/UEFA-Champions%20League-0066CC?style=for-the-badge&logo=uefa&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge)

**Análisis Multitécnico de Rendimiento de Jugadores de Fútbol de Élite**

[🔍 Ver Notebook](#) | [📊 Datos](#datos) | [📈 Resultados](#resultados) | [🎯 Conclusiones](CONCLUSIONES_ANALISIS_UEFA_CL_2025.md)

</div>

---

## 📋 Tabla de Contenidos

- [Descripción del Proyecto](#-descripción-del-proyecto)
- [Datos](#-datos)
- [Metodología](#-metodología)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Técnicas Aplicadas](#-técnicas-aplicadas)
- [Visualizaciones](#-visualizaciones)
- [Conclusiones](#-conclusiones)
- [Autor](#-autor)


---

## 🎯 Descripción del Proyecto

Este proyecto realiza un **análisis exploratorio exhaustivo** de los datos de rendimiento de jugadores participantes en la **UEFA Champions League 2025** (primeros 4 partidos de la fase de grupos). Combina técnicas estadísticas clásicas con algoritmos de machine learning para:

✅ Caracterizar el rendimiento de jugadores de fútbol de élite  
✅ Identificar patrones tácticos y físicos por posición  
✅ Detectar valores atípicos mediante 4 técnicas complementarias  
✅ Proporcionar insights para toma de decisiones deportivas  

### 🎓 Contexto Académico

**Asignatura:** Introducción a la Ciencia de Datos  
**Nivel:** Pregrad  
**Objetivos Académicos:**
1. Construcción y documentación de base de datos deportiva
2. Análisis exploratorio de datos (EDA) con visualizaciones avanzadas
3. Aplicación de múltiples técnicas de detección de outliers
4. Interpretación contextual de resultados estadísticos

---

## 📊 Datos

### Fuente de Datos

- **Origen:** UEFA Champions League - Estadísticas Oficiales Temporada 2025
- **URL:** [GitHub - Master Dataset](https://raw.githubusercontent.com/Emma-Ok/Data-science-project/main/sources/master_df.csv)
- **Método de Recolección:** Scraping de sitio web oficial UEFA + datos de tracking GPS
- **Período:** Primeros 4 partidos de la fase de grupos 2025

### Características del Dataset

| Característica | Valor |
|----------------|-------|
| **Total de Registros** | 908 jugadores |
| **Total de Variables** | 50 variables |
| **Variables Continuas** | 9 (18%) |
| **Variables Discretas** | 34 (68%) |
| **Variables Categóricas** | 7 (14%) |
| **Período de Análisis** | 4 partidos (fase de grupos) |

### Variables por Categoría

#### 🔹 Variables Categóricas (7)
- `player_name` - Nombre del jugador
- `nationality` - Nacionalidad
- `field_position` - Posición general (portero, defensa, mediocampo, delantero)
- `position` - Posición específica (GK, DF, MF, FW)
- `team` - Equipo al que pertenece

#### 🔹 Variables Discretas (34)
Métricas contables de rendimiento:
- **Ofensivas:** `goals`, `assists`, `total_attempts`, `inside_area`, `outside_area`
- **Defensivas:** `tackles`, `tackles_won`, `tackles_lost`, `balls_recovered`
- **Disciplinarias:** `yellow_cards`, `red_cards`, `fouls_committed`
- **Portería:** `saves`, `goals_conceded`, `clean_sheets`, `saves_on_penalty`
- **Contextuales:** `age`, `matches_appearance`

#### 🔹 Variables Continuas (9)
Métricas medibles en escala continua:
- **Biométricas:** `weight(kg)`, `height(cm)`
- **Físicas:** `distance_covered(km/h)`, `top_speed`
- **Técnicas:** `passing_accuracy(%)`, `crossing_accuracy(%)`
- **Rendimiento:** `minutes_played`, `attempts_on_target`

---

## 📑 TABLA 1: Documentación Completa de Variables (50 Variables)

### Variables Categóricas

| Variable | Tipo | Definición | Fuente |
|----------|------|------------|--------|
| `player_name` | Categórica | Nombre completo del jugador registrado en la competición | UEFA Champions League - Registro oficial de plantilla |
| `nationality` | Categórica | Nacionalidad o país de origen del jugador | UEFA Champions League - Registro oficial de jugadores |
| `field_position` | Categórica | Ubicación general del jugador en el campo (portero, defensa, mediocampista, delantero) | UEFA Champions League - Clasificación oficial de posiciones |
| `position` | Categórica | Posición específica del jugador (FW, MF, DF, GK) | UEFA Champions League - Clasificación oficial de posiciones |
| `team` | Categórica | Nombre del equipo al que pertenece el jugador | UEFA Champions League - Datos oficiales de plantilla |

### Variables Discretas

| Variable | Tipo | Definición | Fuente |
|----------|------|------------|--------|
| `age` | Discreta | Edad del jugador expresada en años completos | UEFA Champions League - Registro oficial de jugadores |
| `id_team` | Discreta | Identificador numérico único del equipo | UEFA Champions League - Codificación interna de equipos |
| `yellow_cards` | Discreta | Número total de tarjetas amarillas recibidas | UEFA Champions League - Estadísticas disciplinarias oficiales |
| `red_cards` | Discreta | Número total de tarjetas rojas recibidas | UEFA Champions League - Estadísticas disciplinarias oficiales |
| `matches_appearance` | Discreta | Cantidad de partidos en los que el jugador ha participado | UEFA Champions League - Actas oficiales de partido |
| `assists` | Discreta | Número total de asistencias de gol realizadas | UEFA Champions League - Estadísticas de rendimiento oficial |
| `corners_taken` | Discreta | Número de tiros de esquina ejecutados | UEFA Champions League - Datos de rendimiento técnico |
| `offsides` | Discreta | Número de veces en posición de fuera de juego | UEFA Champions League - Estadísticas de partido |
| `dribbles` | Discreta | Cantidad de regates exitosos realizados | UEFA Champions League - Datos de rendimiento ofensivo |
| `goals` | Discreta | Número total de goles anotados en la temporada | UEFA Champions League - Estadísticas de rendimiento oficial |
| `inside_area` | Discreta | Cantidad de goles/tiros dentro del área | UEFA Champions League - Estadísticas de tiro |
| `outside_area` | Discreta | Cantidad de goles/tiros fuera del área | UEFA Champions League - Estadísticas de tiro |
| `right_foot` | Discreta | Número de goles/disparos con pie derecho | UEFA Champions League - Estadísticas de tiro |
| `left_foot` | Discreta | Número de goles/disparos con pie izquierdo | UEFA Champions League - Estadísticas de tiro |
| `head` | Discreta | Número de goles/disparos de cabeza | UEFA Champions League - Estadísticas de tiro |
| `other` | Discreta | Goles/disparos con otra parte del cuerpo | UEFA Champions League - Estadísticas de tiro |
| `penalties_scored` | Discreta | Cantidad de penales convertidos | UEFA Champions League - Estadísticas oficiales |
| `saves` | Discreta | Cantidad de atajadas (porteros) | UEFA Champions League - Estadísticas de porteros |
| `goals_conceded` | Discreta | Goles recibidos (porteros) | UEFA Champions League - Estadísticas de porteros |
| `saves_on_penalty` | Discreta | Penales detenidos (porteros) | UEFA Champions League - Estadísticas de porteros |
| `clean_sheets` | Discreta | Partidos con arco invicto (porteros) | UEFA Champions League - Estadísticas de porteros |
| `punches_made` | Discreta | Despejes con puños (porteros) | UEFA Champions League - Estadísticas de porteros |
| `balls_recovered` | Discreta | Recuperaciones de balón | UEFA Champions League - Datos de rendimiento defensivo |
| `tackles` | Discreta | Total de entradas realizadas | UEFA Champions League - Estadísticas defensivas oficiales |
| `tackles_won` | Discreta | Entradas ganadas exitosamente | UEFA Champions League - Estadísticas defensivas oficiales |
| `tackles_lost` | Discreta | Entradas perdidas | UEFA Champions League - Estadísticas defensivas oficiales |
| `clearance_attempted` | Discreta | Despejes intentados | UEFA Champions League - Estadísticas defensivas oficiales |

### Variables Continuas

| Variable | Tipo | Definición | Fuente |
|----------|------|------------|--------|
| `weight(kg)` | Continua | Peso corporal en kilogramos | UEFA Champions League - Ficha biométrica oficial |
| `height(cm)` | Continua | Altura en centímetros | UEFA Champions League - Ficha biométrica oficial |
| `fouls_committed` | Continua | Promedio de faltas cometidas por partido | UEFA Champions League - Estadísticas disciplinarias |
| `fouls_suffered` | Continua | Promedio de faltas recibidas por partido | UEFA Champions League - Estadísticas disciplinarias |
| `total_attempts` | Continua | Promedio de intentos de disparo | UEFA Champions League - Estadísticas de tiro |
| `attempts_on_target` | Continua | Tiros dirigidos al arco | UEFA Champions League - Estadísticas de tiro |
| `attempts_off_target` | Continua | Tiros desviados | UEFA Champions League - Estadísticas de tiro |
| `blocked` | Continua | Tiros bloqueados | UEFA Champions League - Estadísticas de tiro |
| `passing_accuracy(%)` | Continua | Porcentaje de precisión en pases | UEFA Champions League - Datos oficiales 2025 |
| `passes_attempted` | Continua | Pases intentados | UEFA Champions League - Estadísticas de pases |
| `passes_completed` | Continua | Pases completados con éxito | UEFA Champions League - Estadísticas de pases |
| `crossing_accuracy(%)` | Continua | Porcentaje de acierto en centros | UEFA Champions League - Datos oficiales 2025 |
| `crosses_attempted` | Continua | Centros intentados | UEFA Champions League - Estadísticas ofensivas |
| `crosses_completed` | Continua | Centros completados | UEFA Champions League - Estadísticas ofensivas |
| `free_kick_taken` | Continua | Tiros libres ejecutados | UEFA Champions League - Estadísticas de balón parado |
| `distance_covered(km/h)` | Continua | Distancia total recorrida (km/h) | UEFA Champions League - Tracking físico oficial |
| `top_speed` | Continua | Velocidad máxima alcanzada | UEFA Champions League - Sistema de tracking físico |
| `minutes_played` | Continua | Minutos totales disputados | UEFA Champions League - Actas oficiales de partido |

---

## 📑 TABLA 2: Variables Seleccionadas para Análisis Principal (9 Variables)

Esta tabla documenta las **9 variables clave** seleccionadas para el análisis exploratorio y detección de outliers, elegidas por su relevancia en la evaluación del rendimiento deportivo.

| Variable | Tipo | Definición | Relevancia para el Análisis | Fuente |
|----------|------|------------|----------------------------|--------|
| **`age`** | Discreta | Edad del jugador (años completos) | Permite analizar la relación entre madurez deportiva y rendimiento técnico-físico. Identificar pico de rendimiento. | UEFA Champions League - Registro oficial 2025 |
| **`goals`** | Discreta | Total de goles anotados | Mide directamente la eficacia ofensiva y capacidad de finalización. Métrica fundamental de rendimiento. | UEFA Champions League - Estadísticas oficiales 2025 |
| **`assists`** | Discreta | Total de asistencias realizadas | Evalúa creatividad y contribución al juego colectivo. Complementa métricas ofensivas. | UEFA Champions League - Estadísticas oficiales 2025 |
| **`tackles_won`** | Discreta | Entradas exitosas en disputa | Refleja efectividad defensiva y capacidad de recuperación de balón. Clave para evaluación de defensores. | UEFA Champions League - Estadísticas defensivas 2025 |
| **`total_attempts`** | Continua | Promedio de intentos de tiro | Indica agresividad ofensiva y oportunidades creadas. Correlaciona con potencial goleador. | UEFA Champions League - Estadísticas de tiro 2025 |
| **`passing_accuracy(%)`** | Continua | Porcentaje de precisión en pases | Determina calidad técnica y control del juego. Fundamental para mediocampistas constructores. | UEFA Champions League - Datos oficiales 2025 |
| **`distance_covered(km/h)`** | Continua | Distancia total recorrida | Mide resistencia física y compromiso táctico. Indicador de carga de trabajo del jugador. | UEFA Champions League - Tracking GPS oficial 2025 |
| **`top_speed`** | Continua | Velocidad máxima alcanzada | Evalúa capacidad explosiva y potencial en acciones de alta velocidad. Crucial para extremos y delanteros. | UEFA Champions League - Sistema tracking 2025 |
| **`minutes_played`** | Continua | Minutos totales disputados | Contextualiza todas las métricas (rendimiento por minuto). Diferencia titulares vs suplentes. | UEFA Champions League - Actas oficiales 2025 |

### Criterios de Selección

Las 9 variables fueron seleccionadas aplicando los siguientes criterios:

✅ **Cobertura Multidimensional:**
- Dimensión Ofensiva: `goals`, `assists`, `total_attempts`
- Dimensión Defensiva: `tackles_won`
- Dimensión Física: `distance_covered`, `top_speed`
- Dimensión Técnica: `passing_accuracy`
- Dimensión Contextual: `age`, `minutes_played`

✅ **Relevancia Deportiva:**
- Variables utilizadas por scouts profesionales
- Métricas estándar en análisis de rendimiento UEFA
- Indicadores reconocidos en literatura científica deportiva

✅ **Calidad de Datos:**
- Completitud > 98% (mínimos valores faltantes)
- Precisión garantizada por sistemas oficiales UEFA
- Variabilidad suficiente para análisis estadístico

✅ **Capacidad Discriminativa:**
- Permiten diferenciar niveles de rendimiento (élite vs promedio)
- Capturan variabilidad entre posiciones (FW, MF, DF, GK)
- Sensibles a estilos tácticos y estrategias de equipo

---

## 🔬 Metodología

### Fase 1: Construcción de Base de Datos

1. **Recolección de Datos**
   - Scraping de estadísticas oficiales UEFA
   - Integración de datos de tracking físico (GPS)
   - Validación de integridad de datos

2. **Documentación de Variables**
   - Clasificación por tipo (continua, discreta, categórica)
   - Definiciones operacionales claras
   - Identificación de fuentes para cada variable

3. **Limpieza y Preparación**
   - Eliminación de duplicados
   - Tratamiento de valores faltantes (< 2%)
   - Estandarización de nombres de equipos y posiciones

### Fase 2: Análisis Exploratorio de Datos (EDA)

#### Estadísticos Descriptivos
- Media, mediana, moda
- Desviación estándar, varianza
- Percentiles (25, 50, 75)
- Rango, mínimo, máximo

#### Visualizaciones Univariadas
- **Histogramas:** Distribución de variables continuas
- **Boxplots:** Identificación visual de outliers y dispersión
- **Gráficas de Barras:** Frecuencias de variables categóricas/discretas

#### Visualizaciones Bivariadas/Multivariadas
- **Matriz de Correlaciones:** Relaciones entre variables continuas
- **Scatter Plots:** Patrones de asociación (ej: intentos vs goles)
- **Heatmaps:** Rendimiento por posición y equipo
- **Pairplots:** Análisis multivariado (9 variables simultáneas)
- **Tablas Cruzadas:** Promedios por grupos (posición, edad, equipo)

### Fase 3: Detección de Atípicos (Outliers)

Se aplicaron **4 técnicas complementarias** para identificar valores atípicos:

#### 1️⃣ IQR (Interquartile Range)
```python
# Técnica estadística clásica
Lower Bound = Q1 - 1.5 × IQR
Upper Bound = Q3 + 1.5 × IQR
Outliers: valores fuera de [Lower Bound, Upper Bound]
```
- **Sensibilidad:** Media
- **Ventaja:** Robusta ante distribuciones asimétricas
- **Resultado:** ~85 outliers detectados (9.4%)

#### 2️⃣ Z-Score (Puntuación Estandarizada)
```python
# Basado en desviaciones estándar
Z = (X - μ) / σ
Outliers: |Z| > 3
```
- **Sensibilidad:** Alta (más conservador)
- **Ventaja:** Detecta valores verdaderamente extremos
- **Resultado:** ~45 outliers detectados (5.0%)

#### 3️⃣ Isolation Forest (Machine Learning)
```python
# Algoritmo de ensemble
from sklearn.ensemble import IsolationForestClassifier
contamination = 0.05  # 5% esperado de outliers
```
- **Sensibilidad:** Media-Alta
- **Ventaja:** Detecta combinaciones atípicas multivariadas
- **Resultado:** 45 outliers detectados (5.0%)

#### 4️⃣ DBSCAN (Clustering por Densidad)
```python
# Clustering no supervisado
from sklearn.cluster import DBSCAN
eps = 1.5
min_samples = 5
Outliers: cluster = -1 (baja densidad)
```
- **Sensibilidad:** Variable
- **Ventaja:** Identifica perfiles únicos sin cluster
- **Resultado:** 38 outliers detectados (4.2%)

---

## 📁 Estructura del Proyecto

```bash
project_root/
├── 📂 article/
│   └── report_1_Emmanuel_Valbuena.pdf            
│
├── 📂 classroom_project/
│   ├── Cp1_variable_analysis_Emmanuel_Valbuena.ipynb
│   ├── py_Emmanuel_Valbuena_01_intro.ipynb
│
├── 📂 practical_sessions/
│   ├── Ps_1_Emmanuel_Valbuena.ipynb              
│   ├── Ps_2_Emmanuel_Valbuena.ipynb              
│   ├── Ps_3_Emmanuel_Valbuena.ipynb              
│   └── Ps_4_Emmanuel_Valbuena.ipynb              
│
├── 📂 sources/
│   ├── KaggleLink.txt                            # Enlace al dataset original en Kaggle
│   ├── UEFAChampionsDataScience.ipynb            # Notebook auxiliar para procesamiento y extracción de datos
│   ├── master_df.csv                             # Dataset consolidado y procesado para el análisis principal
│   └── tabla_descriptiva_variables.csv           # Resumen estadístico de las variables seleccionadas
│
          # Notebook introductorio (contexto, entorno y librerías base)
│
└── README.md                                     # Documentación principal del proyecto

---

## 🛠️ Instalación

### Prerrequisitos

- Python 3.8 o superior
- pip (gestor de paquetes de Python)
- Jupyter Notebook o JupyterLab

### Pasos de Instalación

1. **Clonar el repositorio:**
```bash
git clone https://github.com/Emma-Ok/Data-science-project.git
cd Data_analysis
```

2. **Crear entorno virtual (recomendado):**
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

3. **Instalar dependencias:**
```bash
pip install -r requirements.txt
```

### Dependencias Principales

```txt
pandas==2.0.3
numpy==1.24.3
matplotlib==3.7.2
seaborn==0.12.2
scikit-learn==1.3.0
scipy==1.11.1
jupyter==1.0.0
```

---


## 🔧 Técnicas Aplicadas

### Estadística Descriptiva
- ✅ Medidas de tendencia central (media, mediana, moda)
- ✅ Medidas de dispersión (desviación estándar, varianza, rango)
- ✅ Percentiles y cuartiles
- ✅ Distribución de frecuencias

### Visualización de Datos
- ✅ **Histogramas:** Distribución de variables continuas
- ✅ **Boxplots:** Detección visual de outliers
- ✅ **Bar Charts:** Frecuencias de categorías
- ✅ **Scatter Plots:** Relaciones bivariadas
- ✅ **Heatmaps:** Correlaciones y tablas cruzadas
- ✅ **Pairplots:** Análisis multivariado

### Análisis de Correlación
- ✅ Matriz de correlación de Pearson
- ✅ Pruebas de significancia estadística
- ✅ Interpretación contextual deportiva

### Detección de Outliers
- ✅ **IQR (Interquartile Range):** Método estadístico clásico
- ✅ **Z-Score:** Basado en desviaciones estándar
- ✅ **Isolation Forest:** Machine Learning (sklearn)
- ✅ **DBSCAN:** Clustering por densidad

### Reducción de Dimensionalidad
- ✅ **PCA (Principal Component Analysis):** Visualización 2D de datos multivariados

---

## 📊 Visualizaciones

### Galería de Gráficas

<details>
<summary>📈 Histogramas - Distribuciones Univariadas</summary>

- Variables continuas: `passing_accuracy`, `distance_covered`, `top_speed`, `minutes_played`
- Identificación de patrones de distribución (normal, asimétrica, bimodal)
- Detección visual de concentración de valores

</details>

<details>
<summary>📊 Boxplots - Análisis de Dispersión</summary>

- Visualización de cuartiles (Q1, Q2, Q3)
- Identificación de outliers superiores e inferiores
- Comparación de dispersión entre variables

</details>

<details>
<summary>📊 Gráficas de Barras - Variables Discretas</summary>

- Distribución de `goals`, `assists`, `tackles_won`
- Frecuencias por posición, equipo, nacionalidad
- Identificación de categorías dominantes

</details>

<details>
<summary>🔥 Matriz de Correlaciones - Heatmap</summary>

- Correlaciones entre 9 variables principales
- Escala de colores: rojo (positiva) a azul (negativa)
- Identificación de relaciones significativas

</details>

<details>
<summary>🎯 Scatter Plots - Relaciones Bivariadas</summary>

- `total_attempts` vs `goals`: Eficiencia ofensiva
- `passing_accuracy` vs `assists`: Calidad técnica
- `age` vs `top_speed`: Declive físico
- Líneas de regresión y R²

</details>

<details>
<summary>🌐 Pairplot - Análisis Multivariado</summary>

- 9 variables × 9 variables = 81 gráficas
- Distribuciones en diagonal (KDE)
- Scatter plots bivariados fuera de diagonal
- Coloración por posición (FW, MF, DF, GK)

</details>

<details>
<summary>🔍 Outliers IQR - Detección Estadística</summary>

- Boxplots con límites IQR marcados
- Outliers resaltados en rojo
- Comparación por variable

</details>

<details>
<summary>🤖 Outliers Isolation Forest - ML</summary>

- Proyección PCA en 2D
- Outliers vs Normal (colores diferenciados)
- Scores de anomalía

</details>

<details>
<summary>🔬 Outliers DBSCAN - Clustering</summary>

- Visualización de clusters detectados
- Outliers = puntos sin cluster (cluster = -1)
- Distribución de observaciones por cluster

</details>

<details>
<summary>⚖️ Comparación de Técnicas - Heatmap</summary>

- % de outliers por variable y técnica
- Identificación de consensos
- Sensibilidad comparada

</details>

---

## 💡 Conclusiones

### Principales Hallazgos

1. **✅ Especialización Posicional Clara**
   - Los delanteros dominan métricas ofensivas (goles, intentos)
   - Mediocampistas lideran en precisión técnica (pases)
   - Defensores destacan en recuperación defensiva (tackles)

2. **✅ Edad Óptima de Rendimiento: 24-28 años**
   - Pico de combinación física + técnica + experiencia
   - Jugadores jóvenes (< 23): alta velocidad, menor precisión
   - Jugadores veteranos (> 30): técnica preservada, pérdida física

3. **✅ Importancia de la Eficiencia**
   - Correlación moderada intentos-goles (r=0.58)
   - No basta tener muchos intentos, la calidad cuenta
   - Goleadores de élite: alta conversión (goles/intentos)

4. **✅ Outliers = Excelencia Deportiva**
   - 4-9% del dataset son outliers según técnica
   - Representan jugadores de clase mundial (Mbappé, Haaland)
   - **Decisión:** NO eliminar, mantener para análisis de élite

5. **✅ Validación de Estrategias Tácticas**
   - Datos confirman roles diferenciados por posición
   - Distancia recorrida correlaciona con minutos jugados (r=0.89)
   - Precisión de pases facilita asistencias (r=0.45)

### Fase 4: Preparación Avanzada de Datos

- **Normalidad y forma de las distribuciones:** Se contrastaron Shapiro-Wilk, asimetría y Q-Q plots para resguardar la forma real de `passing_accuracy(%)`, `distance_covered(km/h)`, `goals` y `total_attempts`. Se mantuvieron las distribuciones originales y solo se sugieren transformaciones logarítmicas cuando un modelo estrictamente gaussiano lo requiera.
- **Valores faltantes MNAR:** El patrón responde a roles deportivos (suplentes sin minutos, métricas no capturadas por posición). Imputar agregaría sesgos, por lo que se optó por conservarlos, documentarlos y suplirlos con indicadores binarios o análisis segmentados cuando sea necesario.
- **Codificación híbrida:** Variables con ≤10 categorías usan One-Hot Encoding; las de alta cardinalidad (club, nacionalidad) emplean Label Encoding documentado. El resultado es `df_encoded`, listo para pipelines y dashboards sin perder interpretabilidad.
- **Escalamiento multitécnica:** Se generaron `df_standardized`, `df_normalized` y `df_robust`. RobustScaler se declara estándar porque protege la señal de los outliers legítimos sin distorsionar al resto de jugadores.
- **Entregables y próximos pasos:** Además de los datasets anteriores, se liberaron reportes visuales de missing y comparativas antes/después. El stack queda listo para modelado predictivo, clustering táctico y análisis explicable (Feature Importance/SHAP) sobre `df_robust + df_encoded`.

### Recomendaciones

**Para Equipos Técnicos:**
- Priorizar jugadores en rango 24-28 años para rendimiento inmediato
- Monitorear carga física (minutos > 300 en 4 partidos = riesgo)
- Inversión en mediocampistas con passing_accuracy > 85%

**Para Scouting:**
- Buscar delanteros con alta conversión (goles/intentos)
- Identificar jóvenes velocistas (< 23 años, top_speed > 34 km/h)
- Valorar versatilidad (outliers multivariados de Isolation Forest)

**Para Analistas de Rendimiento:**
- Segmentar análisis por posición (evitar comparaciones injustas)
- Contextualizar métricas por minutos jugados
- Aplicar análisis multivariado (no solo variables individuales)

### Limitaciones del Estudio

⚠️ **Período Limitado:** Solo 4 partidos (fase inicial de grupos)  
⚠️ **Contexto Ausente:** No se consideran formaciones tácticas, rivales específicos  
⚠️ **Variables Omitidas:** Falta información sobre presión, duelos aéreos, temperatura  

---

## 👤 Autor

**Emmanuel Valbuena**  
📧 Email: [emmanuel.bustamante@udea.educ.co]  


**Rol:** Analista de Datos 
**Institución:** [Universidad de Antioquia]  
**Fecha:** Octubre 2025



<div align="center">

**⚽ Hecho con pasión por la ciencia de datos y el fútbol ⚽**

![Footer](https://img.shields.io/badge/Made%20with-Python%20%F0%9F%90%8D-blue?style=for-the-badge)
![Footer](https://img.shields.io/badge/Data%20Science-⚽%20Football-green?style=for-the-badge)

</div>
