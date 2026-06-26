# 💳 De la deuda al dato: *Machine Learning* para predecir defaults 

Modelado predictivo a partir de datos crediticios de Taiwán (año 2005) bajo el marco del **Trabajo Práctico Final** para la materia *Introducción al Aprendizaje Automático* (LCD-UNSAM)

---

## 📚 Contexto

En el año 2005, Taiwán sufrió una crisis en su sistema bancario originada por el impago de las tarjetas de crédito por parte de los clientes de los bancos. Las causas estructurales pueden encontrarse en la falta de responsabilidad por parte de las entidades bancarias al evaluar a sus potenciales clientes y en las dificultades de estos últimos para afrontar los gastos por la coyuntura del país en ese momento. 

A pesar de esas condiciones iniciales, **¿los bancos se podrían haber anticipado a la situación e identificado a potenciales deudores con anterioridad?**

## 🎯 Definición del Problema

El impago de tarjetas de crédito genera múltiples problemáticas a las entidades bancarias entre las que se encuentran: pérdidas por créditos incobrables, costos de recuperación (gestión de cobranza, procesos judiciales, etc.), impacto en el flujo de caja que afecta directamente la capacidad de otorgar nuevos créditos, un deterioro en la reputación del banco sobre la calidad de su gestión, entre otros.

Por todas estas razones, resulta clave para estas entidades anticiparse a la situación e identificar con tiempo y efectividad al segmento de clientes con mayores probabilidades de entrar en mora. Teniendo esta información se pueden dirigir los esfuerzos específicamente a ese grupo, tomando medidas preventivas en etapas tempranas de la deuda e incluso mejorar los criterios de decisión previos para el otorgamiento de nuevo crédito.

## 💡 Solución Propuesta

En este proyecto trabajamos con la familia de modelos de árboles de decisión para identificar a potenciales deudores utilizando datos demográficos y de comportamiento financiero para el aprendizaje de los modelos. 

Dado que el costo de no identificar a clientes deudores es mucho más elevado al costo de catalogar como deudor a un cliente que al final paga su deuda (en este último caso la pérdida económica solo implicaría la primera etapa de la gestión de cobranza), buscamos optimizar el modelo para que **maximice el** ***recall*** que es justamente la métrica que nos indica la capacidad de identificar a deudores reales sobre todo el total de deudores.

## 📊 Resultados Principales

| Modelo | Recall | Precision | F1-Score | AUC-ROC |
|--------|--------|-----------|----------|---------|
| Benchmark (Aleatorio) | ~20% | ~20% | ~20% | ~0.49 |
| Árbol — 2 variables | 52.3% | 33.4% | 40.8% | 0.656 |
| Árbol — completo | 60.8% | 30.3% | 40.5% | 0.658 |
| Árbol — GridSearchCV | 64.0% | 29.7% | 40.5% | 0.660 |
| Random Forest | 57.9% | 33.9% | 42.7% | 0.686 |

Los meses de deuda acumulados (`n_meses_deuda_sep`) y el límite de crédito (`limite_credito`) son consistentemente las variables más importantes para la predicción, concentrando el 83% de la importancia total en los modelos de árbol.

## 📁 Estructura del Proyecto

```
prediccion-default-crediticio/
├── data/
│   ├── raw/                        # Dataset original (UCI ML Repository)
│   └── processed/                  # Dataset limpio generado por el pipeline
├── notebooks/
│   ├── 01-carga_limpieza.ipynb     # Exploración inicial y limpieza de datos
│   ├── 02-exploración.ipynb        # Análisis exploratorio (EDA)
│   ├── 03-feature-engineering.ipynb # Transformación de variables
│   └── 04-modelado.ipynb           # Entrenamiento y evaluación de modelos
├── src/
│   ├── data/
│   │   ├── load_data.py            # Carga y guardado de datos
│   │   └── preprocess.py           # Pipeline de preprocesamiento
│   ├── features/
│   │   └── feature_engineering.py  # Transformación de features
│   └── visualization/
│       ├── config_vis.py           # Configuración visual global
│       └── ml_plots.py             # Plots de frontera de decisión
├── requirements.txt
└── README.md
```

## 📩 Datos Utilizados

El dataset fue obtenido desde el [UCI Irvine Machine Learning Repository](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) y fue recolectado por Yeh y Lien (2009) con el objetivo de evaluar la precisión predictiva de la probabilidad de default a través de distintos métodos de minería de datos.

Contiene observaciones de pago de clientes de un banco taiwanés en 2005, incluyendo variables demográficas (género, estado civil, edad, nivel educativo), el estado de pago mensual para el período Abril–Septiembre, los montos facturados y pagados por mes, y una variable binaria que indica si el cliente cayó en default en octubre.

Luego de la limpieza (eliminación de registros con transiciones de deuda ilógicas y casos de pago superiores a la factura sin deuda previa), el dataset final quedó con **25.492 observaciones y 9 variables**.

## 📈 Metodología

1. **Limpieza de datos** — se eliminaron 4.508 registros con inconsistencias en el historial de deuda o en los montos de pago, y se reagruparon categorías mal documentadas en las variables `educacion` y `estado_civil`.

2. **Análisis exploratorio (EDA)** — se analizaron las distribuciones univariadas y las correlaciones con el target. Se identificó que `meses_deuda_sep` y `limite_credito` son los predictores más prometedores.

3. **Feature engineering** — se descompuso la columna `meses_deuda_sep` (cuya codificación mezcla estados cualitativos y cuantitativos) en 5 variables separadas, y se construyeron dos ratios financieros: utilización del crédito y compromiso de pago.

4. **Modelado** — se entrenaron y evaluaron cuatro modelos de forma incremental: árbol de decisión con 2 variables, árbol completo con pipeline, árbol optimizado con GridSearchCV y Random Forest. Se utilizó validación cruzada estratificada (5-fold) para estimar la varianza de las métricas.

## 🛠️ Tecnologías Utilizadas

- **Python 3.13** — lenguaje de desarrollo
- **Jupyter Notebooks** — entorno interactivo para análisis y documentación
- **scikit-learn** — implementación de modelos y pipelines de ML
- **seaborn & matplotlib** — visualización de datos y resultados
- **numpy & pandas** — manipulación y análisis de datos

## ⚙️ Instalación

```bash
git clone https://github.com/tu-usuario/prediccion-default-crediticio.git
cd prediccion-default-crediticio
pip install -r requirements.txt
```

Los notebooks deben ejecutarse en orden desde la carpeta `notebooks/`. El notebook de limpieza genera el CSV procesado que usan los siguientes.

## 🧠 Conclusiones y Aprendizajes

- El historial de mora reciente y la capacidad crediticia del cliente (límite de crédito) son las señales más directas de riesgo de default, superando en importancia a las variables demográficas.
- El desbalance de clases (78% no default / 22% default) requiere estrategias explícitas: `class_weight='balanced'` y el uso del recall como métrica principal en lugar del accuracy.
- La descomposición de `meses_deuda_sep` en features separadas es un ejemplo concreto de cómo el conocimiento del dominio mejora la representación de los datos para los modelos.
- La validación cruzada estratificada es clave para obtener estimaciones confiables de las métricas cuando el dataset está desbalanceado.

## 🧑‍💻 Autores | Contacto

Estamos abiertos a recibir ideas, sugerencias o comentarios! Podés contactarnos por LinkedIn o Gmail.
- **Bruno Inguanzo** · [LinkedIn](https://www.linkedin.com/in/bruno-inguanzo-974021212/) · [brunoinguanzo14@gmail.com](mailto:brunoinguanzo14@gmail.com)
- **Javier Valdez** · [LinkedIn](https://www.linkedin.com/in/javiervaldez2/) · [javiervaldez145@gmail.com](mailto:javiervaldez145@gmail.com) 
- **Matías Vergara** · [LinkedIn](https://www.linkedin.com/in/matiasvergaravicencio/) · [ma.vergaravicencio@gmail.com](mailto:ma.vergaravicencio@gmail.com)
