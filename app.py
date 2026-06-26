import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score

import src.data.load_data as ld
import src.data.preprocess as pp
import src.features.feature_engineering as fe


# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

DEFAULTS = {
    "limite": 150_000, "genero": 2, "educacion": 2, "estado_civil": 2,
    "edad": 35, "meses_deuda": -1, "pago": 2_000, "factura": 20_000,
}

PRESETS = {
    "sin_riesgo": {
        "label": "✅ Sin riesgo",
        "caption": "Profesional, pagó el total a tiempo, límite alto",
        "values": {
            "limite": 300_000, "genero": 2, "educacion": 2, "estado_civil": 2,
            "edad": 38, "meses_deuda": -1, "pago": 15_000, "factura": 15_000,
        },
    },
    "zona_gris": {
        "label": "🟡 Zona gris",
        "caption": "Pago mínimo, factura elevada respecto al límite",
        "values": {
            "limite": 100_000, "genero": 1, "educacion": 3, "estado_civil": 1,
            "edad": 45, "meses_deuda": 0, "pago": 1_500, "factura": 30_000,
        },
    },
    "alto_riesgo": {
        "label": "🔴 Alto riesgo",
        "caption": "Meses de atraso acumulados, límite de crédito bajo",
        "values": {
            "limite": 50_000, "genero": 1, "educacion": 4, "estado_civil": 2,
            "edad": 27, "meses_deuda": 3, "pago": 0, "factura": 48_000,
        },
    },
}

BEST_PARAMS_M3 = {
    "max_depth": 4,
    "min_samples_leaf": 20,
    "min_samples_split": 2,
    "class_weight": {0: 1, 1: 4},
}

METRICAS_CONOCIDAS = pd.DataFrame(
    {
        "Modelo": [
            "Benchmark (Aleatorio)",
            "Árbol — 2 variables",
            "Árbol — completo",
            "Árbol — GridSearchCV",
            "Random Forest",
        ],
        "Recall": ["~20%", "52.3%", "60.8%", "64.0%", "57.9%"],
        "Precision": ["~20%", "33.4%", "30.3%", "29.7%", "33.9%"],
        "F1-Score": ["~20%", "40.8%", "40.5%", "40.5%", "42.7%"],
        "AUC-ROC": ["~0.49", "0.656", "0.658", "0.660", "0.686"],
    }
).set_index("Modelo")

DESCRIPCIONES = {
    "m1": "Árbol — 2 variables",
    "m2": "Árbol — completo",
    "m3": "Árbol — GridSearchCV",
    "m4": "Random Forest",
}

DETALLE_MODELOS = {
    "m1": "Árbol de decisión entrenado únicamente con **límite de crédito** y **meses de deuda**. Las 2 variables con mayor correlación con el target.",
    "m2": "Árbol de decisión con las **10 features** (numéricas + categóricas con one-hot encoding) y un pipeline de scikit-learn.",
    "m3": "Mismo pipeline que el Modelo 2 pero con los hiperparámetros óptimos encontrados por GridSearchCV, penalizando más los falsos negativos.",
    "m4": "Ensemble de 200 árboles de decisión. Reduce la varianza y aprovecha mejor las features secundarias.",
}


# ---------------------------------------------------------------------------
# Carga y entrenamiento (cacheado)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Cargando datos y entrenando modelos…")
def cargar_modelos_y_datos():
    # Preprocesamiento sin side-effect de guardado
    df = ld.load_raw_data()
    df = pp.eliminar_id(df)
    df = pp.renombrar_columnas(df)
    df = pp.normalizar_categorias(df)
    df = pp.filtrar_inconsistencias_deuda(df)
    df = pp.filtrar_inconsistencias_factura_pago(df)
    df = pp.seleccionar_columnas_septiembre(df)
    df = fe.aplicar_feature_engineering(df)

    cols = fe.get_feature_cols()
    num_features = cols["numericas"]
    cat_features = cols["categoricas"]

    X = df[num_features + cat_features]
    y = df["default_oct"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train_m1 = X_train[["limite_credito", "n_meses_deuda_sep"]]
    X_test_m1 = X_test[["limite_credito", "n_meses_deuda_sep"]]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
        ]
    )

    # Modelo 1 — árbol 2 variables
    m1 = DecisionTreeClassifier(max_depth=4, class_weight="balanced", random_state=42)
    m1.fit(X_train_m1, y_train)

    # Modelo 2 — árbol completo
    m2 = Pipeline([
        ("preprocessor", ColumnTransformer([
            ("num", "passthrough", num_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
        ])),
        ("classifier", DecisionTreeClassifier(max_depth=4, class_weight="balanced", random_state=42)),
    ])
    m2.fit(X_train, y_train)

    # Modelo 3 — árbol con mejores parámetros del GridSearch
    m3 = Pipeline([
        ("preprocessor", ColumnTransformer([
            ("num", "passthrough", num_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
        ])),
        ("classifier", DecisionTreeClassifier(
            random_state=42, **BEST_PARAMS_M3
        )),
    ])
    m3.fit(X_train, y_train)

    # Modelo 4 — Random Forest
    m4 = Pipeline([
        ("preprocessor", ColumnTransformer([
            ("num", "passthrough", num_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
        ])),
        ("classifier", RandomForestClassifier(
            n_estimators=200, max_depth=8, class_weight="balanced",
            random_state=42, n_jobs=-1,
        )),
    ])
    m4.fit(X_train, y_train)

    modelos = {"m1": m1, "m2": m2, "m3": m3, "m4": m4}

    # Métricas reales sobre test set
    metricas = {}
    for key, modelo in modelos.items():
        X_t = X_test_m1 if key == "m1" else X_test
        y_pred = modelo.predict(X_t)
        y_proba = modelo.predict_proba(X_t)[:, 1]
        metricas[key] = {
            "Recall": recall_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "F1-Score": f1_score(y_test, y_pred),
            "AUC-ROC": roc_auc_score(y_test, y_proba),
        }

    return modelos, metricas, num_features, cat_features


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def construir_input_df(limite, genero, educacion, estado_civil, edad,
                        meses_deuda, pago, factura):
    fila = {
        "limite_credito": limite,
        "genero": genero,
        "educacion": educacion,
        "estado_civil": estado_civil,
        "edad": edad,
        "meses_deuda_sep": meses_deuda,
        "pago_sep": pago,
        "factura_sep": factura,
    }
    df_input = pd.DataFrame([fila])
    df_fe = fe.aplicar_feature_engineering(df_input)
    return df_fe


def mostrar_card_modelo(modelo, X_input, key, col):
    proba = modelo.predict_proba(X_input)[0][1]
    es_default = proba >= 0.5
    nombre = DESCRIPCIONES[key]

    with col:
        st.markdown(f"**{nombre}**")
        if es_default:
            st.error("RIESGO ALTO")
        else:
            st.success("RIESGO BAJO")
        st.metric(label="Prob. de default", value=f"{proba:.1%}")
        st.progress(float(proba))


# ---------------------------------------------------------------------------
# App principal
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="Predictor de Default",
        page_icon="💳",
        layout="wide",
    )

    modelos, metricas, num_features, cat_features = cargar_modelos_y_datos()

    if "inputs" not in st.session_state:
        st.session_state.inputs = DEFAULTS.copy()

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.title("💳 Default Crediticio")
        st.markdown(
            """
            App de predicción de impago de tarjeta de crédito basada en datos de
            un banco taiwanés (2005).

            Ingresá el perfil de un cliente y los **4 modelos** estimarán la
            probabilidad de que entre en default el mes siguiente.
            """
        )
        st.divider()

        st.subheader("Rendimiento de los modelos")
        st.caption("Evaluados sobre el 20% del dataset reservado para test.")

        metricas_df = pd.DataFrame(metricas).T.rename(index=DESCRIPCIONES)
        metricas_df = metricas_df.map(lambda x: f"{x:.1%}" if x < 1 else f"{x:.3f}")
        st.dataframe(metricas_df, use_container_width=True)

        st.divider()
        st.markdown(
            "📂 [Ver repositorio en GitHub](https://github.com/matvergara/prediccion-default-crediticio)"
        )
        st.caption("Datos: [UCI ML Repository](https://archive.uci.edu/dataset/350/default+of+credit+card+clients)")

    # ── Área principal ───────────────────────────────────────────────────────
    st.title("Predictor de Default Crediticio")
    st.markdown(
        "Completá el perfil del cliente con sus datos demográficos y el "
        "comportamiento de pago de septiembre. Los modelos predirán si "
        "entrará en default en octubre."
    )

    # ── Perfiles de ejemplo ──────────────────────────────────────────────────
    st.markdown("**Cargar perfil de ejemplo:**")
    bc1, bc2, bc3 = st.columns(3)
    for col, key in zip([bc1, bc2, bc3], PRESETS):
        preset = PRESETS[key]
        with col:
            if st.button(
                preset["label"],
                use_container_width=True,
                help=preset["caption"],
            ):
                st.session_state.inputs = preset["values"].copy()

    st.markdown("")  # espacio visual

    # ── Formulario ───────────────────────────────────────────────────────────
    vals = st.session_state.inputs

    with st.form("input_form"):
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Datos financieros")
            limite = st.number_input(
                "Límite de crédito (NT$)",
                min_value=0, max_value=1_000_000,
                value=vals["limite"], step=10_000,
            )
            pago = st.number_input(
                "Monto pagado en septiembre (NT$)",
                min_value=0, max_value=2_000_000,
                value=vals["pago"], step=500,
            )
            factura = st.number_input(
                "Factura de septiembre (NT$)",
                min_value=0, max_value=2_000_000,
                value=vals["factura"], step=500,
            )
            edad = st.slider("Edad", min_value=18, max_value=80, value=vals["edad"])

        with col2:
            st.subheader("Datos demográficos y comportamiento")
            genero = st.selectbox(
                "Género",
                options=[1, 2],
                format_func=lambda x: "Masculino" if x == 1 else "Femenino",
                index=vals["genero"] - 1,
            )
            educacion = st.selectbox(
                "Nivel educativo",
                options=[1, 2, 3, 4],
                format_func=lambda x: {
                    1: "Posgrado", 2: "Universidad",
                    3: "Secundaria", 4: "Otros"
                }[x],
                index=vals["educacion"] - 1,
            )
            estado_civil = st.selectbox(
                "Estado civil",
                options=[1, 2, 3],
                format_func=lambda x: {
                    1: "Casado/a", 2: "Soltero/a", 3: "Otros"
                }[x],
                index=vals["estado_civil"] - 1,
            )
            meses_deuda = st.selectbox(
                "Estado de pago en septiembre",
                options=[-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8],
                format_func=lambda x: {
                    -2: "Sin uso (-2)",
                    -1: "Pagó el total a tiempo (-1)",
                     0: "Pago mínimo, sin mora (0)",
                     1: "1 mes de atraso",
                     2: "2 meses de atraso",
                     3: "3 meses de atraso",
                     4: "4 meses de atraso",
                     5: "5 meses de atraso",
                     6: "6 meses de atraso",
                     7: "7 meses de atraso",
                     8: "8+ meses de atraso",
                }[x],
                index=vals["meses_deuda"] + 2,
            )

        submitted = st.form_submit_button("Predecir", use_container_width=True, type="primary")

    # ── Resultados ───────────────────────────────────────────────────────────
    if submitted:
        df_input = construir_input_df(
            limite, genero, educacion, estado_civil,
            edad, meses_deuda, pago, factura,
        )

        X_full = df_input[num_features + cat_features]
        X_m1 = df_input[["limite_credito", "n_meses_deuda_sep"]]

        st.divider()
        st.subheader("Resultados")

        c1, c2, c3, c4 = st.columns(4)
        mostrar_card_modelo(modelos["m1"], X_m1,  "m1", c1)
        mostrar_card_modelo(modelos["m2"], X_full, "m2", c2)
        mostrar_card_modelo(modelos["m3"], X_full, "m3", c3)
        mostrar_card_modelo(modelos["m4"], X_full, "m4", c4)

        with st.expander("Ver variables transformadas del cliente"):
            st.dataframe(df_input.T.rename(columns={0: "Valor"}), use_container_width=True)

        with st.expander("¿Qué hace cada modelo?"):
            for key, desc in DETALLE_MODELOS.items():
                st.markdown(f"**{DESCRIPCIONES[key]}** — {desc}")


if __name__ == "__main__":
    main()
