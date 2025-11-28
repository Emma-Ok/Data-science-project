import importlib.util
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

DBSCAN = None
IsolationForest = None
StandardScaler = None
HAS_SKLEARN = False

try:  # Optional advanced outlier techniques
    from sklearn.cluster import DBSCAN as _DBSCAN
    from sklearn.ensemble import IsolationForest as _IsolationForest
    from sklearn.preprocessing import StandardScaler as _StandardScaler

    DBSCAN = _DBSCAN
    IsolationForest = _IsolationForest
    StandardScaler = _StandardScaler
    HAS_SKLEARN = True
except ImportError:  # pragma: no cover - optional dependency
    pass
# Optional dependency detection (statsmodels is required for Plotly trendlines)
HAS_STATSMODELS = importlib.util.find_spec("statsmodels") is not None

# -----------------------------------------------------------------------------
# Page configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="UEFA Champions League 2025 – Explorador Interactivo",
    page_icon="⚽",
    layout="wide",
)

# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------
DATA_URL_DEFAULT = (
    "https://raw.githubusercontent.com/Emma-Ok/Data-science-project/main/sources/master_df.csv"
)
DATA_DICTIONARY_PATH = Path("tabla_descriptiva_variables.csv")


def load_dataset(source_url: str = DATA_URL_DEFAULT) -> pd.DataFrame:
    """Load the master dataset from GitHub or a user-provided file."""
    if "http" in source_url:
        df = pd.read_csv(source_url)
    else:
        df = pd.read_csv(source_url)
    return df


@st.cache_data(show_spinner=False)
def get_dataset(source: str = DATA_URL_DEFAULT) -> pd.DataFrame:
    df = load_dataset(source)
    # Normalise column names to simplify downstream operations
    df.columns = df.columns.str.strip()
    return df


@st.cache_data(show_spinner=False)
def get_dictionary() -> pd.DataFrame:
    if DATA_DICTIONARY_PATH.exists():
        df_dict = pd.read_csv(DATA_DICTIONARY_PATH)
        df_dict.columns = df_dict.columns.str.strip()
        return df_dict
    return pd.DataFrame()


def format_metric(value: float, suffix: str = "") -> str:
    if pd.isna(value):
        return "N/D"
    if isinstance(value, (int, np.integer)) or (float(value).is_integer()):
        return f"{int(value):,}{suffix}".replace(",", ".")
    return f"{value:,.2f}{suffix}".replace(",", ".")


# -----------------------------------------------------------------------------
# Sidebar controls
# -----------------------------------------------------------------------------
st.sidebar.header("⚙️ Configuración")

with st.sidebar:
    st.markdown(
        """
        Selecciona el origen del dataset. Por defecto se utiliza la versión
        alojada en GitHub (908 jugadores × 50 variables).
        """
    )
    remote_source = st.toggle("Usar fuente remota (GitHub)", value=True)
    custom_file = st.file_uploader("o carga un CSV propio", type=["csv"])

    if custom_file is not None:
        dataset_source = custom_file
    elif remote_source:
        dataset_source = DATA_URL_DEFAULT
    else:
        dataset_source = st.text_input("Ruta local al CSV", value="master_df.csv")

    df = get_dataset(dataset_source)

    st.markdown("---")
    st.caption("Filtrado global")

    position_col_candidates = [c for c in df.columns if "position" in c.lower()]
    team_col_candidates = [c for c in df.columns if "team" in c.lower()]
    nation_col_candidates = [c for c in df.columns if "nation" in c.lower()]
    player_col_candidates = [c for c in df.columns if "player" in c.lower()]

    position_col = position_col_candidates[0] if position_col_candidates else None
    team_col = team_col_candidates[0] if team_col_candidates else None
    nation_col = nation_col_candidates[0] if nation_col_candidates else None
    player_col = player_col_candidates[0] if player_col_candidates else None

    selected_positions = (
        st.multiselect(
            "Posiciones",
            sorted(df[position_col].dropna().unique()),
            default=None,
        )
        if position_col
        else []
    )
    selected_teams = (
        st.multiselect(
            "Equipos",
            sorted(df[team_col].dropna().unique()),
            default=None,
        )
        if team_col
        else []
    )
    selected_nations = (
        st.multiselect(
            "Nacionalidades",
            sorted(df[nation_col].dropna().unique()),
            default=None,
        )
        if nation_col
        else []
    )

    df_filtered = df.copy()
    if position_col and selected_positions:
        df_filtered = df_filtered[df_filtered[position_col].isin(selected_positions)]
    if team_col and selected_teams:
        df_filtered = df_filtered[df_filtered[team_col].isin(selected_teams)]
    if nation_col and selected_nations:
        df_filtered = df_filtered[df_filtered[nation_col].isin(selected_nations)]

    numeric_filter_candidates = df.select_dtypes(include=["number", "float", "int"]).columns.tolist()
    if numeric_filter_candidates:
        st.markdown("---")
        st.caption("Filtro numérico avanzado")
        numeric_filter_col = st.selectbox(
            "Variable numérica",
            numeric_filter_candidates,
            index=numeric_filter_candidates.index("minutes_played")
            if "minutes_played" in numeric_filter_candidates
            else 0,
        )
        min_value = float(df[numeric_filter_col].min())
        max_value = float(df[numeric_filter_col].max())
        selected_range = st.slider(
            "Rango permitido",
            min_value,
            max_value,
            (min_value, max_value),
        )
        df_filtered = df_filtered[
            df_filtered[numeric_filter_col].between(selected_range[0], selected_range[1])
        ]

    st.markdown("---")
    st.caption("Variables destacadas")

    default_focus_vars = [
        "age",
        "goals",
        "assists",
        "tackles_won",
        "total_attempts",
        "passing_accuracy(%)",
        "distance_covered(km/h)",
        "top_speed",
        "minutes_played",
    ]
    available_numeric_cols = (
        df_filtered.select_dtypes(include=["number", "float", "int"]).columns.tolist()
    )
    focus_vars = [col for col in default_focus_vars if col in available_numeric_cols]
    metric_options = focus_vars if focus_vars else available_numeric_cols
    selected_metric = (
        st.selectbox("Variable principal", metric_options)
        if metric_options
        else None
    )

# -----------------------------------------------------------------------------
# Main layout
# -----------------------------------------------------------------------------
st.title("⚽ Explorador Interactivo – UEFA Champions League 2025")
st.caption(
    "Plataforma interactiva basada en Streamlit para documentar, visualizar y"
    " extraer conclusiones del rendimiento de jugadores de élite."
)

st.info(
    "Los filtros del panel lateral se aplican a todas las secciones. Usa la"
    " barra superior de pestañas para navegar entre los apartados del estudio."
)

# -----------------------------------------------------------------------------
# KPI cards
# -----------------------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Jugadores", format_metric(df_filtered.shape[0]))
with col2:
    st.metric("Equipos", format_metric(df_filtered[team_col].nunique()) if team_col else "N/D")
with col3:
    st.metric(
        "Posiciones",
        format_metric(df_filtered[position_col].nunique()) if position_col else "N/D",
    )
with col4:
    avg_label = (
        f"Promedio de {selected_metric}" if selected_metric else "Promedio variable principal"
    )
    avg_value = (
        format_metric(df_filtered[selected_metric].mean())
        if selected_metric and selected_metric in df_filtered.columns
        else "N/D"
    )
    st.metric(avg_label, avg_value)

# -----------------------------------------------------------------------------
# Tabs structure
# -----------------------------------------------------------------------------
(
    tab_overview,
    tab_story,
    tab_explorer,
    tab_visuals,
    tab_categorical,
    tab_outliers,
    tab_conclusions,
    tab_dictionary,
) = st.tabs(
    [
        "Visión General",
        "Dataset & Narrativa",
        "Explorador de Datos",
        "Visualizaciones",
        "Análisis Categórico",
        "Análisis de Atípicos",
        "Conclusiones",
        "Diccionario de Variables",
    ]
)

# -----------------------------------------------------------------------------
# Overview tab
# -----------------------------------------------------------------------------
with tab_overview:
    st.subheader("Contexto del Proyecto")
    st.markdown(
        """
        Este explorador resume el análisis exploratorio aplicado al dataset de
        rendimiento de jugadores de la UEFA Champions League 2025 (fase de
        grupos). Está diseñado para apoyar a analistas, cuerpos técnicos y
        estudiantes que deseen profundizar en la interpretación de variables
        técnicas, físicas y tácticas.
        """
    )

    st.markdown("### Principales indicadores")
    metrics_cols = st.columns(3)
    metric_map = {
        "goals": "Goles Promedio",
        "assists": "Asistencias Promedio",
        "passing_accuracy(%)": "Precisión de Pase (%)",
        "distance_covered(km/h)": "Distancia Recorrida (km/h)",
        "top_speed": "Velocidad Máxima (km/h)",
        "minutes_played": "Minutos Jugados",
    }

    for idx, (col_name, label) in enumerate(metric_map.items()):
        if col_name in df_filtered.columns:
            with metrics_cols[idx % 3]:
                st.metric(
                    label,
                    format_metric(df_filtered[col_name].mean()),
                    help=f"Media calculada sobre {df_filtered.shape[0]} jugadores filtrados.",
                )

    st.markdown("### Jugadores destacados")
    if not isinstance(selected_metric, str):
        st.warning("Selecciona una variable numérica válida para el ranking de jugadores.")
    elif selected_metric in df_filtered.columns:
        top_n_players = st.slider("Cantidad de jugadores a mostrar", 5, 30, 10, 5)
        top_player_columns: list[str] = []
        if isinstance(player_col, str):
            top_player_columns.append(player_col)
        top_player_columns.append(selected_metric)
        if isinstance(team_col, str):
            top_player_columns.append(team_col)
        top_players = (
            df_filtered.loc[df_filtered[selected_metric].notna(), top_player_columns]
            .sort_values(by=selected_metric, ascending=False)  # type: ignore[arg-type]
            .head(top_n_players)
        )
        st.dataframe(top_players, width="stretch")

        if isinstance(player_col, str):
            st.markdown("### Ficha interactiva del jugador")
            player_options = sorted(df_filtered[player_col].dropna().unique())
            if player_options:
                selected_player = st.selectbox("Jugador", player_options)
                player_snapshot = (
                    df_filtered[df_filtered[player_col] == selected_player]
                    .drop_duplicates(subset=[player_col])
                    .iloc[0]
                )
                st.dataframe(
                    player_snapshot.astype(str).to_frame("Valor"),
                    width="stretch",
                )
            else:
                st.info("No hay jugadores disponibles con los filtros actuales.")
    else:
        st.warning("La variable seleccionada no está disponible en el dataset filtrado.")

# -----------------------------------------------------------------------------
# Dataset story tab
# -----------------------------------------------------------------------------
with tab_story:
    st.subheader("Historia y narrativa del dataset")
    st.markdown(
        """
        Este proyecto documenta 908 jugadores y 50 variables provenientes de las fuentes
        oficiales de la UEFA Champions League 2025. La base combina métricas físicas,
        técnicas, contextuales y de rendimiento para cubrir todo el ciclo analítico descrito
        en el notebook original (Fases 1 a 4).
        """
    )

    story_cols = st.columns(3)
    with story_cols[0]:
        st.metric("Registros analizados", "908 jugadores")
        st.metric("Variables continuas", "9")
    with story_cols[1]:
        st.metric("Variables discretas", "34")
        st.metric("Variables categóricas", "7")
    with story_cols[2]:
        st.metric("Partidos considerados", "4 (fase de grupos)")
        st.metric("Técnicas de outliers", "IQR · Z-Score · IF · DBSCAN")

    st.markdown("### Resumen por fases del estudio")
    phase_details = {
        "Fase 1 – Construcción de base": "- Documentación exhaustiva de 50 variables\n- Verificación de calidad y trazabilidad (UEFA / tracking GPS)\n- Clasificación por tipo y relevancia táctica",
        "Fase 2 – EDA": "- Estadísticos descriptivos para 9 variables clave\n- Visualizaciones univariadas y bivariadas con matices tácticos\n- Hallazgos: delanteros concentran goles, MF lideran en pase, rango óptimo 24-28 años",
        "Fase 3 – Atípicos": "- Uso combinado de IQR, Z-Score, Isolation Forest y DBSCAN\n- Se decidió conservar outliers por representar a la élite competitiva\n- Identificación de perfiles únicos (suplentes de lujo, velocistas extremos)",
        "Fase 4 – Preparación": "- Evaluación formal de normalidad y escalamiento\n- Justificación de no imputar (patrón MNAR, valores informativos)\n- Codificación híbrida y escalamiento robusto para futuros modelos",
    }
    selected_phase = st.selectbox("Selecciona una fase para profundizar", list(phase_details.keys()))
    st.info(phase_details[selected_phase])

    st.markdown("### Decisiones metodológicas clave")
    decision_cols = st.columns(2)
    with decision_cols[0]:
        st.success(
            """
            **Valores faltantes (MNAR)**
            - No se imputan: representan roles y contextos reales.
            - Se sugiere usar indicadores binarios o modelos robustos.
            - Documentado con referencias (Little & Rubin, Van Buuren).
            """
        )
    with decision_cols[1]:
        st.warning(
            """
            **Tratamiento de outliers**
            - Se mantienen para preservar a la élite competitiva.
            - Útiles para scouting, estrategias tácticas y comparativas.
            - Se recomienda escalamiento robusto antes de modelar.
            """
        )

    st.markdown("### Ideas de análisis adicionales")
    with st.expander("¿Qué más se puede explorar con esta app?", expanded=True):
        st.markdown(
            """
            - Comparar perfiles físicos vs. técnicos por equipo o nacionalidad.
            - Identificar jugadores con carga física alta pero pocos minutos (riesgo de lesión).
            - Diseñar rankings personalizados combinando goles, asistencias y precisión.
            - Preparar features para clustering táctico o modelos predictivos.
            - Seguir evolución temporal incorporando partidos adicionales.
            """
        )

# -----------------------------------------------------------------------------
# Explorer tab
# -----------------------------------------------------------------------------
with tab_explorer:
    st.subheader("Vista tabular")
    available_columns = df_filtered.columns.tolist()
    columns_to_show = st.multiselect(
        "Columnas a visualizar",
        available_columns,
        default=available_columns[: min(len(available_columns), 12)],
    )
    st.dataframe(
        df_filtered[columns_to_show] if columns_to_show else df_filtered,
        width="stretch",
    )

    with st.expander("Resumen estadístico rápido"):
        st.dataframe(df_filtered.describe().T, width="stretch")

    st.download_button(
        label="Descargar datos filtrados",
        data=df_filtered.to_csv(index=False).encode("utf-8"),
        file_name="uefa_cl2025_filtrado.csv",
        mime="text/csv",
    )

# -----------------------------------------------------------------------------
# Visualisations tab
# -----------------------------------------------------------------------------
with tab_visuals:
    st.subheader("Visualizaciones interactivas")
    numeric_cols = df_filtered.select_dtypes(include=["number", "float", "int"]).columns.tolist()
    categorical_cols = df_filtered.select_dtypes(include=["object", "category"]).columns.tolist()

    if len(numeric_cols) < 1:
        st.warning("Se necesitan variables numéricas para generar gráficas.")
    else:
        col_x, col_y = st.columns(2)
        with col_x:
            default_x_idx = (
                numeric_cols.index(selected_metric)
                if isinstance(selected_metric, str) and selected_metric in numeric_cols
                else 0
            )
            x_axis = st.selectbox("Variable eje X", numeric_cols, index=default_x_idx)
        with col_y:
            default_y_idx = 1 if len(numeric_cols) > 1 else 0
            y_axis = st.selectbox("Variable eje Y", numeric_cols, index=default_y_idx)

        color_choices = ["(Sin color)"] + categorical_cols
        color_dimension = st.selectbox("Color por", color_choices, index=0)
        color_col = None if color_dimension == "(Sin color)" else color_dimension

        if HAS_STATSMODELS:
            add_trendline = st.toggle("Añadir línea de tendencia", value=False)
        else:
            st.toggle(
                "Añadir línea de tendencia",
                value=False,
                disabled=True,
                help="Instala 'statsmodels' para habilitar esta opción",
            )
            add_trendline = False
        scatter_fig = px.scatter(
            df_filtered,
            x=x_axis,
            y=y_axis,
            color=color_col,
            hover_data=df_filtered.columns,
            hover_name=player_col if player_col else None,
            trendline="ols" if add_trendline else None,
            title=f"{y_axis} vs {x_axis}",
            template="plotly_dark",
        )
        st.plotly_chart(scatter_fig, width="stretch")

        st.markdown("---")
        hist_variable = st.selectbox("Variable para histograma", numeric_cols)
        hist_fig = px.histogram(
            df_filtered,
            x=hist_variable,
            nbins=40,
            color=color_col,
            marginal="box",
            template="plotly_dark",
        )
        st.plotly_chart(hist_fig, width="stretch")

        st.markdown("### Boxplot por categoría")
        if categorical_cols:
            box_col1, box_col2 = st.columns(2)
            with box_col1:
                category_for_box = st.selectbox("Categoría", categorical_cols)
            with box_col2:
                metric_for_box = st.selectbox(
                    "Variable numérica",
                    numeric_cols,
                    index=numeric_cols.index(selected_metric)
                    if isinstance(selected_metric, str) and selected_metric in numeric_cols
                    else 0,
                )

            box_fig = px.box(
                df_filtered,
                x=category_for_box,
                y=metric_for_box,
                color=category_for_box,
                points="all",
                template="plotly_dark",
            )
            st.plotly_chart(box_fig, width="stretch")
        else:
            st.info("No se detectaron variables categóricas para generar boxplots.")

        st.markdown("### Matriz de dispersión multivariable")
        scatter_matrix_vars = st.multiselect(
            "Selecciona hasta 5 variables",
            numeric_cols,
            default=numeric_cols[: min(len(numeric_cols), 4)],
        )
        if len(scatter_matrix_vars) > 5:
            scatter_matrix_vars = scatter_matrix_vars[:5]
        if len(scatter_matrix_vars) >= 2:
            scatter_matrix_data_cols = list(scatter_matrix_vars)
            scatter_matrix_color = None
            if color_col and color_col in df_filtered.columns:
                if color_col not in scatter_matrix_data_cols:
                    scatter_matrix_data_cols.append(color_col)
                scatter_matrix_color = color_col

            scatter_matrix_fig = px.scatter_matrix(
                df_filtered[scatter_matrix_data_cols],
                dimensions=scatter_matrix_vars,
                color=scatter_matrix_color,
                template="plotly_dark",
            )
            st.plotly_chart(scatter_matrix_fig, width="stretch")
        else:
            st.caption("Selecciona al menos dos variables para ver la matriz de dispersión.")

        st.markdown("### Matriz de correlación")
        corr_vars = st.multiselect(
            "Variables para la correlación",
            numeric_cols,
            default=numeric_cols[: min(len(numeric_cols), 6)],
        )
        if len(corr_vars) >= 2:
            corr_matrix = df_filtered[corr_vars].corr(numeric_only=True)
            corr_fig = px.imshow(
                corr_matrix,
                text_auto=True,
                color_continuous_scale="RdBu",
                zmin=-1,
                zmax=1,
                template="plotly_dark",
            )
            st.plotly_chart(corr_fig, width="stretch")
        else:
            st.caption("Selecciona al menos dos variables para construir la matriz de correlación.")

# -----------------------------------------------------------------------------
# Categorical analysis tab
# -----------------------------------------------------------------------------
with tab_categorical:
    st.subheader("Análisis categórico interactivo")
    categorical_cols = df_filtered.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = df_filtered.select_dtypes(include=["number", "float", "int"]).columns.tolist()

    if not categorical_cols or not numeric_cols:
        st.info("Se requieren variables categóricas y numéricas para generar este análisis.")
    else:
        group_col = st.selectbox("Agrupar por", categorical_cols)
        metrics_default = [m for m in ["goals", "assists", "minutes_played"] if m in numeric_cols]
        metrics_selected = st.multiselect(
            "Variables numéricas a resumir",
            numeric_cols,
            default=metrics_default if metrics_default else numeric_cols[:1],
        )
        agg_options = {
            "Promedio": "mean",
            "Mediana": "median",
            "Suma": "sum",
            "Máximo": "max",
        }
        agg_choice_label = st.selectbox("Estadístico", list(agg_options.keys()))
        agg_choice = agg_options[agg_choice_label]

        if not metrics_selected:
            st.warning("Selecciona al menos una variable numérica para agregar.")
        else:
            grouped = (
                df_filtered.groupby(group_col)[metrics_selected]
                .agg(agg_choice)
                .reset_index()
                .sort_values(by=metrics_selected[0], ascending=False)
            )
            st.dataframe(grouped, width="stretch")

            chart_metric = st.selectbox(
                "Variable para el gráfico",
                metrics_selected,
                index=0,
            )
            chart_fig = px.bar(
                grouped,
                x=group_col,
                y=chart_metric,
                color=group_col,
                title=f"{agg_choice_label} de {chart_metric} por {group_col}",
                template="plotly_dark",
            )
            st.plotly_chart(chart_fig, width="stretch")

            with st.expander("Tabla dinámica (pivot)"):
                pivot_metric = st.selectbox(
                    "Métrica para pivot",
                    metrics_selected,
                    key="pivot_metric",
                )
                pivot_table = (
                    df_filtered.pivot_table(
                        index=group_col,
                        values=pivot_metric,
                        aggfunc=agg_choice,
                    )
                    .sort_values(by=pivot_metric, ascending=False)
                    .reset_index()
                )
                st.dataframe(pivot_table, width="stretch")

        st.caption(
            "Usa este módulo para construir rankings por equipo, posición, nacionalidad u"
            " otras variables categóricas disponibles en el dataset."
        )

# -----------------------------------------------------------------------------
# Outliers tab
# -----------------------------------------------------------------------------
with tab_outliers:
    st.subheader("Detección interactiva de atípicos")
    st.markdown(
        """
        En esta sección puedes contrastar distintos enfoques para detectar valores atípicos.
        Usa metodologías paramétricas (Z-Score), robustas (IQR) o algoritmos basados
        en aprendizaje automático (Isolation Forest y DBSCAN) para identificar perfiles
        singulares justificando por qué se conservan o eliminan.
        """
    )

    numeric_cols = df_filtered.select_dtypes(include=["number", "float", "int"]).columns.tolist()
    if not numeric_cols:
        st.warning("Se requieren variables numéricas para analizar atípicos.")
    else:
        technique_catalog = [
            ("Z-Score (paramétrico)", "zscore"),
            ("IQR (robusto)", "iqr"),
        ]
        has_multivariate_algorithms = (
            HAS_SKLEARN
            and IsolationForest is not None
            and DBSCAN is not None
            and StandardScaler is not None
        )
        if has_multivariate_algorithms:
            technique_catalog.extend(
                [
                    ("Isolation Forest (multivariante)", "iforest"),
                    ("DBSCAN (densidad)", "dbscan"),
                ]
            )
        else:
            st.info(
                "Instala 'scikit-learn' para habilitar Isolation Forest y DBSCAN"  # pragma: no cover
            )

        technique_labels = [label for label, _ in technique_catalog]
        technique_map = {label: code for label, code in technique_catalog}
        technique_choice = st.selectbox("Técnica de detección", technique_labels)
        technique_code = technique_map[technique_choice]

        technique_explanations = {
            "zscore": "Ideal para distribuciones aproximadamente normales. Marca atípicos cuando el puntaje Z supera un umbral absoluto.",
            "iqr": "Robusta frente a colas pesadas. Usa cuartiles y el rango intercuartílico para aislar valores extremos.",
            "iforest": "Modelo de aislamiento basado en árboles aleatorios. Detecta anomalías multivariantes controlando la proporción esperada.",
            "dbscan": "Algoritmo basado en densidad: puntos con pocas vecindades cercanas se etiquetan como ruido (atípicos).",
        }
        st.caption(technique_explanations.get(technique_code, ""))

        outliers_df: pd.DataFrame | None = None
        criteria_text = ""

        if technique_code in {"zscore", "iqr"}:
            metric_default_idx = (
                numeric_cols.index(selected_metric)
                if isinstance(selected_metric, str) and selected_metric in numeric_cols
                else 0
            )
            metric_for_outliers = st.selectbox(
                "Variable objetivo",
                numeric_cols,
                index=metric_default_idx,
            )
            series = df_filtered[metric_for_outliers].dropna()

            if series.empty:
                st.warning("No hay datos disponibles para la variable seleccionada.")
            else:
                if technique_code == "zscore":
                    threshold = st.slider("Umbral |Z|", 1.0, 4.0, 3.0, 0.5)
                    mean_val = series.mean()
                    std_val = series.std(ddof=0)
                    if std_val == 0:
                        st.warning("La variable seleccionada no presenta variación.")
                    else:
                        z_scores = (series - mean_val) / std_val
                        mask_outliers = z_scores.abs() >= threshold
                        outliers_df = (
                            df_filtered.loc[z_scores.index[mask_outliers]]
                            .assign(z_score=z_scores[mask_outliers])
                            .sort_values("z_score", ascending=False)
                        )
                        criteria_text = f"|Z| ≥ {threshold}"
                else:  # IQR
                    iqr_multiplier = st.slider("Multiplicador IQR", 1.0, 3.0, 1.5, 0.1)
                    q1 = series.quantile(0.25)
                    q3 = series.quantile(0.75)
                    iqr = q3 - q1
                    lower = q1 - iqr_multiplier * iqr
                    upper = q3 + iqr_multiplier * iqr
                    mask_outliers = (series < lower) | (series > upper)
                    outliers_df = (
                        df_filtered.loc[series.index[mask_outliers]]
                        .assign(limite_inferior=lower, limite_superior=upper)
                        .sort_values(metric_for_outliers, ascending=False)
                    )
                    criteria_text = f"< {lower:.2f} o > {upper:.2f}"

        elif (
            technique_code == "iforest"
            and has_multivariate_algorithms
            and IsolationForest is not None
        ):
            default_features = (
                numeric_cols[: min(len(numeric_cols), 3)]
                if numeric_cols
                else []
            )
            feature_selection = st.multiselect(
                "Variables para el modelo",
                numeric_cols,
                default=default_features,
            )
            if len(feature_selection) < 2:
                st.warning("Selecciona al menos dos variables para Isolation Forest.")
            else:
                contamination = st.slider(
                    "Proporción esperada de anomalías",
                    0.01,
                    0.20,
                    0.05,
                    0.01,
                )
                subset = df_filtered[feature_selection].dropna()
                if subset.empty:
                    st.info("No hay registros completos para las variables elegidas.")
                else:
                    model = IsolationForest(
                        contamination=contamination,
                        random_state=42,
                    )
                    predictions = model.fit_predict(subset)
                    anomaly_scores = model.decision_function(subset)
                    mask_outliers = predictions == -1
                    outlier_idx = subset.index[mask_outliers]
                    outliers_df = (
                        df_filtered.loc[outlier_idx]
                        .assign(anomaly_score=anomaly_scores[mask_outliers])
                        .sort_values("anomaly_score")
                    )
                    criteria_text = f"Isolation Forest (contaminación {contamination:.0%})"

                    if len(feature_selection) >= 2:
                        viz_cols = feature_selection[:2]
                        viz_df = subset.assign(
                            Estado=np.where(mask_outliers, "Atípico", "Regular")
                        )
                        fig_iforest = px.scatter(
                            viz_df,
                            x=viz_cols[0],
                            y=viz_cols[1],
                            color="Estado",
                            title="Proyección 2D de Isolation Forest",
                            template="plotly_dark",
                        )
                        st.plotly_chart(fig_iforest, width="stretch")

        elif (
            technique_code == "dbscan"
            and has_multivariate_algorithms
            and DBSCAN is not None
            and StandardScaler is not None
        ):
            default_features = (
                numeric_cols[: min(len(numeric_cols), 3)]
                if numeric_cols
                else []
            )
            feature_selection = st.multiselect(
                "Variables para DBSCAN",
                numeric_cols,
                default=default_features,
                key="dbscan_features",
            )
            if len(feature_selection) < 2:
                st.warning("Selecciona al menos dos variables para DBSCAN.")
            else:
                eps_value = st.slider("Radio (eps)", 0.1, 5.0, 1.5, 0.1)
                min_samples = st.slider("Vecinos mínimos", 3, 15, 5)
                subset = df_filtered[feature_selection].dropna()
                if subset.empty:
                    st.info("No hay registros completos para las variables elegidas.")
                else:
                    scaler = StandardScaler()
                    scaled_subset = scaler.fit_transform(subset)
                    clustering = DBSCAN(eps=eps_value, min_samples=min_samples).fit(scaled_subset)
                    labels = clustering.labels_
                    mask_outliers = labels == -1
                    outlier_idx = subset.index[mask_outliers]
                    outliers_df = (
                        df_filtered.loc[outlier_idx]
                        .assign(cluster="Ruido")
                    )
                    criteria_text = f"Ruido detectado con eps={eps_value:.1f}, min_samples={min_samples}"

                    viz_cols = feature_selection[:2]
                    viz_df = subset.assign(
                        Estado=np.where(mask_outliers, "Atípico", "Regular")
                    )
                    fig_dbscan = px.scatter(
                        viz_df,
                        x=viz_cols[0],
                        y=viz_cols[1],
                        color="Estado",
                        title="Proyección 2D de DBSCAN",
                        template="plotly_dark",
                    )
                    st.plotly_chart(fig_dbscan, width="stretch")

        if outliers_df is not None:
            st.metric("Atípicos detectados", format_metric(outliers_df.shape[0]))
            if criteria_text:
                st.caption(f"Criterio aplicado: {criteria_text}")
            st.dataframe(outliers_df, width="stretch")
            st.caption(
                "Interpreta los perfiles detectados antes de decidir si se excluyen o"
                " se mantienen: en este proyecto se conservan para describir a la"
                " élite competitiva."
            )
        else:
            st.info("Ajusta los parámetros para identificar atípicos con la técnica seleccionada.")

# -----------------------------------------------------------------------------
# Conclusions tab
# -----------------------------------------------------------------------------
with tab_conclusions:
    st.subheader("Conclusiones principales")
    conclusions_text = textwrap.dedent(
        """
        ### Fase 1 – Calidad y trazabilidad
        - 908 jugadores × 50 variables respaldadas por fuentes oficiales UEFA (tracking y scouting).
        - Documentación exhaustiva del diccionario y clasificación por tipo (continuas, discretas, categóricas) para garantizar reproducibilidad.
        - Auditoría de valores faltantes y verificación de consistencia entre métricas físicas/tácticas.

        ### Fase 2 – Lectura exploratoria
        - Delanteros aportan >70 % de los goles mientras los mediocampistas maximizan la precisión de pase (>85 %).
        - El rango etario 24-28 años equilibra carga física y productividad ofensiva; extremos jóvenes destacan en velocidad.
        - Variables físicas (distancia recorrida, top speed) explican diferencias tácticas por equipo y determinan riesgos de fatiga.

        ### Fase 3 – Valores atípicos multitécnica
        - Z-Score permite detectar desbordes paramétricos rápidos; IQR confirma robustez frente a colas pesadas.
        - Isolation Forest y DBSCAN aíslan perfiles multivariantes (suplentes explosivos, velocistas extremos) sin asumir distribuciones.
        - Decisión: conservar atípicos porque describen la élite competitiva y aportan valor en scouting y planificación táctica.

        ### Fase 4 – Preparación y gobernanza
        - No se imputan faltantes (patrón MNAR); se sugiere crear indicadores o modelos tolerantes a ausencia.
        - Codificación híbrida (One-Hot para baja cardinalidad, Label/Target para alta) y escalamiento robusto (RobustScaler) antes de modelar.
        - Se dispone de `df_robust` y `df_encoded` como bases listas para machine learning.

        ### Hallazgos complementarios
        - La app interactiva permite comparar técnicas de outliers y comprender el impacto táctico jugador a jugador.
        - Rankings custom de goles+asistencias y métricas físicas facilitan detectar roles ocultos (impacto vs. minutos).

        ### Próximos pasos sugeridos
        1. Modelado predictivo (xG, contribución ofensiva) usando las bases escaladas.
        2. Clustering táctico avanzado (K-Means, HDBSCAN) con features físicos/técnicos.
        3. Actualizar la base con fases KO para evaluar evolución temporal y fatiga acumulada.
        """
    )
    st.markdown(conclusions_text)

    st.success(
        "Estas conclusiones derivan del notebook original y están alineadas con"
        " la documentación académica generada para el proyecto."
    )

# -----------------------------------------------------------------------------
# Data dictionary tab
# -----------------------------------------------------------------------------
with tab_dictionary:
    st.subheader("Diccionario de variables")
    dict_df = get_dictionary()
    if dict_df.empty:
        st.info("No se encontró `tabla_descriptiva_variables.csv` en el directorio actual.")
    else:
        search_term = st.text_input("Buscar variable")
        dict_filtered = dict_df.copy()
        if search_term:
            mask = dict_filtered["Variable"].str.contains(search_term, case=False, na=False)
            dict_filtered = dict_filtered[mask]
        st.dataframe(dict_filtered, width="stretch")

        st.download_button(
            "Descargar diccionario",
            dict_filtered.to_csv(index=False).encode("utf-8"),
            file_name="diccionario_variables.csv",
            mime="text/csv",
        )

# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------
st.markdown("---")
st.caption(
    "Construido con Streamlit • Emmanuel Valbuena • UEFA Champions League 2025"
)
