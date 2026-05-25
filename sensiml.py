# -*- coding: utf-8 -*-
"""
sensiml.py
=========================
sensitivity interpretable ML for xgb_sculpt

Framework para análisis global de sensibilidad, interacciones
e interpretabilidad de modelos tabulares de machine learning.

El módulo integra tres enfoques complementarios:

1. Sensibilidad global basada en Sobol–Saltelli
2. Sensibilidad observacional basada en datos
3. Interpretabilidad mediante Valores de Shapley Condicionales

Diseñado específicamente para pipelines sklearn/XGBoost procedentes
de ``xgb_sculpt``, aunque gran parte de la API funciona con cualquier
modelo tipo caja negra.

---------------------------------------------------------------------------
ARQUITECTURA
---------------------------------------------------------------------------

NIVEL 1 — Funciones genéricas independientes del modelo
-------------------------------------------------------

Estas funciones trabajan directamente sobre funciones matemáticas o
callables arbitrarios:

    sobol_from_function()
        → Índices Sobol S1 y ST

    sobol_second_order_from_function()
        → Índices Sobol S2 (interacciones por pares)

    sobol_first_order_from_data()
        → Sensibilidad observacional basada en E[Y | Xi]

Estas funciones son además la base de los tests matemáticos del módulo.

NIVEL 2 — Wrappers para pipelines sklearn/XGBoost
-------------------------------------------------

Funciones adaptadas a pipelines con preprocesado y modelo:

    sobol_indices()
        → Wrapper de sobol_from_function()

    sobol_second_order()
        → Wrapper de sobol_second_order_from_function()

    sobol_data_indices()
        → Wrapper de sobol_first_order_from_data()

    conditional_shapley_effects()
        → Valores de Shapley Condicionales usando shaprpy

VISUALIZACIÓN
-------------

    sobol_plot()
        → Barras S1/ST con interacciones

    sobol_interaction_heatmap()
        → Heatmap de índices S2

    plot_shapley_effects()
        → Visualizaciones estándar SHAP

VALIDACIÓN MATEMÁTICA
---------------------

    validate_mathematical_properties()
    test_linear_model()
    test_interaction_model()
    test_second_order()
    test_data_driven()
    run_all_tests()

---------------------------------------------------------------------------
FILOSOFÍA DEL MÓDULO
---------------------------------------------------------------------------

El objetivo del framework es separar claramente tres conceptos distintos:

- Sensibilidad causal intrínseca del modelo
- Dependencias estadísticas observadas en los datos
- Explicaciones locales y globales condicionadas

Por ello se implementan simultáneamente:

- Sobol clásico (rompe correlaciones)
- Sobol observacional basado en datos reales
- Shapley condicional respetando dependencias estadísticas

Esto permite distinguir:

    causalidad      vs correlación
    sensibilidad    vs predictibilidad
    efectos directos vs interacciones

---------------------------------------------------------------------------
DEPENDENCIAS
---------------------------------------------------------------------------

Base:
    pip install SALib>=1.5 pandas numpy matplotlib scikit-learn

Modelos:
    pip install xgboost

Shapley condicional:
    pip install shaprpy shap

Requiere R instalado junto con su módulo shapr
---------------------------------------------------------------------------
ADVERTENCIA
---------------------------------------------------------------------------

Este módulo implementa técnicas avanzadas de análisis de sensibilidad
e interpretabilidad estadística.

Está dirigido a usuarios con experiencia en:

- machine learning tabular
- teoría Sobol/Shapley
- inferencia estadística
- pipelines sklearn
- análisis de correlaciones
- interpretación de modelos

No es una herramienta "plug-and-play" para usuarios sin
conocimientos sobre análisis de sensibilidad global e
interpretabilidad de modelos.

---------------------------------------------------------------------------
COMPATIBILIDAD
---------------------------------------------------------------------------

Desarrollado y validado principalmente con:

- Python >= 3.10
- scikit-learn >= 1.3
- SALib >= 1.5
- shap >= 0.44
- shaprpy >= 0.3
- xgboost >= 2.0

---------------------------------------------------------------------------
Nota
---------------------------------------------------------------------------

Uso orientado a research tooling de ML.

Uso básico — pipeline
---------------------
    import xgb_sculpt as xs
    import sobol_sensitivity as sob

    models, cv_results, final_model = xs.run_pipe(X, y, ...)
    s1st = sob.sobol_indices(final_model, X_sel)
    s2   = sob.sobol_second_order(final_model, X_sel)

    fig1, ax1 = sob.sobol_plot(s1st)
    fig2, ax2 = sob.sobol_interaction_heatmap(s2)
"""

import logging
import warnings

from pathlib import Path
from typing import Any
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# =============================================================================
# Logging — delega la configuración al cliente importador
# =============================================================================
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# =============================================================================
# Constantes
# =============================================================================
RANDOM_STATE = 42
_VALID_PLOTS = {"bar", "beeswarm", "waterfall", "heatmap", "scatter2d"}

# =============================================================================
# API pública
# =============================================================================
__all__ = [
    # Nivel 1: genéricas basadas en modelo
    "sobol_from_function",
    "sobol_second_order_from_function",
    # Nivel 1: basada en datos observados
    "sobol_first_order_from_data",
    # Nivel 2: wrappers pipeline
    "sobol_indices",
    "sobol_second_order",
    "sobol_data_indices",
    # Visualización
    "sobol_plot",
    "sobol_interaction_heatmap",
    # Validación
    "validate_mathematical_properties",
    "test_linear_model",
    "test_interaction_model",
    "test_second_order",
    "test_data_driven",
    "run_all_tests",
    # Shap effects
    "conditional_shapley_effects",
    "plot_shapley_effects",
]


# =============================================================================
# HELPERS INTERNOS
# =============================================================================

def _import_salib():
    """Importa SALib utilizando el módulo moderno 'sample.sobol'."""
    try:
        from SALib.sample import sobol as sobol_sampler
        from SALib.analyze import sobol as sobol_analyze
        return sobol_sampler, sobol_analyze
    except ImportError as exc:
        raise ImportError(
            "SALib es necesario para este módulo.\n"
            "Instalar con: pip install 'SALib>=1.5'"
        ) from exc


def _validate_n_samples(n_samples):
    """Valida que n_samples sea una potencia de 2."""
    if n_samples < 1 or (n_samples & (n_samples - 1)) != 0:
        raise ValueError(
            f"n_samples debe ser una potencia de 2 (ej. 512, 1024, 2048). "
            f"Recibido: {n_samples}"
        )


def _sobol_sample(sobol_sampler, problem, n_samples, calc_second_order, random_state):
    """Ejecuta el muestreo de Sobol/Saltelli compatible con SALib>=1.5."""
    return sobol_sampler.sample(
        problem, n_samples,
        calc_second_order=calc_second_order,
        seed=random_state,
    )


def _sobol_analyze(sobol_analyze, problem, Y, calc_second_order, conf_level, random_state):
    """Ejecuta el análisis de Sobol con protección robusta para el parámetro 'seed'."""
    try:
        return sobol_analyze.analyze(
            problem, Y,
            calc_second_order=calc_second_order,
            conf_level=conf_level,
            seed=random_state,
            print_to_console=False,
        )
    except TypeError:
        # Fallback si la versión de análisis de la API de SALib no acepta la firma con semilla
        return sobol_analyze.analyze(
            problem, Y,
            calc_second_order=calc_second_order,
            conf_level=conf_level,
            print_to_console=False,
        )


def _extract_pipeline_components(final_model, X, bounds_percentile):
    """Extrae componentes de un pipeline y resuelve colisiones de nombres de variables."""
    pipeline = final_model[0] if isinstance(final_model, (list, tuple)) else final_model
    
    prep  = pipeline.named_steps["prep"]
    model = pipeline.named_steps["model"]

    Xt = prep.transform(X)

    try:
        import scipy.sparse as sp
        if sp.issparse(Xt):
            Xt = Xt.toarray()
    except ImportError:
        pass

    # Extraer nombres base limpios
    raw_names = [n.split("__", 1)[-1] for n in prep.get_feature_names_out()]
    
    # Reparar proactivamente nombres duplicados para evitar que SALib colapse
    feature_names = []
    seen = {}
    for name in raw_names:
        if name in seen:
            seen[name] += 1
            feature_names.append(f"{name}_{seen[name]}")
        else:
            seen[name] = 1
            feature_names.append(name)

    bounds = []
    for i in range(Xt.shape[1]):
        col = Xt[:, i]
        lo  = float(np.percentile(col, 1))  if bounds_percentile else float(col.min())
        hi  = float(np.percentile(col, 99)) if bounds_percentile else float(col.max())
        if lo == hi:
            lo, hi = lo - 1e-6, hi + 1e-6
        bounds.append([lo, hi])

    return model.predict, bounds, feature_names


def _clip_and_build_s1st(Si, feature_names):
    """Construye DataFrame de S1/ST clipeando negativos para coherencia."""
    s1_clip = np.maximum(Si["S1"], 0.0)
    st_clip = np.maximum(Si["ST"], 0.0)
    return (
        pd.DataFrame({
            "feature":     feature_names,
            "S1":          s1_clip,
            "S1_conf":     Si["S1_conf"],
            "ST":          st_clip,
            "ST_conf":     Si["ST_conf"],
            "interaction": np.maximum(st_clip - s1_clip, 0.0),
        })
        .sort_values("ST", ascending=False)
        .reset_index(drop=True)
    )


# =============================================================================
# NIVEL 1 — FUNCIONES GENÉRICAS
# =============================================================================

def sobol_from_function(fn, bounds, names=None, n_samples=1024, conf_level=0.95, random_state=RANDOM_STATE):
    """Calcula los índices de Sobol S1 y ST para cualquier función caja negra."""
    sobol_sampler, sobol_analyze = _import_salib()
    _validate_n_samples(n_samples)

    D = len(bounds)
    if names is None:
        names = [f"X{i}" for i in range(D)]
    if len(names) != D:
        raise ValueError(f"len(names)={len(names)} debe coincidir con len(bounds)={D}")

    problem = {"num_vars": D, "names": names, "bounds": bounds}

    logger.info("sobol_from_function | %d variables, N=%d", D, n_samples)

    param_values = _sobol_sample(sobol_sampler, problem, n_samples,
                                 calc_second_order=False,
                                 random_state=random_state)

    Y = np.asarray(fn(param_values), dtype=float)
    var_Y = float(np.var(Y))

    if var_Y < 1e-10:
        warnings.warn("La varianza de Y es prácticamente cero.", UserWarning, stacklevel=2)

    Si = _sobol_analyze(sobol_analyze, problem, Y,
                        calc_second_order=False,
                        conf_level=conf_level,
                        random_state=random_state)

    return _clip_and_build_s1st(Si, names)


def sobol_second_order_from_function(fn, bounds, names=None, n_samples=2048, conf_level=0.95, top_n=None, threshold_S2=0.0, random_state=RANDOM_STATE):
    """Calcula los índices de segundo orden S2 (interacciones por pares)."""
    sobol_sampler, sobol_analyze = _import_salib()
    _validate_n_samples(n_samples)

    D = len(bounds)
    if names is None:
        names = [f"X{i}" for i in range(D)]
    if len(names) != D:
        raise ValueError(f"len(names)={len(names)} debe coincidir con len(bounds)={D}")

    problem = {"num_vars": D, "names": names, "bounds": bounds}

    param_values = _sobol_sample(sobol_sampler, problem, n_samples,
                                 calc_second_order=True,
                                 random_state=random_state)

    Y = np.asarray(fn(param_values), dtype=float)

    Si = _sobol_analyze(sobol_analyze, problem, Y,
                        calc_second_order=True,
                        conf_level=conf_level,
                        random_state=random_state)

    S2      = np.array(Si["S2"])
    S2_conf = np.array(Si["S2_conf"])

    rows = []
    for i in range(D):
        for j in range(i + 1, D):
            s2_val  = float(max(S2[i, j],      0.0))
            s2_conf = float(S2_conf[i, j])
            rows.append({
                "feature_i": names[i],
                "feature_j": names[j],
                "S2":        s2_val,
                "S2_conf":   s2_conf,
            })

    # Reparación del crash: se definen las columnas de antemano si la lista está vacía (D=1)
    result = (
        pd.DataFrame(rows, columns=["feature_i", "feature_j", "S2", "S2_conf"])
        .sort_values("S2", ascending=False)
        .reset_index(drop=True)
    )

    if threshold_S2 > 0.0:
        result = result[result["S2"] >= threshold_S2].reset_index(drop=True)
    if top_n is not None:
        result = result.head(top_n).reset_index(drop=True)

    return result


# =============================================================================
# NIVEL 2 — WRAPPERS
# =============================================================================

def sobol_indices(final_model, X, n_samples=1024, conf_level=0.95, bounds_percentile=True, random_state=RANDOM_STATE):
    """Calcula los índices S1 y ST para un pipeline sklearn/XGBoost."""
    predict_fn, bounds, feature_names = _extract_pipeline_components(final_model, X, bounds_percentile)
    return sobol_from_function(fn=predict_fn, bounds=bounds, names=feature_names, n_samples=n_samples, conf_level=conf_level, random_state=random_state)


def sobol_second_order(final_model, X, n_samples=2048, conf_level=0.95, bounds_percentile=True, top_n=None, threshold_S2=0.0, random_state=RANDOM_STATE):
    """Calcula los índices S2 (interacciones por pares) para un pipeline."""
    predict_fn, bounds, feature_names = _extract_pipeline_components(final_model, X, bounds_percentile)
    return sobol_second_order_from_function(fn=predict_fn, bounds=bounds, names=feature_names, n_samples=n_samples, conf_level=conf_level, top_n=top_n, threshold_S2=threshold_S2, random_state=random_state)


# =============================================================================
# VISUALIZACIÓN
# =============================================================================

def sobol_plot(sobol_df, threshold_ST=0.05, conf_level=0.95, save_path=None):
    """Visualiza S1 y ST como barras horizontales apiladas."""
    if sobol_df.empty:
        warnings.warn("El DataFrame de Sobol está vacío.")
        return plt.subplots()

    df    = sobol_df.copy().sort_values("ST", ascending=True).reset_index(drop=True)
    n     = len(df)
    y_pos = np.arange(n)

    fig, ax = plt.subplots(figsize=(9, max(4, n * 0.45)))

    relevant  = df["ST"].values >= threshold_ST
    color_s1  = ["#3a7ebf" if r else "#cccccc" for r in relevant]
    color_int = ["#e8873a" if r else "#e0e0e0" for r in relevant]

    ax.barh(y_pos, df["S1"], color=color_s1, height=0.6, label="S1 — Efecto directo")
    ax.barh(y_pos, df["interaction"], left=df["S1"], color=color_int, height=0.6, label="ST − S1 — Interacciones")

    conf_pct = int(round(conf_level * 100))
    ax.errorbar(df["ST"], y_pos, xerr=df["ST_conf"], fmt="none", color="#555555", capsize=3, linewidth=1, label=f"IC {conf_pct}% de ST")
    ax.axvline(threshold_ST, color="red", linestyle="--", linewidth=1.2, alpha=0.7, label=f"Umbral ST = {threshold_ST}")

    labels = [f"{row['feature']}{ ' ✗' if row['ST'] < threshold_ST else ''}" for _, row in df.iterrows()]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)

    max_x = float((df.loc[relevant, "ST"] + df.loc[relevant, "ST_conf"]).max()) if relevant.any() else float(df["ST"].max()) if n > 0 else 0.0
    ax.set_xlim(left=0, right=max(max_x * 1.30 + 0.01, threshold_ST * 1.5))

    for i, row in df.iterrows():
        if row["ST"] >= threshold_ST:
            ax.text(row["ST"] + row["ST_conf"] + 0.005, i, f"ST={row['ST']:.3f}", va="center", fontsize=7.5, color="#333333")

    ax.set_xlabel("Índice de Sobol", fontsize=10)
    ax.set_title("Análisis de sensibilidad de Sobol", fontsize=10)
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig, ax


def sobol_interaction_heatmap(s2_df, threshold_S2=0.0, save_path=None):
    """Visualiza los índices S2 como mapa de calor de forma segura frente a filtrados estrictos o vacíos."""
    # Reparación del crash: Manejo seguro si no quedan datos que mostrar por filtrado extremo
    if s2_df.empty:
        warnings.warn("El DataFrame de interacciones está vacío. No hay datos válidos para el heatmap.")
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.text(0.5, 0.5, "Sin interacciones significativas", ha="center", va="center", color="gray", fontsize=10)
        ax.set_axis_off()
        return fig, ax

    all_features = list(dict.fromkeys(s2_df["feature_i"].tolist() + s2_df["feature_j"].tolist()))
    D     = len(all_features)
    idx = {f: i for i, f in enumerate(all_features)}

    mat = np.full((D, D), np.nan)
    for _, row in s2_df.iterrows():
        i = idx[row["feature_i"]]
        j = idx[row["feature_j"]]
        val = row["S2"] if row["S2"] >= threshold_S2 else np.nan
        mat[i, j] = val
        mat[j, i] = val

    fig, ax = plt.subplots(figsize=(max(6, D * 0.55), max(5, D * 0.50)))

    mask = np.tril(np.ones((D, D), dtype=bool), k=-1)
    mat_plot = mat.copy()
    mat_plot[mask] = np.nan

    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad(color="#f5f5f5")

    vmax = float(np.nanmax(mat_plot)) if not np.all(np.isnan(mat_plot)) else 1.0
    im   = ax.imshow(mat_plot, cmap=cmap, vmin=0, vmax=vmax, aspect="auto")

    for i in range(D):
        for j in range(i, D):
            val = mat_plot[i, j]
            if i == j or np.isnan(val):
                continue
            text_color = "white" if val > vmax * 0.6 else "#333333"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=8, color=text_color)

    for i in range(D):
        ax.text(i, i, all_features[i], ha="center", va="center", fontsize=8.5, fontweight="bold", color="#333333",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#e8e8e8", edgecolor="none"))

    ax.set_xticks(range(D))
    ax.set_xticklabels(all_features, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(D))
    ax.set_yticklabels(all_features, fontsize=8)

    cbar = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label("S2 — Índice de interacción", fontsize=9)
    ax.set_title("Índices de interacción de Sobol (S2)", fontsize=10)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig, ax


# =============================================================================
# VALIDACIÓN MATEMÁTICA Y TESTS
# =============================================================================

def validate_mathematical_properties(sobol_df, tol=1e-6, raise_on_fail=True):
    """Verifica las propiedades invariantes de los índices de Sobol."""
    failures = []
    def check(label, ok, detail=""):
        if not ok:
            failures.append(label)
            if raise_on_fail: raise AssertionError(f"{label.strip()} — {detail}")

    p1_mask = sobol_df["ST"] >= sobol_df["S1"] - tol
    check("P1 — S1 <= ST:            ", p1_mask.all(), f"Violación en: {sobol_df.loc[~p1_mask,'feature'].tolist()}")
    diff = np.abs(sobol_df["S1"] + sobol_df["interaction"] - sobol_df["ST"])
    check("P2 — S1 + interaction==ST:", (diff <= tol).all(), f"Diferencia máxima: {diff.max():.2e}")
    check("P3 — S1>=0, ST>=0:        ", (sobol_df["S1"] >= 0).all() and (sobol_df["ST"] >= 0).all())
    suma = sobol_df["S1"].sum()
    check("P4 — ΣS1 <= 1.1:          ", suma <= 1.1, f"ΣS1 = {suma:.3f}")
    return len(failures) == 0


def test_linear_model(n_samples=1024, verbose=True):
    """Test con Y = 3·X1 + X2 — valores teóricos exactos."""
    sob = sobol_from_function(fn=lambda X: 3*X[:,0] + X[:,1], bounds=[[0,1]]*4, names=["X1","X2","X3","X4"], n_samples=n_samples, random_state=RANDOM_STATE)
    atol = 0.05
    assert abs(sob.loc[sob["feature"]=="X1","S1"].values[0] - 0.90) < atol
    assert sob.loc[sob["feature"]=="X3","ST"].values[0] < atol
    validate_mathematical_properties(sob, raise_on_fail=True)
    return True


def test_interaction_model(n_samples=1024, verbose=True):
    """Test con Y = X1·X2 — interacciones detectables."""
    sob = sobol_from_function(fn=lambda X: X[:,0] * X[:,1], bounds=[[0,1]]*4, names=["X1","X2","X3","X4"], n_samples=n_samples, random_state=RANDOM_STATE)
    atol = 0.06
    assert abs(sob.loc[sob["feature"]=="X1","S1"].values[0] - 3/7) < atol
    validate_mathematical_properties(sob, raise_on_fail=True)
    return True


def test_second_order(n_samples=2048, verbose=True):
    """Test con Y = X1·X2 + X3·X4 — valida los índices S2 por pares."""
    s2 = sobol_second_order_from_function(fn=lambda X: X[:,0]*X[:,1] + X[:,2]*X[:,3], bounds=[[0,1]]*6, names=["X1","X2","X3","X4","X5","X6"], n_samples=n_samples, random_state=RANDOM_STATE)
    top2_pairs = set(zip(s2.head(2)["feature_i"].tolist(), s2.head(2)["feature_j"].tolist()))
    assert {("X1","X2"), ("X3","X4")} == top2_pairs
    return True


def run_all_tests(n_samples=1024, verbose=True):
    """Ejecuta todos los tests de validación del módulo."""
    try:
        test_linear_model(n_samples, verbose)
        test_interaction_model(n_samples, verbose)
        test_second_order(n_samples * 2, verbose)
        test_data_driven(verbose=verbose)
        print("\n  >>> TODOS LOS TESTS PASADOS CON ÉXITO ✓ <<<")
        return True
    except Exception as e:
        print(f"\n  >>> ERROR EN LA SUITE DE TESTS: {e} ✗ <<<")
        return False


# =============================================================================
# ENFOQUE DIRECTO DESDE DATOS (MODEL-FREE / GIVEN-DATA)
# =============================================================================

def sobol_first_order_from_data(
    X,
    y,
    n_neighbors=None,
    n_bootstrap=200,
    conf_level=0.95,
    cv_folds=5,
    random_state=RANDOM_STATE,
):
    """Estima los índices de primer orden S1 directamente desde datos observados.

    A diferencia de ``sobol_from_function`` e ``sobol_indices``, este método
    no asume independencia entre variables.
    Trabaja sobre el dataset, aproximando su estructura de correlación.

    Implementa el estimador de la esperanza condicionada ``E[Y | Xi]`` mediante
    k-Nearest Neighbors con validación cruzada, que reproduce el experimento
    mental de 'rebanar y promediar' localmente:

        S1_i = Var(E[Y | Xi]) / Var(Y)

    La validación cruzada elimina el sesgo de sobreajuste que produce usar los
    mismos datos para ajustar el kNN y para estimar la varianza.

    Args:
        X (pd.DataFrame o np.ndarray): Matriz de features. Puede ser el espacio
            original o el espacio preprocesado (post-TargetEncoder).
        y (np.ndarray o pd.Series): Vector del target.
        n_neighbors (int, optional): Número de vecinos para el promedio local.
            Por defecto usa ``max(5, sqrt(n))`` (regla de oro para estimadores
            de expectativa local).
        n_bootstrap (int, optional): Número de remuestras bootstrap para
            calcular el intervalo de confianza de S1. Por defecto 200.
        conf_level (float, optional): Nivel de confianza para el IC.
            Por defecto 0.95.
        cv_folds (int, optional): Número de folds para la validación cruzada
            del estimador kNN. Controla el sesgo: más folds → menos sesgo,
            más coste. Por defecto 5.
        random_state (int, optional): Semilla para reproducibilidad del
            bootstrap y la validación cruzada. Por defecto ``RANDOM_STATE``.

    Returns:
        pd.DataFrame: Tabla compatible con ``sobol_plot``, con columnas:

            - ``feature``: nombre de la variable.
            - ``S1``: índice de primer orden estimado desde datos, clipeado a 0.
            - ``S1_conf``: semiancho del IC bootstrap al nivel ``conf_level``.
            - ``ST``: igual a ``S1`` (no disponible sin muestreo sintético).
            - ``ST_conf``: igual a ``S1_conf``.
            - ``interaction``: 0.0 (no estimable desde datos de primer orden).
            - ``method``: ``"data-driven"`` para distinguir del Sobol clásico.

            Ordenada por ``S1`` descendente.

    Nota:
        **Efecto "Caballo de Troya" (correlación):** si X2 ≈ X1 en los datos
        observados, ``S1_data(X2)`` será artificialmente alto aunque X2 no
        aparezca en la fórmula del target. Esto refleja que
        X2 predice Y en este dataset gracias a su correlación con X1.
        Úsese junto con el Sobol clásico (``sobol_indices``) para distinguir
        causalidad de correlación.

        **Comparación de métodos:**

        - ``sobol_indices``: mide la sensibilidad intrínseca del modelo,
          rompiendo correlaciones mediante muestreo independiente.
        - ``sobol_first_order_from_data``: mide la predictibilidad
          observacional de cada variable, respetando correlaciones reales.

    Example:
        >>> s1_model = sobol_indices(final_model, X_sel)
        >>> s1_data  = sobol_first_order_from_data(X_sel, y)
        >>>
        >>> # Comparar: variables con S1_data >> S1_model (correlaciones)
        >>> fig, ax = sobol_plot(s1_data, threshold_ST=0.05)
    """
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.model_selection import cross_val_predict, KFold

    # Normalizar formatos de entrada
    X_df  = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
    y_arr = np.asarray(y, dtype=float)

    var_Y = float(np.var(y_arr))
    if var_Y < 1e-10:
        raise ValueError(
            "La varianza del target Y es prácticamente cero. "
            "No es posible calcular índices de Sobol normalizados."
        )

    n = len(y_arr)

    # Número de vecinos: regla de oro sqrt(n), mínimo 5
    if n_neighbors is None:
        n_neighbors = max(5, int(np.sqrt(n)))

    # Asegurar que cv_folds sea compatible con n
    cv_folds = min(cv_folds, n // n_neighbors)
    cv_folds = max(2, cv_folds)

    logger.info(
        "sobol_first_order_from_data | %d features, n=%d, k=%d vecinos, %d folds CV",
        len(X_df.columns), n, n_neighbors, cv_folds,
    )

    # Semilla reproducible para bootstrap y CV
    rng = np.random.default_rng(random_state)
    cv  = KFold(n_splits=cv_folds, shuffle=True,
                random_state=int(rng.integers(0, 2**31)))

    # Percentil para IC bootstrap
    alpha     = 1.0 - conf_level
    pct_lo    = alpha / 2.0 * 100
    pct_hi    = (1.0 - alpha / 2.0) * 100

    rows = []

    for col in X_df.columns:
        Xi = X_df[col].values.reshape(-1, 1)

        knn = KNeighborsRegressor(n_neighbors=n_neighbors, weights="uniform")

        # ── Estimación principal con CV ──────────────────────────────────
        # cross_val_predict garantiza que cada punto se predice con un
        # modelo que no lo vio durante el ajuste → sin sesgo de sobreajuste
        ey_cv = cross_val_predict(knn, Xi, y_arr, cv=cv)
        s1_main = float(max(0.0, np.var(ey_cv) / var_Y))

        # ── Bootstrap para intervalo de confianza ────────────────────────
        # Cada remuestreo calcula S1 sobre una muestra con reemplazamiento,
        # capturando la variabilidad de estimación del estimador kNN+CV
        boot_s1 = np.empty(n_bootstrap)
        for b in range(n_bootstrap):
            idx     = rng.integers(0, n, size=n)
            Xi_b    = Xi[idx]
            y_b     = y_arr[idx]
            var_b   = float(np.var(y_b))
            if var_b < 1e-10:
                boot_s1[b] = 0.0
                continue
            cv_b = KFold(n_splits=cv_folds, shuffle=True,
                         random_state=int(rng.integers(0, 2**31)))
            ey_b     = cross_val_predict(knn, Xi_b, y_b, cv=cv_b)
            boot_s1[b] = max(0.0, np.var(ey_b) / var_b)

        # IC como semiancho del percentil bilateral
        s1_conf = float(
            (np.percentile(boot_s1, pct_hi) - np.percentile(boot_s1, pct_lo)) / 2.0
        )

        rows.append({
            "feature":     str(col),
            "S1":          s1_main,
            "S1_conf":     s1_conf,
            "ST":          s1_main,   # no estimable sin muestreo sintético
            "ST_conf":     s1_conf,
            "interaction": 0.0,
            "method":      "data-driven",
        })

        logger.debug(
            "sobol_first_order_from_data | %s: S1=%.4f ± %.4f",
            col, s1_main, s1_conf,
        )

    result = (
        pd.DataFrame(rows)
        .sort_values("S1", ascending=False)
        .reset_index(drop=True)
    )

    if not result.empty:
        top = result.iloc[0]
        logger.info(
            "sobol_first_order_from_data | Top feature: %s (S1=%.4f ± %.4f)",
            top["feature"], top["S1"], top["S1_conf"],
        )

    return result


def sobol_data_indices(
    final_model,
    X,
    y,
    n_neighbors=None,
    n_bootstrap=200,
    conf_level=0.95,
    cv_folds=5,
    random_state=RANDOM_STATE,
):
    """Wrapper de ``sobol_first_order_from_data`` para pipelines sklearn.

    Transforma ``X`` al espacio preprocesado del pipeline antes de calcular
    los índices, de modo que el análisis se realiza sobre las mismas variables
    que ve internamente XGBoost (post-TargetEncoder, post-StandardScaler).

    Args:
        final_model (list[Pipeline]): Pipeline con pasos ``"prep"`` y
            ``"model"``, tal como devuelve ``xgb_sculpt.run_pipe``.
        X (pd.DataFrame): Datos originales de referencia.
        y (np.ndarray o pd.Series): Vector del target.
        n_neighbors (int, optional): Ver ``sobol_first_order_from_data``.
        n_bootstrap (int, optional): Ver ``sobol_first_order_from_data``.
        conf_level (float, optional): Nivel de confianza. Por defecto 0.95.
        cv_folds (int, optional): Ver ``sobol_first_order_from_data``.
        random_state (int, optional): Semilla. Por defecto ``RANDOM_STATE``.

    Returns:
        pd.DataFrame: Ver ``sobol_first_order_from_data``.

    Example:
        >>> s1_model = sobol_indices(final_model, X_sel)
        >>> s1_data  = sobol_data_indices(final_model, X_sel, y)
        >>>
        >>> # Comparar los dos enfoques juntos
        >>> fig1, _ = sobol_plot(s1_model, threshold_ST=0.05)
        >>> fig2, _ = sobol_plot(s1_data,  threshold_ST=0.05)
    """
    prep = final_model[0].named_steps["prep"]
    Xt   = prep.transform(X)

    # Protección contra salida sparse
    try:
        import scipy.sparse as sp
        if sp.issparse(Xt):
            Xt = Xt.toarray()
    except ImportError:
        pass

    feature_names = [n.split("__", 1)[-1] for n in prep.get_feature_names_out()]
    X_transformed = pd.DataFrame(Xt, columns=feature_names)

    return sobol_first_order_from_data(
        X=X_transformed,
        y=y,
        n_neighbors=n_neighbors,
        n_bootstrap=n_bootstrap,
        conf_level=conf_level,
        cv_folds=cv_folds,
        random_state=random_state,
    )


def test_data_driven(verbose=True):
    """Test de ``sobol_first_order_from_data`` con función matemática conocida.

    Dataset sintético con correlación severa entre X1 y X2:
      - Y = 5·X1² + ruido  (solo X1 contribuye causalmente)
      - X2 ≈ X1 (correlación ~0.998)
      - X3 = ruido independiente

    Comportamiento esperado vs Sobol clásico:

    - **Sobol clásico** (``sobol_from_function``): S1(X1) >> S1(X2) ≈ 0.
      Rompe la correlación en el muestreo → descubre la causalidad real.
    - **Sobol datos** (este test): S1(X1) alto Y S1(X2) también alto.
      Respeta la correlación → X2 predice Y porque covaría con X1.

    El test verifica que S1(X3) ≈ 0 en ambos casos (ruido independiente
    siempre debe ser irrelevante independientemente del método).

    Returns:
        bool: True si el test pasa.
    """
    print("\n" + "="*60)
    print("TEST 4: Sobol desde datos con correlación severa")
    print("  Y = 5·X1²,  X2 ≈ X1 (corr≈0.998),  X3 = ruido puro")
    print("  Esperado: S1(X3) ≈ 0; S1(X1) y S1(X2) ambos positivos")
    print("="*60)

    rng = np.random.default_rng(RANDOM_STATE)
    n   = 800

    X1  = rng.uniform(0, 1, n)
    X2  = X1 + rng.normal(0, 0.05, n)   # clon con ruido mínimo
    X3  = rng.uniform(0, 1, n)           # ruido independiente
    Y   = 5 * X1**2 + rng.normal(0, 0.1, n)

    X_df = pd.DataFrame({"X1": X1, "X2": X2, "X3": X3})

    s1 = sobol_first_order_from_data(
        X_df, Y,
        n_bootstrap=50,   # reducido para velocidad del test
        cv_folds=3,
        random_state=RANDOM_STATE,
    )

    if verbose:
        print(s1[["feature", "S1", "S1_conf"]].to_string(index=False))
        corr_x1x2 = float(np.corrcoef(X1, X2)[0, 1])
        print(f"\n  Correlación X1-X2: {corr_x1x2:.4f}")

    s1_x1 = s1.loc[s1["feature"] == "X1", "S1"].values[0]
    s1_x2 = s1.loc[s1["feature"] == "X2", "S1"].values[0]
    s1_x3 = s1.loc[s1["feature"] == "X3", "S1"].values[0]

    # X3 es ruido independiente → siempre debe ser ≈ 0
    assert s1_x3 < 0.10, \
        f"S1(X3)={s1_x3:.4f} debería ser ≈ 0 (ruido independiente)"

    # X1 y X2 covarían con Y — ambos deben tener S1 positivo
    assert s1_x1 > 0.20, \
        f"S1(X1)={s1_x1:.4f} debería ser > 0.20 (causa real de Y)"
    assert s1_x2 > 0.10, \
        f"S1(X2)={s1_x2:.4f} debería ser > 0.10 (correlado con X1 → efecto arrastre)"

    if verbose:
        print(
            f"\n  Interpretación: X2 muestra S1={s1_x2:.3f} aunque no está "
            f"en la fórmula de Y.\n"
            f"  Esto es el 'efecto arrastre' de la correlación X1-X2.\n"
            f"  Comparar con sobol_from_function para ver la diferencia causal."
        )

    print("TEST 4: PASADO ✓")
    return True


# =============================================================================
# NIVEL 3 — SHAPLEY CONDITIONAL
# =============================================================================

def conditional_shapley_effects(
    final_model,
    X_train,
    n_samples_eval: int = 100,
    approach: str = "gaussian",
    random_state: int = 42,
) -> tuple[pd.DataFrame, object]:
    """
    Calcula importancias mediante Valores de Shapley Condicionales
    usando el paquete ``shapr-python``.
 
    Parameters
    ----------
    final_model : sklearn.pipeline.Pipeline
        Pipeline sklearn ya entrenado. Debe exponer un método ``predict``.
    X_train : pd.DataFrame
        Datos de entrenamiento originales (sin transformar).
    n_samples_eval : int, default=100
        Número máximo de filas de ``X_train`` sobre las que se calculan
        las explicaciones individuales. Se limita a ``len(X_train)``
        si este valor es menor.
    approach : str, default="gaussian"
        Método de aproximación para shapr. Opciones habituales:
        ``"gaussian"``, ``"copula"``, ``"empirical"``.
    random_state : int, default=42
        Semilla para los muestreos aleatorios (reproducibilidad).
 
    Returns
    -------
    df_results : pd.DataFrame
        Columnas ``["feature", "Conditional_Shapley_Importance"]``,
        ordenado de mayor a menor importancia.
    shap_exp : shap.Explanation
        Objeto SHAP estándar con ``.values``, ``.data`` y ``.feature_names``.
        Puede pasarse directamente a ``plot_shapley_effects()``.
 
    Raises
    ------
    ImportError
        Si ``shaprpy`` no está instalado.
    TypeError
        Si ``final_model`` no es un ``Pipeline`` o ``X_train`` no es un
        ``DataFrame``.
    ValueError
        Si ``X_train`` está vacío.
    """
    # ------------------------------------------------------------------
    # Dependencia opcional
    # ------------------------------------------------------------------
    try:
        import shaprpy
    except ImportError as exc:
        raise ImportError(
            "El paquete 'shaprpy' es necesario para esta función.\n"
            "Instalar con:  pip install shaprpy"
        ) from exc
 
    # ------------------------------------------------------------------
    # Normalización de final_model
    # Acepta un Pipeline suelto o una lista/tupla de un único elemento.
    # ------------------------------------------------------------------
    if isinstance(final_model, (list, tuple)):
        if len(final_model) != 1:
            raise ValueError(
                f"Si final_model es una lista o tupla debe contener exactamente "
                f"un Pipeline; se recibieron {len(final_model)} elementos."
            )
        logger.debug("final_model recibido como secuencia; extrayendo el primer elemento.")
        final_model = final_model[0]
 
    # ------------------------------------------------------------------
    # Validaciones defensivas
    # ------------------------------------------------------------------
    if not isinstance(final_model, Pipeline):
        raise TypeError("final_model debe ser un sklearn.pipeline.Pipeline entrenado.")
    if not isinstance(X_train, pd.DataFrame):
        raise TypeError("X_train debe ser un pandas.DataFrame.")
    if X_train.empty:
        raise ValueError("X_train está vacío.")
 
    pipeline     = final_model
    preprocessor = pipeline[:-1]   # todos los pasos menos el modelo final
    estimator    = pipeline[-1]     # solo el estimador final: model
 
    # ------------------------------------------------------------------
    # X_train transformado con nombres originales recuperados
    # get_feature_names_out() devuelve "paso__nombre_original"; con
    # split("__", 1)[-1] recuperamos el nombre limpio.
    # ------------------------------------------------------------------
    Xt = preprocessor.transform(X_train)

    try:
        import scipy.sparse as sp
        if sp.issparse(Xt):
            Xt = Xt.toarray()
    except ImportError:
        pass
        
    X_train_t = pd.DataFrame(
        Xt,
        columns=[n.split("__", 1)[-1] for n in preprocessor.get_feature_names_out()],
        index=X_train.index,
    )
 
    feature_names: list[str] = list(X_train_t.columns)
    n_features = len(feature_names)
 
    # ------------------------------------------------------------------
    # Función predict compatible con shaprpy
    # Firma requerida: f(model, newdata) -> np.ndarray
    # ------------------------------------------------------------------
    def _predict_func(model, newdata: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Wrapper con la firma que exige shaprpy: (model, newdata)."""
        if not isinstance(newdata, pd.DataFrame):
            if newdata.shape[1] != n_features:
                raise ValueError(
                    f"_predict_func recibió {newdata.shape[1]} columnas "
                    f"pero se esperaban {n_features}."
                )
            newdata = pd.DataFrame(newdata, columns=feature_names)
        return np.asarray(model.predict(newdata)).ravel()
 
    # ------------------------------------------------------------------
    # Subconjunto de evaluación (sobre datos transformados)
    # ------------------------------------------------------------------
    n_eval = min(n_samples_eval, len(X_train_t))
    X_eval = X_train_t.sample(n=n_eval, random_state=random_state)
    logger.info(
        "shapr: calculando explicaciones para %d muestras (approach=%s)",
        n_eval,
        approach,
    )
 
    # ------------------------------------------------------------------
    # Baseline: media de predicciones sobre una muestra representativa
    # ------------------------------------------------------------------
    n_baseline = min(2_000, len(X_train_t))
    X_baseline = X_train_t.sample(n=n_baseline, random_state=random_state)
    prediction_zero = float(np.mean(_predict_func(estimator, X_baseline)))
    logger.debug(
        "prediction_zero = %.6f  (calculado sobre %d filas)",
        prediction_zero,
        n_baseline,
    )
 
    # ------------------------------------------------------------------
    # Cálculo de Shapley condicionales
    # Se pasa el estimador final (no el pipeline completo) porque
    # x_train y x_explain ya están transformados.
    # ------------------------------------------------------------------
    explanation = shaprpy.explain(
        model=estimator,
        x_train=X_train_t,
        x_explain=X_eval,
        approach=approach,
        phi0=prediction_zero,
        predict_model=_predict_func,
        seed=random_state,
    )
 
    # ------------------------------------------------------------------
    # Extracción de valores Shapley
    #
    # to_shap() devuelve un shap.Explanation estándar:
    #   .values        -> np.ndarray (n_eval, n_features), sin columna "none"
    #   .feature_names -> lista de nombres ya alineada con .values
    # Usar este objeto evita tener que gestionar la columna "none" o
    # depender de la estructura interna de get_results().
    # ------------------------------------------------------------------
    shap_exp = explanation.to_shap()
 
    returned_features = list(shap_exp.feature_names)
    global_scores = np.mean(np.abs(shap_exp.values), axis=0)
 
    missing = set(feature_names) - set(returned_features)
    if missing:
        logger.warning(
            "shaprpy no devolvió valores para las siguientes features: %s. "
            "El resultado puede estar incompleto.",
            missing,
        )
 
    # ------------------------------------------------------------------
    # Resultado final
    # ------------------------------------------------------------------
    df_results = (
        pd.DataFrame(
            {
                "feature": returned_features,
                "Conditional_Shapley_Importance": global_scores,
            }
        )
        .sort_values("Conditional_Shapley_Importance", ascending=False)
        .reset_index(drop=True)
    )
 
    logger.info("shaprpy: importancias calculadas para %d features.", len(df_results))
    return df_results, shap_exp

# ---------------------------------------------------------------------------
# Visualización
# ---------------------------------------------------------------------------
    
def plot_shapley_effects(
    shap_explanation: Any,
    plots: list[str] | None = None,
    max_display: int = 20,
    sample_idx: int = 0,
    output_dir: str | Path | None = None,
    show: bool = True,
) -> dict[str, Any]:
    """
    Genera plots estándar SHAP a partir de un ``shap.Explanation``.
    """
    try:
        import matplotlib.pyplot as plt
        import shap
    except ImportError as exc:
        raise ImportError(
            "Los paquetes 'shap' y 'matplotlib' son necesarios.\n"
            "Instalar con:\npip install shap matplotlib"
        ) from exc

    plots_to_run = list(_VALID_PLOTS) if plots is None else list(plots)
    invalid = set(plots_to_run) - _VALID_PLOTS

    if invalid:
        raise ValueError(
            f"Plots no reconocidos: {invalid}. Valores válidos: {_VALID_PLOTS}"
        )

    n_samples = shap_explanation.values.shape[0]

    if sample_idx < 0 or sample_idx >= n_samples:
        raise ValueError(
            f"sample_idx={sample_idx} fuera de rango (n_samples={n_samples})"
        )

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        show = False

    figures: dict[str, Any] = {}

    def _finalize_plot(name: str):
        fig = plt.gcf()
        if output_dir is not None:
            path = output_dir / f"{name}.png"
            fig.savefig(path, bbox_inches="tight", dpi=150)
            logger.info("Plot guardado: %s", path)

        if show:
            plt.show()

        figures[name] = fig
        plt.close(fig)

    ordered = [
        p for p in ["bar", "beeswarm", "waterfall", "heatmap", "scatter2d"]
        if p in plots_to_run
    ]

    if "bar" in ordered:
        plt.figure()
        shap.plots.bar(shap_explanation, max_display=max_display, show=False)
        _finalize_plot("bar")

    if "beeswarm" in ordered:
        plt.figure()
        shap.plots.beeswarm(shap_explanation, max_display=max_display, show=False)
        _finalize_plot("beeswarm")

    if "waterfall" in ordered:
        plt.figure()
        shap.plots.waterfall(shap_explanation[sample_idx], show=False)
        _finalize_plot(f"waterfall_sample{sample_idx}")

    if "heatmap" in ordered:
        plt.figure()
        shap.plots.heatmap(shap_explanation, max_display=max_display, show=False)
        _finalize_plot("heatmap")

    if "scatter2d" in ordered:
        feature_names = list(shap_explanation.feature_names)

        if len(feature_names) < 2:
            logger.warning("scatter2d requiere al menos 2 features; se omite.")
        else:
            global_importance = np.mean(np.abs(shap_explanation.values), axis=0)
            top_indices = np.argsort(global_importance)[::-1]
            top_feature = feature_names[top_indices[0]]

            n_color_features = min(4, len(feature_names) - 1)
            color_features = [
                feature_names[top_indices[i]] for i in range(1, n_color_features + 1)
            ]

            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle(
                f"SHAP Scatter — {top_feature} vs features siguientes", fontsize=13
            )

            for i, ax in enumerate(axes.flat):
                if i < n_color_features:
                    shap.plots.scatter(
                        shap_explanation[:, top_feature],
                        color=shap_explanation[:, color_features[i]],
                        ax=ax,
                        show=False,
                    )
                    ax.set_title(f"color: {color_features[i]}", fontsize=10)
                else:
                    ax.set_visible(False)

            plt.tight_layout()

            if output_dir is not None:
                path = output_dir / "scatter2d.png"
                fig.savefig(path, bbox_inches="tight", dpi=150)
                logger.info("Plot guardado: %s", path)

            if show:
                plt.show()

            figures["scatter2d"] = fig
            plt.close(fig)

    return figures