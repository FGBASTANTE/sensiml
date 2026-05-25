# sensiml
Sensitivity interpretable ML  Framework for xgb_sculpt: análisis global de sensibilidad, interacciones e interpretabilidad de modelos tabulares de machine learning.

<img width="450" height="300" alt="logo_sensiml" src="https://github.com/user-attachments/assets/a1337871-f4d6-45cd-979c-dad16fcc0a4d" />

Incluye:

- Índices de Sobol (S1, ST, S2)
- Sensibilidad observacional basada en datos
- Valores de Shapley Condicionales
- Visualizaciones integradas
- Tests matemáticos de validación

Compatible con pipelines sklearn/XGBoost y modelos tipo caja negra.

---

## Dependencias

```bash
pip install SALib>=1.5 pandas numpy matplotlib scikit-learn
pip install xgboost
pip install shap shaprpy
```
Requiere R y su módulo shapr instalados

## Ejemplos

import sensiml as sss
### Sobol clásico
s1 = sss.sobol_indices(final_model, X)
### Interacciones Sobol
s2 = sss.sobol_second_order(final_model, X)
### Sobol observacional
s1_data = sss.sobol_data_indices(final_model, X, y)
### Shapley condicional
shap_df, shap_exp = sss.conditional_shapley_effects(
    final_model,
    X,
    approach="empirical",
)
### Visualización
sss.sobol_plot(s1)
sss.sobol_interaction_heatmap(s2)
sss.plot_shapley_effects(shap_exp)


Este código está orientado a usuarios con experiencia en: análisis de sensibilidad, interpretabilidad ML, estadística multivariante, pipelines sklearn/XGBoost...
No se recomienda su uso como herramienta “plug-and-play” sin comprender las hipótesis estadísticas y las limitaciones metodológicas de cada técnica y las derivadas de su implemetación.
