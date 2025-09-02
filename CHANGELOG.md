# Changelog

## [0.2.8] - 2025-09-02
- Feature: soporte de idioma ("es"/"en") en reportes Markdown, gráficos y CLI mediante parámetro `lang`.
- API/CLI: agrega parámetro `lang` a `make_report` y CLI.
- Docs/README: ejemplos y explicación de uso multilenguaje.

## [0.2.7] - 2025-08-30
- Packaging: declara `requires-python >=3.9` y **Trove classifiers** (3.9–3.13) en `pyproject.toml`.
- Docs: descripción actualizada (incluye forecast) y correcciones menores de formato.

## [0.2.6] - 2025-08-30
- Docs: README simplificado para instalación por `pip` y uso **solo en Python** (sección CLI movida fuera).
- Ejemplos: clarifica entradas esperadas y salidas (Markdown + PNGs en `evalcards_reports/`).

## [0.2.5] - 2025-08-30
- Docs: agrega badges de estado (CI/Publish, PyPI, wheel, licencia) al README.
- Meta: añade URLs del repositorio (Homepage/Repository/Issues).

## [0.2.4] - 2025-08-30
- Release: estabiliza publicación automática (ajustes en `release.yml`, permisos y orden de pasos).
- Docs: guía breve sobre cómo versionar y publicar con tags (`vX.Y.Z`).

## [0.2.3] - 2025-08-30
- CI/Release: hardening del pipeline de publicación (mejor manejo de credenciales y disparo por tags).
- Docs: añade ejemplos de uso en Python (binaria, multiclase, regresión, forecast) en el README.

## [0.2.2] - 2025-08-30
- CI: agrega workflow de tests (matrix 3.9–3.13) y suite mínima (binaria, regresión, multiclase OvR, forecast).
- Fix: fuerza backend de Matplotlib a **Agg** para entornos sin GUI (tests/servidores).
- Release: primer workflow de publicación a PyPI (por tag `v*`).

## [0.2.1] - 2025-08-30
- Release de prueba del flujo CI/CD (publicación por tag).

## [0.2.0] - 2025-08-30
- Multiclase OvR: curvas ROC/PR por clase y `roc_auc_ovr_macro`.
- Forecasting: sMAPE (%) y MASE + gráficos (fit, residuales).
- Backend matplotlib forzado a Agg (tests/servidores sin GUI).
- README reorganizado.

## [0.1.0] - 2025-08-30
- Publicación: clasificación binaria, regresión, CLI y reportes Markdown.
