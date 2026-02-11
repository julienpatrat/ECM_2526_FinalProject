# US dataset — Feature dictionary (X1..X18)

Ce document décrit la signification des variables `X1..X18` du dataset US (bankruptcy prediction).
Les définitions ci-dessous proviennent de l’article de référence associé au dataset.

## Features (variables explicatives)

- **X1** — Current assets (actifs courants)
- **X2** — Cost of goods sold (coût des marchandises vendues)
- **X3** — Depreciation & amortization (dotations aux amortissements)
- **X4** — EBITDA
- **X5** — Inventory (stocks)
- **X6** — Net income (résultat net)
- **X7** — Total receivables (créances totales / clients)
- **X8** — Market value (valeur de marché / capitalisation)
- **X9** — Net sales (ventes nettes)
- **X10** — Total assets (total de l’actif)
- **X11** — Total long-term debt (dettes à long terme)
- **X12** — EBIT
- **X13** — Gross profit (profit / marge brute)
- **X14** — Total current liabilities (passifs courants)
- **X15** — Retained earnings (résultats non distribués)
- **X16** — Total revenue (revenu / chiffre d’affaires total)
- **X17** — Total liabilities (total du passif / dettes)
- **X18** — Total operating expenses (charges d’exploitation totales)

## Autres colonnes (métadonnées)
- `company_name` : identifiant d’entreprise anonymisé (ex : `C_1`, `C_2`, …)
- `fyear` : année fiscale
- `status_label` : label de faillite (transformé en `class` dans notre pipeline)
- `Division`, `MajorGroup` : catégories sectorielles (présentes dans le fichier brut)

## Référence
- *Machine Learning for Bankruptcy Prediction in the American Stock Market: Dataset and Benchmarks*, Future Internet (2022). DOI: **10.3390/fi14080244**

