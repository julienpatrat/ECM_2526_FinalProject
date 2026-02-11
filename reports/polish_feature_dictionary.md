# Polish dataset — Feature dictionary (Attr1..Attr64)

Ce document décrit la signification des variables `Attr1..Attr64` du dataset **Polish Companies Bankruptcy** (UCI).
Le dataset contient 5 fichiers (1stYear..5thYear) correspondant à des horizons de prédiction différents.
Pour `1stYear`, les ratios sont observés au début de la période et la variable `class` indique la faillite dans les 5 ans.  
Source : UCI + articles associés. 

## Structure générale
- **Attr1..Attr64** : ratios/indicateurs financiers (variables explicatives, numériques)
- **class** : cible binaire
  - `1` = entreprise en faillite (bankrupted)
  - `0` = entreprise non faillie sur l’horizon considéré

## Dictionnaire des variables (ratios)
- **Attr1** — Net Profit / Total Assets  
- **Attr2** — Total Liabilities / Total Assets  
- **Attr3** — Working Capital / Total Assets  
- **Attr4** — Current assets / Short-term liabilities  
- **Attr5** — (Cash + Short-term securities + Receivables - Short-term liabilities) / (Operating expenses - Depreciation)  
- **Attr6** — Retained Earnings / Total Assets  
- **Attr7** — EBIT / Total Assets  
- **Attr8** — Book Value of Equity / Total Liabilities  
- **Attr9** — Sales / Total Assets  
- **Attr10** — Equity / Total Assets  
- **Attr11** — (Gross Profit + Extraordinary Items + Financial Expenses) / Total Assets  
- **Attr12** — Gross Profit / Short-term liabilities  
- **Attr13** — (Gross Profit + Depreciation) / Sales  
- **Attr14** — (Gross Profit + Interest) / Total Assets  
- **Attr15** — (Total liabilities * 365) / (Gross Profit + Depreciation)  
- **Attr16** — (Gross Profit + Depreciation) / Total liabilities  
- **Attr17** — Total assets / Total liabilities  
- **Attr18** — Gross Profit / Total Assets  
- **Attr19** — Gross Profit / Sales  
- **Attr20** — (Inventory * 365) / Sales  
- **Attr21** — Sales (n) / Sales (n-1)  
- **Attr22** — Profit on operating activities / Total Assets  
- **Attr23** — Net Profit / Sales  
- **Attr24** — Gross Profit (in 3 years) / Total Assets  
- **Attr25** — (Equity - Share Capital) / Total Assets  
- **Attr26** — (Net Profit + Depreciation) / Total liabilities  
- **Attr27** — Profit on operating activities / Financial expenses  
- **Attr28** — Working Capital / Fixed Assets  
- **Attr29** — Logarithm of total assets  
- **Attr30** — (Total liabilities - Cash) / Sales  
- **Attr31** — (Gross Profit + Interest) / Sales  
- **Attr32** — (Current liabilities * 365) / Cost of products sold  
- **Attr33** — Operating expenses / Short-term liabilities  
- **Attr34** — Operating expenses / Total liabilities  
- **Attr35** — Profit on sales / Total assets  
- **Attr36** — Total sales / Total assets  
- **Attr37** — (Current assets - Inventories) / Long-term liabilities  
- **Attr38** — Constant capital / Total assets  
- **Attr39** — Profit on sales / Sales  
- **Attr40** — (Current assets - Inventory - Receivables) / Short-term liabilities  
- **Attr41** — Total liabilities / ((Profit on operating activities + Depreciation) * (12/365))  
- **Attr42** — Profit on operating activities / Sales  
- **Attr43** — Rotation receivables + inventory turnover in days  
- **Attr44** — (Receivables * 365) / Sales  
- **Attr45** — Net Profit / Inventory  
- **Attr46** — (Current assets - Inventory) / Short-term liabilities  
- **Attr47** — (Inventory * 365) / Cost of products sold  
- **Attr48** — EBITDA (profit on operating activities + depreciation) / Total assets  
- **Attr49** — EBITDA (profit on operating activities + depreciation) / Sales  
- **Attr50** — Current assets / Total liabilities  
- **Attr51** — Short-term liabilities / Total assets  
- **Attr52** — (Short-term liabilities * 365) / Cost of products sold  
- **Attr53** — Equity / Fixed assets  
- **Attr54** — Constant capital / Fixed assets  
- **Attr55** — Working capital  
- **Attr56** — (Sales - Cost of products sold) / Sales  
- **Attr57** — (Current assets - Inventory - short-term liabilities) / (Sales - Gross profit - Depreciation)  
- **Attr58** — Total costs / Total sales  
- **Attr59** — Long-term liabilities / Equity  
- **Attr60** — Sales / Inventory  
- **Attr61** — Sales / Receivables  
- **Attr62** — (Short-term liabilities * 365) / Sales  
- **Attr63** — Sales / Short-term liabilities  
- **Attr64** — Sales / Fixed assets  

## Références
- UCI Machine Learning Repository — *Polish Companies Bankruptcy Data* (description du dataset, horizons 1stYear..5thYear).  
- Articles utilisant le dataset et listant les ratios en Table “feature description”. 

