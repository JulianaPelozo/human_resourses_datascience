# Atividade de Regressão: Predição de Salários no Departamento de RH

Projeto de ciência de dados que usa técnicas de regressão para identificar quais fatores determinam, direta ou indiretamente, os salários em uma organização, e compara três modelos de machine learning na tarefa de prever o salário de um funcionário.

## Integrantes

- Carolina Santos
- Eduardo Mello
- Juliana Pelozo
- Luiz Felipe Moraes
- Miguel Veloso
- Saulo de Lucena
- Saulo Monteiro
- Victor Militino

## Dataset

Foi utilizado o [Human Resources Data Set](https://www.kaggle.com/datasets/rhuebner/human-resources-data-set) (arquivo `HRDataset_v14.csv`), disponível no Kaggle.

| Item | Valor |
|---|---|
| Registros (funcionários) | 311 |
| Colunas originais | 36 |
| Variável alvo | `Salary` |
| Salário mínimo / mediano / máximo | 45.046 / 62.810 / 250.000 |
| Valores ausentes | `DateofTermination` (207) e `ManagerID` (8) |

## Metodologia

1. **Exploração dos dados**: dimensões, tipos, estatísticas descritivas, valores ausentes e gráficos (distribuição de salário, salário por departamento, salário por desempenho, satisfação vs. engajamento, atrasos vs. ausências).
2. **Tratamento dos dados**
   - `ManagerID` ausente preenchido com `-1`.
   - Criação da variável indicadora `WasTerminated` a partir de `DateofTermination`.
   - Seleção de 15 variáveis numéricas como features.
   - Verificação de outliers de salário pelo método do IQR (limites de 30.699,75 a 96.837,75; 29 outliers identificados, mantidos no modelo).
3. **Divisão treino/teste**: 80% / 20% (`random_state=42`), resultando em 248 registros de treino e 63 de teste.
4. **Padronização** com `StandardScaler` (usada apenas na Regressão Linear).
5. **Modelagem** com três algoritmos: Regressão Linear, Random Forest (100 árvores) e XGBoost (100 estimadores).
6. **Avaliação** com R², MAE e MSE, além de gráficos de valores reais vs. preditos, análise de resíduos e importância das variáveis.

## Resultados

| Modelo | R² | MAE | MSE | RMSE (calculado) |
|---|---|---|---|---|
| Regressão Linear | 0,3739 | 13.502,37 | 506.809.021,52 | ~22.512 |
| Random Forest | 0,5444 | 10.196,18 | 368.798.329,01 | ~19.204 |
| **XGBoost** | **0,6073** | **9.078,27** | **317.870.624,00** | **~17.829** |

O XGBoost obteve o melhor desempenho, explicando cerca de 60,7% da variância dos salários no conjunto de teste, com erro médio absoluto de aproximadamente 9.078 em unidades de salário do dataset.

### Variáveis mais importantes

| Posição | Random Forest | XGBoost |
|---|---|---|
| 1 | `DeptID` (0,324) | `DeptID` (0,430) |
| 2 | `PositionID` (0,277) | `PositionID` (0,264) |
| 3 | `SpecialProjectsCount` (0,111) | `SpecialProjectsCount` (0,189) |
| 4 | `Absences` (0,086) | `GenderID` (0,032) |
| 5 | `EngagementSurvey` (0,064) | `EngagementSurvey` (0,023) |

Departamento e cargo concentram a maior parte do poder explicativo nos dois modelos, seguidos pelo número de projetos especiais.

## Conclusões

- Os modelos baseados em árvores (Random Forest e XGBoost) superaram a Regressão Linear, indicando relações não lineares e interações entre as variáveis.
- Departamento, cargo e participação em projetos especiais são os principais determinantes do salário neste dataset.
- O desempenho (R² de ~0,61 no melhor caso) é moderado: há margem para melhoria.

## Limitações e próximos passos

- **Amostra pequena** (311 registros) e conjunto de teste de apenas 63 registros, o que torna as métricas sensíveis à divisão escolhida. Recomenda-se validação cruzada.
- **Outliers** de salário (até 250.000) foram mantidos e impactam MSE e RMSE.
- **Variáveis categóricas codificadas como números** (`DeptID`, `PositionID`, `MaritalStatusID`) são tratadas como numéricas; one-hot encoding ou tratamento categórico nativo pode melhorar o resultado.
- **Variáveis de status** (`EmpStatusID`, `Termd`, `WasTerminated`) são redundantes entre si e devem ser avaliadas quanto à relevância e a possíveis vieses.
- **Sem ajuste de hiperparâmetros**: os modelos usam configurações quase padrão. Busca em grade ou aleatória (`GridSearchCV`, `RandomizedSearchCV`) é um próximo passo natural.
- **Variáveis sensíveis** (como `GenderID`) aparecem entre as mais importantes no XGBoost, o que justifica uma análise de equidade salarial antes de qualquer uso prático.

## Como executar

1. Baixe o dataset `HRDataset_v14.csv` no Kaggle (link acima).
2. Abra `atividade_regressao.ipynb` no Google Colab. O notebook monta o Google Drive e lê o arquivo em `/content/drive/MyDrive/HRDataset_v14.csv`; ajuste o caminho se necessário.
3. Execute as células em ordem.

Para executar localmente, instale as dependências e troque a leitura do Drive por `pd.read_csv('HRDataset_v14.csv')`:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
```

## Tecnologias

Python, pandas, NumPy, Matplotlib, Seaborn, scikit-learn e XGBoost.

## Material de estudo

A apostila [`apostila_metodos.md`](apostila_metodos.md) explica cada método utilizado neste trabalho.
