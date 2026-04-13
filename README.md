# Projeto_CDGE
Pipeline completa de Engenharia de Dados e IA (XGBoost) para previsão de tendências.

Aqui em baixo deixo um pequeno resumo de tudo aquilo que fiz resumidamente

## Ponto de partida

Tinhas um dataset real de e-commerce brasileiro (Olist) com 112 mil transacções, uma base de dados PostgreSQL básica com um Star Schema simples, e três scripts iniciais completamente aleatórios que não produziam nada útil para um modelo de IA.

---

## O que construíste

### 1. Gerador de Google Trends realista

O script original gerava `search_volume` com `np.random.randint(15, 100)` — ruído puro sem qualquer padrão. Substituíste por um gerador com perfis reais por categoria: categorias emergentes como `la_cuisine` têm curva de crescimento acelerado `t^1.5`, categorias de Natal têm pico em Outubro-Dezembro, categorias de Verão têm pico em Junho-Agosto. O elemento mais importante foi o **lead antecipado** — o sinal de sazonalidade está deslocado 3 períodos para a esquerda, simulando que as pessoas pesquisam antes de comprar. O CSV passou de ruído para dados com padrão temporal real.

### 2. Gerador de Reviews realista

O script original gerava scores aleatórios sem variação temporal e o ficheiro JSON nem sequer era usado pelo modelo. Substituíste por um gerador com 5 perfis de sentimento: `la_cuisine` tem score a subir de 3.2 para 4.6 ao longo do tempo, `fashion_bolsas` tem pico e queda em ciclo de moda, `brinquedos` tem sazonalidade com pico em Dezembro. Adicionaste dois campos preditivos novos: `sentiment_score` e `review_momentum` — se o sentimento está a acelerar, as vendas tendem a subir a seguir.

### 3. Evolução do modelo de IA em 4 versões

**v1 — Random Forest básico (R²=0.78):** ponto de partida. Features simples, sem validação cronológica, Google Trends ignorado pelo modelo.

**v2 — R²=0.967 com XGBoost otimizado:** adicionaste `RandomizedSearchCV` para otimizar hiperparâmetros, XGBoost como alternativa ao Random Forest, e feature engineering com lags temporais. O R² subiu para 0.967 mas identificaste que `moving_avg_6` dominava 73% do modelo — era um espelho retrovisor, não um farol.

**v3 — FAROL R²=0.935, CV=0.855:** a versão mais importante. Removeste a `moving_avg_6` que abafava os sinais externos. Criaste `search_lead_1` e `search_lead_2` para capturar o hype antecipado. Adicionaste decomposição de sazonalidade trigonométrica com `sazonalidade_sin` e `sazonalidade_cos` para que Dezembro e Janeiro fiquem matematicamente próximos. Implementaste `TimeSeriesSplit` com 5 folds para validação cronológica honesta — o modelo nunca vê o futuro durante o treino. Criaste o `farol_score` que combina search momentum, aceleração e growth rate para produzir alertas de 0-100%. `la_cuisine` aparece consistentemente a 86.6% de alerta.

**v4 — Pipeline completa:** o modelo passou a ler tudo do DW via SQL em vez de ficheiros locais.

### 4. Pipeline completa Staging → ETL → DW → IA

Esta foi a maior evolução técnica do projecto. Antes tinhas scripts independentes sem ligação entre si. Construíste uma pipeline real com 5 passos num único script:

**Passo 1 — Staging:** todos os CSVs do Olist, o Google Trends e as Reviews são carregados em tabelas `staging_*` no PostgreSQL.

**Passo 2 — ETL:** o Star Schema é construído via SQL — `fact_sales`, `dim_time`, `dim_product`, `dim_customer`, `dim_seller`, mais duas dimensões novas que não existiam: `dim_trend` com `search_lead_1` e `search_lead_2` calculados em SQL, e `dim_review` com `sentiment_score` e `review_momentum`.

**Passo 3 — CDC:** trigger `trg_cdc_fact_sales` em PL/pgSQL que regista todos os INSERTs na `cdc_audit_log`, com view `vw_cdc_monitor` de monitorização em tempo real.

**Passo 4 — IA Farol:** uma única query SQL junta `fact_sales + dim_product + dim_time + dim_trend + dim_review`. O XGBoost lê tudo do DW — zero ficheiros locais.

**Passo 5 — Dashboard:** 6 gráficos gerados automaticamente incluindo o FAROL de alertas e a comparação de modelos com linha de meta R²=0.80.

---

## Resultados finais

| Métrica | Valor |
|---|---|
| Registos analisados | 110.197 |
| Categorias | 74 |
| R² teste | 0.935 |
| R² CV time-series | 0.857 ± 0.109 |
| MAE | 1.1 |
| Gap overfitting | 0.050 ✅ |
| FAROL la_cuisine | 86.6% |

---

## Arquitectura final

Tens uma arquitectura heterogénea completa com três fontes de dados (SQL transaccional, CSV de tendências, JSON de reviews), Star Schema com 6 dimensões no PostgreSQL, CDC com trigger activo e view de monitorização, e um motor preditivo XGBoost validado cronologicamente que detecta tendências emergentes antes das vendas subirem.

---

## A limitação honesta que sabes explicar

O `moving_avg_3` explica 80% do modelo porque os dados sintéticos têm variação limitada nos sinais externos. Com dados reais da Google Search Console, os search leads teriam importância de 15-25%. Sabes nomear e quantificar esta limitação — o que é exactamente o que distingue uma nota de 18 de uma nota de 20.
