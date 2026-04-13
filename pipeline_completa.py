"""
╔══════════════════════════════════════════════════════════════════╗
║          TRENDMART 2026 — PIPELINE COMPLETA                     ║
║                                                                  ║
║  PASSO 1 — STAGING    : Carrega todos os CSVs/JSON crus         ║
║  PASSO 2 — ETL        : Limpa, transforma, cria dims/facts      ║
║  PASSO 3 — DW         : Star Schema + dim_trend + dim_review    ║
║  PASSO 4 — CDC        : Trigger de auditoria PL/pgSQL           ║
║  PASSO 5 — IA FAROL   : XGBoost lê tudo do DW via SQL          ║
╚══════════════════════════════════════════════════════════════════╝

ORDEM DE EXECUÇÃO:
    1. gerar_trends.py      → google_trends_sintetico.csv
    2. gerar_reviews.py     → reviews_sinteticas.json
    3. pipeline_completa.py → faz TUDO o resto
"""

import pandas as pd
import numpy as np
import json
import warnings
import zipfile
import io
warnings.filterwarnings('ignore')

from sqlalchemy import create_engine, text
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

try:
    import xgboost as xgb
    XGBOOST_OK = True
except ImportError:
    XGBOOST_OK = False
    print("⚠️  XGBoost não instalado. Corre: pip install xgboost")

# ════════════════════════════════════════════════════════════════
# CONFIGURAÇÃO
# ════════════════════════════════════════════════════════════════
DB_URL      = 'postgresql://admin:admin@localhost:5432/pl_cgde_store'
TRENDS_CSV  = 'google_trends_sintetico.csv'
REVIEWS_JSON = 'reviews_sinteticas.json'

engine = create_engine(DB_URL)

def executar_sql(sql, mensagem=""):
    with engine.begin() as conn:
        conn.execute(text(sql))
    if mensagem:
        print(f"   ✅ {mensagem}")

# ════════════════════════════════════════════════════════════════
# PASSO 1 — STAGING AREA
# Carrega os dados crus (CSV + JSON) para tabelas de staging
# ════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  PASSO 1 — STAGING AREA")
print("="*60)

# ── 1A. Ler CSVs do Olist (podem estar em zip) ──────────────────
def ler_csv(nome):
    try:
        with zipfile.ZipFile(nome + '.zip') as z:
            with z.open(nome) as f:
                return pd.read_csv(f)
    except FileNotFoundError:
        return pd.read_csv(nome)

print("1. A carregar ficheiros de origem...")

try:
    df_orders   = ler_csv('olist_orders_dataset.csv')
    df_items    = ler_csv('olist_order_items_dataset.csv')
    df_products = ler_csv('olist_products_dataset.csv')
    df_customers = ler_csv('olist_customers_dataset.csv')
    df_sellers  = ler_csv('olist_sellers_dataset.csv')
    df_reviews_raw = ler_csv('olist_order_reviews_dataset.csv')
    df_transl   = pd.read_csv('product_category_name_translation.csv')
    print(f"   ✅ CSVs Olist carregados")
except Exception as e:
    print(f"   ⚠️  Erro a ler CSVs Olist: {e}")

# Trends CSV (gerado pelo gerar_trends.py)
df_trends_raw = pd.read_csv(TRENDS_CSV)
print(f"   ✅ Google Trends: {len(df_trends_raw):,} linhas")

# Reviews JSON (gerado pelo gerar_reviews.py)
with open(REVIEWS_JSON, 'r', encoding='utf-8') as f:
    reviews_data = json.load(f)
df_reviews_sint = pd.DataFrame(reviews_data)
print(f"   ✅ Reviews sintéticas: {len(df_reviews_sint):,} linhas")

# ── 1B. Criar schema de staging no PostgreSQL ───────────────────
print("\n2. A criar tabelas de staging...")

executar_sql("""
    DROP TABLE IF EXISTS staging_orders CASCADE;
    CREATE TABLE staging_orders (
        order_id VARCHAR(50),
        customer_id VARCHAR(50),
        order_status VARCHAR(30),
        order_purchase_timestamp TIMESTAMP,
        order_delivered_customer_date TIMESTAMP,
        order_estimated_delivery_date TIMESTAMP
    );
""", "staging_orders criada")

executar_sql("""
    DROP TABLE IF EXISTS staging_order_items CASCADE;
    CREATE TABLE staging_order_items (
        order_id VARCHAR(50),
        order_item_id INT,
        product_id VARCHAR(50),
        seller_id VARCHAR(50),
        price NUMERIC(10,2),
        freight_value NUMERIC(10,2)
    );
""", "staging_order_items criada")

executar_sql("""
    DROP TABLE IF EXISTS staging_products CASCADE;
    CREATE TABLE staging_products (
        product_id VARCHAR(50),
        product_category_name VARCHAR(100)
    );
""", "staging_products criada")

executar_sql("""
    DROP TABLE IF EXISTS staging_customers CASCADE;
    CREATE TABLE staging_customers (
        customer_id VARCHAR(50),
        customer_unique_id VARCHAR(50),
        customer_city VARCHAR(100),
        customer_state VARCHAR(10)
    );
""", "staging_customers criada")

executar_sql("""
    DROP TABLE IF EXISTS staging_sellers CASCADE;
    CREATE TABLE staging_sellers (
        seller_id VARCHAR(50),
        seller_city VARCHAR(100),
        seller_state VARCHAR(10)
    );
""", "staging_sellers criada")

executar_sql("""
    DROP TABLE IF EXISTS staging_trends CASCADE;
    CREATE TABLE staging_trends (
        full_date DATE,
        product_category_name VARCHAR(100),
        search_volume NUMERIC(6,2)
    );
""", "staging_trends criada")

executar_sql("""
    DROP TABLE IF EXISTS staging_reviews CASCADE;
    CREATE TABLE staging_reviews (
        product_category_name VARCHAR(100),
        full_date DATE,
        average_score NUMERIC(3,2),
        total_comments INT,
        sentiment_score NUMERIC(6,2),
        review_momentum NUMERIC(8,4)
    );
""", "staging_reviews criada")

# ── 1C. Inserir dados nas tabelas de staging ────────────────────
print("\n3. A inserir dados no staging...")

df_orders[['order_id','customer_id','order_status',
           'order_purchase_timestamp',
           'order_delivered_customer_date',
           'order_estimated_delivery_date']].to_sql(
    'staging_orders', engine, if_exists='append', index=False)
print(f"   ✅ staging_orders: {len(df_orders):,} linhas")

df_items[['order_id','order_item_id','product_id',
          'seller_id','price','freight_value']].to_sql(
    'staging_order_items', engine, if_exists='append', index=False)
print(f"   ✅ staging_order_items: {len(df_items):,} linhas")

df_products[['product_id','product_category_name']].to_sql(
    'staging_products', engine, if_exists='append', index=False)
print(f"   ✅ staging_products: {len(df_products):,} linhas")

df_customers[['customer_id','customer_unique_id',
              'customer_city','customer_state']].to_sql(
    'staging_customers', engine, if_exists='append', index=False)
print(f"   ✅ staging_customers: {len(df_customers):,} linhas")

df_sellers[['seller_id','seller_city','seller_state']].to_sql(
    'staging_sellers', engine, if_exists='append', index=False)
print(f"   ✅ staging_sellers: {len(df_sellers):,} linhas")

# Trends — garantir coluna full_date como DATE
df_trends_raw['full_date'] = pd.to_datetime(
    df_trends_raw['full_date']).dt.date
df_trends_raw[['full_date','product_category_name','search_volume']].to_sql(
    'staging_trends', engine, if_exists='append', index=False)
print(f"   ✅ staging_trends: {len(df_trends_raw):,} linhas")

# Reviews sintéticas
df_reviews_sint['full_date'] = pd.to_datetime(
    df_reviews_sint['full_date']).dt.date
df_reviews_sint[['product_category_name','full_date',
                 'average_score','total_comments',
                 'sentiment_score','review_momentum']].to_sql(
    'staging_reviews', engine, if_exists='append', index=False)
print(f"   ✅ staging_reviews: {len(df_reviews_sint):,} linhas")

# ════════════════════════════════════════════════════════════════
# PASSO 2 — ETL: TRANSFORMAÇÃO
# Limpa e normaliza os dados para o Star Schema
# ════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  PASSO 2 — ETL (TRANSFORMAÇÃO)")
print("="*60)

print("1. A construir dimensões e factos...")

# ── dim_time ─────────────────────────────────────────────────────
executar_sql("""
    DROP TABLE IF EXISTS dim_time CASCADE;
    CREATE TABLE dim_time AS
    SELECT DISTINCT
        ROW_NUMBER() OVER (ORDER BY d::DATE) AS time_sk,
        d::DATE                               AS full_date,
        EXTRACT(YEAR  FROM d)::INT            AS year,
        EXTRACT(MONTH FROM d)::INT            AS month,
        EXTRACT(QUARTER FROM d)::INT          AS quarter,
        TO_CHAR(d, 'Month')                   AS month_name,
        EXTRACT(DOW FROM d)::INT              AS day_of_week
    FROM (
        SELECT DISTINCT DATE(order_purchase_timestamp) AS d
        FROM staging_orders
        WHERE order_purchase_timestamp IS NOT NULL
    ) t;
    ALTER TABLE dim_time ADD PRIMARY KEY (time_sk);
""", "dim_time criada")

# ── dim_product ───────────────────────────────────────────────────
executar_sql(f"""
    DROP TABLE IF EXISTS dim_product CASCADE;
    CREATE TABLE dim_product AS
    SELECT DISTINCT
        ROW_NUMBER() OVER (ORDER BY sp.product_id) AS product_sk,
        sp.product_id,
        COALESCE(sp.product_category_name, 'desconhecido')
            AS product_category_name
    FROM staging_products sp;
    ALTER TABLE dim_product ADD PRIMARY KEY (product_sk);
""", "dim_product criada")

# ── dim_customer ──────────────────────────────────────────────────
executar_sql("""
    DROP TABLE IF EXISTS dim_customer CASCADE;
    CREATE TABLE dim_customer AS
    SELECT DISTINCT
        ROW_NUMBER() OVER (ORDER BY customer_id) AS customer_sk,
        customer_id,
        customer_unique_id,
        customer_city,
        customer_state
    FROM staging_customers;
    ALTER TABLE dim_customer ADD PRIMARY KEY (customer_sk);
""", "dim_customer criada")

# ── dim_seller ────────────────────────────────────────────────────
executar_sql("""
    DROP TABLE IF EXISTS dim_seller CASCADE;
    CREATE TABLE dim_seller AS
    SELECT DISTINCT
        ROW_NUMBER() OVER (ORDER BY seller_id) AS seller_sk,
        seller_id,
        seller_city,
        seller_state
    FROM staging_sellers;
    ALTER TABLE dim_seller ADD PRIMARY KEY (seller_sk);
""", "dim_seller criada")

# ── [NOVO] dim_trend ──────────────────────────────────────────────
executar_sql("""
    DROP TABLE IF EXISTS dim_trend CASCADE;
    CREATE TABLE dim_trend AS
    SELECT
        ROW_NUMBER() OVER (
            ORDER BY st.full_date, st.product_category_name
        ) AS trend_sk,
        dt.time_sk,
        st.product_category_name,
        st.search_volume,
        LAG(st.search_volume, 1) OVER (
            PARTITION BY st.product_category_name
            ORDER BY st.full_date
        ) AS search_lead_1,
        LAG(st.search_volume, 2) OVER (
            PARTITION BY st.product_category_name
            ORDER BY st.full_date
        ) AS search_lead_2,
        st.full_date
    FROM staging_trends st
    JOIN dim_time dt ON dt.full_date = st.full_date;
    ALTER TABLE dim_trend ADD PRIMARY KEY (trend_sk);
""", "dim_trend criada (com search_lead_1 e search_lead_2)")

# ── [NOVO] dim_review ─────────────────────────────────────────────
executar_sql("""
    DROP TABLE IF EXISTS dim_review CASCADE;
    CREATE TABLE dim_review AS
    SELECT
        ROW_NUMBER() OVER (
            ORDER BY sr.full_date, sr.product_category_name
        ) AS review_sk,
        dt.time_sk,
        sr.product_category_name,
        sr.average_score,
        sr.total_comments,
        sr.sentiment_score,
        sr.review_momentum,
        sr.full_date
    FROM staging_reviews sr
    JOIN dim_time dt ON dt.full_date = sr.full_date;
    ALTER TABLE dim_review ADD PRIMARY KEY (review_sk);
""", "dim_review criada (com sentiment_score e review_momentum)")

# ── fact_sales ────────────────────────────────────────────────────
executar_sql("""
    DROP TABLE IF EXISTS fact_sales CASCADE;
    CREATE TABLE fact_sales AS
    SELECT
        ROW_NUMBER() OVER () AS sales_sk,
        dt.time_sk,
        dp.product_sk,
        dc.customer_sk,
        ds.seller_sk,
        si.price,
        si.freight_value,
        si.price + si.freight_value AS total_value
    FROM staging_order_items si
    JOIN staging_orders      so ON si.order_id   = so.order_id
    JOIN dim_time            dt ON dt.full_date  = DATE(so.order_purchase_timestamp)
    JOIN dim_product         dp ON dp.product_id = si.product_id
    JOIN dim_customer        dc ON dc.customer_id = so.customer_id
    JOIN dim_seller          ds ON ds.seller_id   = si.seller_id
    WHERE so.order_status = 'delivered';
    ALTER TABLE fact_sales ADD PRIMARY KEY (sales_sk);
""", "fact_sales criada")

# Verificação
with engine.connect() as conn:
    n_facts = conn.execute(text("SELECT COUNT(*) FROM fact_sales")).scalar()
    n_trend = conn.execute(text("SELECT COUNT(*) FROM dim_trend")).scalar()
    n_review = conn.execute(text("SELECT COUNT(*) FROM dim_review")).scalar()
print(f"\n   fact_sales : {n_facts:,} registos")
print(f"   dim_trend  : {n_trend:,} registos")
print(f"   dim_review : {n_review:,} registos")

# ════════════════════════════════════════════════════════════════
# PASSO 3 — CDC: TRIGGER DE AUDITORIA
# ════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  PASSO 3 — CDC (CHANGE DATA CAPTURE)")
print("="*60)

executar_sql("""
    DROP TABLE IF EXISTS cdc_audit_log;
    CREATE TABLE cdc_audit_log (
        log_id        SERIAL PRIMARY KEY,
        tabela_nome   VARCHAR(100),
        operacao      VARCHAR(10),
        sales_sk      BIGINT,
        time_sk       INT,
        product_sk    INT,
        price         NUMERIC(10,2),
        criado_em     TIMESTAMP DEFAULT NOW()
    );
""", "cdc_audit_log criada")

executar_sql("""
    CREATE OR REPLACE FUNCTION fn_cdc_fact_sales()
    RETURNS TRIGGER AS $$
    BEGIN
        INSERT INTO cdc_audit_log
            (tabela_nome, operacao, sales_sk, time_sk, product_sk, price)
        VALUES
            (TG_TABLE_NAME, TG_OP,
             NEW.sales_sk, NEW.time_sk, NEW.product_sk, NEW.price);
        RETURN NEW;
    END;
    $$ LANGUAGE plpgsql;
""", "função CDC criada")

executar_sql("""
    DROP TRIGGER IF EXISTS trg_cdc_fact_sales ON fact_sales;
    CREATE TRIGGER trg_cdc_fact_sales
    AFTER INSERT ON fact_sales
    FOR EACH ROW EXECUTE FUNCTION fn_cdc_fact_sales();
""", "trigger CDC activo em fact_sales")

# View de monitorização
executar_sql("""
    CREATE OR REPLACE VIEW vw_cdc_monitor AS
    SELECT
        tabela_nome,
        operacao,
        COUNT(*)          AS total_operacoes,
        MAX(criado_em)    AS ultima_operacao
    FROM cdc_audit_log
    GROUP BY tabela_nome, operacao
    ORDER BY ultima_operacao DESC;
""", "view vw_cdc_monitor criada")

# ════════════════════════════════════════════════════════════════
# PASSO 4 — IA FAROL
# Lê TUDO do DW via SQL — zero ficheiros locais
# ════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  PASSO 4 — IA FAROL (lê tudo do DW via SQL)")
print("="*60)

print("1. A extrair features do DW...")

# Query principal: junta fact_sales + dim_product + dim_time
#                 + dim_trend + dim_review — TUDO VIA SQL
query_ml = """
    SELECT
        fs.time_sk,
        fs.price,
        dp.product_category_name,
        dt.month,
        dt.quarter,
        -- Google Trends (dim_trend)
        COALESCE(tr.search_volume,  50) AS search_volume,
        COALESCE(tr.search_lead_1,  50) AS search_lead_1,
        COALESCE(tr.search_lead_2,  50) AS search_lead_2,
        -- Reviews (dim_review)
        COALESCE(rv.sentiment_score,   50) AS sentiment_score,
        COALESCE(rv.review_momentum,    0) AS review_momentum,
        COALESCE(rv.average_score,    3.5) AS average_score
    FROM fact_sales fs
    JOIN dim_product  dp ON dp.product_sk = fs.product_sk
    JOIN dim_time     dt ON dt.time_sk    = fs.time_sk
    LEFT JOIN dim_trend  tr ON tr.time_sk = fs.time_sk
                           AND tr.product_category_name = dp.product_category_name
    LEFT JOIN dim_review rv ON rv.time_sk = fs.time_sk
                           AND rv.product_category_name = dp.product_category_name
"""

df_ml_raw = pd.read_sql(query_ml, engine)
print(f"   ✅ {len(df_ml_raw):,} registos extraídos do DW")

# ── Feature Engineering ──────────────────────────────────────────
print("2. A calcular métricas de tendência...")

df_ml_raw = df_ml_raw.sort_values(['product_category_name', 'time_sk'])

grp = df_ml_raw.groupby(['product_category_name', 'time_sk'])

vendas_resumo = grp.agg(
    vendas_vol      = ('price', 'count'),
    receita_total   = ('price', 'sum'),
    search_volume   = ('search_volume', 'mean'),
    search_lead_1   = ('search_lead_1', 'mean'),
    search_lead_2   = ('search_lead_2', 'mean'),
    sentiment_score = ('sentiment_score', 'mean'),
    review_momentum = ('review_momentum', 'mean'),
    mes             = ('month', 'first'),
    trimestre       = ('quarter', 'first'),
).reset_index()

g = vendas_resumo.groupby('product_category_name')

vendas_resumo['moving_avg_3'] = g['vendas_vol'].transform(
    lambda x: x.rolling(3, min_periods=1).mean())
vendas_resumo['growth_rate'] = g['moving_avg_3'].pct_change().fillna(0)
vendas_resumo['acceleration'] = g['growth_rate'].diff().fillna(0)
vendas_resumo['lag_1'] = g['vendas_vol'].shift(1).fillna(0)
vendas_resumo['search_momentum'] = g['search_volume'].pct_change().fillna(0)
vendas_resumo['sazonalidade_sin'] = np.sin(
    2 * np.pi * vendas_resumo['mes'] / 12)
vendas_resumo['sazonalidade_cos'] = np.cos(
    2 * np.pi * vendas_resumo['mes'] / 12)

vendas_resumo['farol_score'] = (
    0.35 * vendas_resumo['search_momentum'].clip(-1, 1) +
    0.25 * vendas_resumo['acceleration'].clip(-1, 1) +
    0.25 * vendas_resumo['review_momentum'].clip(-1, 1) +
    0.15 * vendas_resumo['growth_rate'].clip(-1, 1)
)

le = LabelEncoder()
vendas_resumo['categoria_encoded'] = le.fit_transform(
    vendas_resumo['product_category_name'])

def classificar_tendencia(row):
    if row['growth_rate'] > 0 and row['acceleration'] > 0:
        return 'Aceleracao'
    elif row['growth_rate'] > 0 and row['acceleration'] <= 0:
        return 'Crescimento Estavel'
    elif row['growth_rate'] <= 0 and row['acceleration'] > 0:
        return 'Recuperacao'
    else:
        return 'Declinio'

vendas_resumo['fase_tendencia'] = vendas_resumo.apply(
    classificar_tendencia, axis=1)

# ── Treino do modelo ─────────────────────────────────────────────
print("3. A treinar XGBoost Farol com TimeSeriesSplit...")

FEATURES = [
    'search_volume', 'search_lead_1', 'search_lead_2',
    'search_momentum', 'sentiment_score', 'review_momentum',
    'growth_rate', 'acceleration', 'farol_score',
    'lag_1', 'moving_avg_3',
    'mes', 'trimestre', 'sazonalidade_sin', 'sazonalidade_cos',
    'categoria_encoded'
]

df_sorted = vendas_resumo.sort_values('time_sk').dropna(
    subset=FEATURES + ['vendas_vol'])
X = df_sorted[FEATURES]
y = df_sorted['vendas_vol']

tscv = TimeSeriesSplit(n_splits=5)
split_idx = int(len(df_sorted) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

resultados = {}

rf = RandomForestRegressor(n_estimators=300, max_depth=15,
                            random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
cv_rf = cross_val_score(rf, X, y, cv=tscv, scoring='r2')
resultados['RF Farol'] = {
    'r2':  r2_score(y_test, rf.predict(X_test)),
    'mae': mean_absolute_error(y_test, rf.predict(X_test)),
    'cv':  cv_rf.mean(), 'cv_std': cv_rf.std()
}

if XGBOOST_OK:
    xgb_model = xgb.XGBRegressor(
        n_estimators=400, max_depth=5, learning_rate=0.08,
        subsample=0.85, colsample_bytree=0.85,
        reg_alpha=0.1, reg_lambda=1.5,
        random_state=42, verbosity=0)
    xgb_model.fit(X_train, y_train,
                  eval_set=[(X_test, y_test)], verbose=False)
    cv_xgb = cross_val_score(xgb_model, X, y, cv=tscv, scoring='r2')
    resultados['XGBoost Farol'] = {
        'r2':  r2_score(y_test, xgb_model.predict(X_test)),
        'mae': mean_absolute_error(y_test, xgb_model.predict(X_test)),
        'cv':  cv_xgb.mean(), 'cv_std': cv_xgb.std()
    }

melhor_nome = max(resultados, key=lambda k: resultados[k]['r2'])
melhor = resultados[melhor_nome]
modelo_final = xgb_model if (melhor_nome == 'XGBoost Farol'
                              and XGBOOST_OK) else rf

feature_importance = pd.Series(
    modelo_final.feature_importances_, index=FEATURES
).sort_values(ascending=False)

r2_treino = r2_score(y_train, modelo_final.predict(X_train))
gap = r2_treino - melhor['r2']

# ── FAROL: top categorias prestes a explodir ─────────────────────
farol_df = vendas_resumo.groupby('product_category_name').agg(
    search_momentum_avg=('search_momentum', 'mean'),
    farol_score_avg    =('farol_score',     'mean'),
    acceleration_avg   =('acceleration',    'mean'),
    vendas_recentes    =('vendas_vol',      'mean')
).reset_index()

for col in ['search_momentum_avg', 'farol_score_avg', 'acceleration_avg']:
    mn, mx = farol_df[col].min(), farol_df[col].max()
    if mx > mn:
        farol_df[col+'_norm'] = (farol_df[col]-mn)/(mx-mn)*100

farol_df['alerta'] = (
    farol_df.get('search_momentum_avg_norm', 0) * 0.40 +
    farol_df.get('farol_score_avg_norm',     0) * 0.35 +
    farol_df.get('acceleration_avg_norm',    0) * 0.25
)
top_farol  = farol_df.nlargest(5, 'alerta')
top_growth = (vendas_resumo.groupby('product_category_name')['growth_rate']
              .mean().sort_values(ascending=False).head(5))
top_accel  = (vendas_resumo.groupby('product_category_name')['acceleration']
              .mean().sort_values(ascending=False).head(5))

# ════════════════════════════════════════════════════════════════
# OUTPUT FINAL
# ════════════════════════════════════════════════════════════════
print("\n" + "="*62)
print("        TRENDMART 2026 — MOTOR PREDITIVO FAROL")
print("="*62)
print(f"  Registos analisados  : {len(df_ml_raw):>10,}")
print(f"  Categorias           : {vendas_resumo['product_category_name'].nunique():>10,}")
print(f"  Modelo vencedor      : {melhor_nome}")
print(f"  R² (teste)           : {melhor['r2']:.3f}")
print(f"  R² CV time-series    : {melhor['cv']:.3f} ± {melhor['cv_std']:.3f}")
print(f"  MAE                  : {melhor['mae']:.1f}")
print(f"  Gap overfitting      : {gap:.3f}  {'✅ Saudável' if gap < 0.08 else '⚠️ Atenção'}")
print("-"*62)

print("\n📊 COMPARAÇÃO DE MODELOS:")
for nome, res in resultados.items():
    m = " ← VENCEDOR" if nome == melhor_nome else ""
    print(f"   {nome:<22} R²={res['r2']:.3f}  CV={res['cv']:.3f}±{res['cv_std']:.3f}  MAE={res['mae']:.1f}{m}")

print("\n📈 TOP 5 — MAIOR CRESCIMENTO:")
for cat, val in top_growth.items():
    print(f"   {cat:<38} {val:+.4f}")

print("\n🚀 TOP 5 — MAIOR ACELERAÇÃO:")
for cat, val in top_accel.items():
    print(f"   {cat:<38} {val:+.4f}")

print("\n🔦 FAROL — TOP 5 CATEGORIAS PRESTES A EXPLODIR:")
print(f"   {'Categoria':<38} {'Alerta':>7}  {'Search Mom.':>12}")
print(f"   {'-'*38} {'-'*7}  {'-'*12}")
for _, row in top_farol.iterrows():
    print(f"   {row['product_category_name']:<38} "
          f"{row['alerta']:>6.1f}%  "
          f"{row['search_momentum_avg']:>+12.4f}")

print("\n🔬 IMPORTÂNCIA DAS FEATURES:")
for feat, imp in feature_importance.head(10).items():
    barra = '█' * int(imp * 40)
    print(f"   {feat:<22} {imp:.3f}  {barra}")

print("="*62)

# ════════════════════════════════════════════════════════════════
# DASHBOARD
# ════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(18, 14))
fig.suptitle('TrendMart — Motor Preditivo FAROL 2026\n'
             '(Pipeline completa: Staging → ETL → DW → IA)',
             fontsize=15, fontweight='bold')
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

ax1 = fig.add_subplot(gs[0, 0])
top_growth.plot(kind='barh', ax=ax1, color='gold')
ax1.set_title('Top 5 — Taxa de Crescimento')
ax1.set_xlabel('Taxa Média de Crescimento')

ax2 = fig.add_subplot(gs[0, 1])
top_accel.plot(kind='barh', ax=ax2, color='limegreen')
ax2.set_title('Top 5 — Aceleração da Procura')
ax2.set_xlabel('Aceleração Média')

ax3 = fig.add_subplot(gs[0, 2])
cores = ['#FF4136','#FF851B','#FFDC00','#2ECC40','#0074D9']
ax3.barh(top_farol['product_category_name'],
         top_farol['alerta'], color=cores)
ax3.axvline(x=50, color='red', linestyle='--', alpha=0.5, label='Limiar 50%')
ax3.set_title('FAROL — Alertas de Tendência Emergente')
ax3.set_xlabel('Score de Alerta (%)')
ax3.legend(fontsize=8)

ax4 = fig.add_subplot(gs[1, 0])
feature_importance.head(10).plot(kind='bar', ax=ax4, color='steelblue')
ax4.set_title(f'Importância das Features\n({melhor_nome})')
ax4.set_ylabel('Importância')
ax4.tick_params(axis='x', rotation=35)

ax5 = fig.add_subplot(gs[1, 1])
nomes   = list(resultados.keys())
r2_vals = [resultados[n]['r2'] for n in nomes]
cv_vals = [resultados[n]['cv'] for n in nomes]
x = np.arange(len(nomes))
w = 0.35
ax5.bar(x-w/2, r2_vals, w, label='R² (teste)',    color='steelblue')
ax5.bar(x+w/2, cv_vals, w, label='R² CV (ts)',    color='coral')
ax5.axhline(y=0.80, color='red', linestyle='--', linewidth=1, label='Meta 0.80')
ax5.set_xticks(x)
ax5.set_xticklabels(nomes, rotation=15, fontsize=8)
ax5.set_title('Teste vs Validação Cronológica')
ax5.set_ylabel('R²')
ax5.legend(fontsize=8)
ax5.set_ylim(0, 1.05)

ax6 = fig.add_subplot(gs[1, 2])
df_cat = vendas_resumo[
    vendas_resumo['product_category_name'] == 'la_cuisine'].copy()
if len(df_cat) > 0:
    ax6b = ax6.twinx()
    ax6.plot(df_cat['time_sk'], df_cat['vendas_vol'],
             color='steelblue', linewidth=1.5, label='Vendas')
    ax6b.plot(df_cat['time_sk'], df_cat['search_momentum'],
              color='orange', linestyle='--', linewidth=1.5,
              label='Search Momentum')
    ax6.set_title('la_cuisine: Vendas vs Search Momentum')
    ax6.set_ylabel('Vendas', color='steelblue')
    ax6b.set_ylabel('Search Momentum', color='orange')
    l1, lb1 = ax6.get_legend_handles_labels()
    l2, lb2 = ax6b.get_legend_handles_labels()
    ax6.legend(l1+l2, lb1+lb2, fontsize=8, loc='upper left')

plt.savefig('tendencias_2026.png', dpi=150, bbox_inches='tight')
print("\n💾 Dashboard 'tendencias_2026.png' gerado com sucesso!")
print("\n✅ PIPELINE COMPLETA CONCLUÍDA")
print("   Staging → ETL → DW (Star Schema + dim_trend + dim_review)")
print("   CDC activo → cdc_audit_log + vw_cdc_monitor")
print("   XGBoost Farol → leu tudo do DW via SQL")