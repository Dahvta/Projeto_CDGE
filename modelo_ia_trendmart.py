"""
╔══════════════════════════════════════════════════════════════════╗
║         TRENDMART 2026 — MOTOR PREDITIVO "FAROL" v4.0           ║
║  Objetivo: Detetar tendências ANTES de as vendas subirem        ║
║  Abordagem: Sinais externos > Autocorrelação                    ║
╚══════════════════════════════════════════════════════════════════╝
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

from sqlalchemy import create_engine
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.inspection import permutation_importance

try:
    import xgboost as xgb
    XGBOOST_DISPONIVEL = True
except ImportError:
    XGBOOST_DISPONIVEL = False
    print("⚠️  XGBoost não instalado. Corre: pip install xgboost")

# ============================================================
# 1. LIGAÇÃO AO DATA WAREHOUSE
# ============================================================
motor_bd = create_engine('postgresql://admin:admin@localhost:5432/pl_cgde_store')

# ============================================================
# 2. EXTRAÇÃO (Arquitetura Heterogénea: SQL + CSV)
# ============================================================
print("1. A extrair dados do Data Warehouse (PostgreSQL)...")
query_vendas = """
    SELECT fs.price, fs.time_sk, dp.product_category_name 
    FROM fact_sales fs 
    JOIN dim_product dp ON fs.product_sk = dp.product_sk
"""
df_vendas = pd.read_sql(query_vendas, motor_bd)
df_trends = pd.read_csv('google_trends_sintetico.csv')

print(f"   ✅ {len(df_vendas):,} registos de vendas extraídos.")
print(f"   ✅ {len(df_trends):,} registos de Google Trends extraídos.")

# ============================================================
# 3. AGREGAÇÃO BASE
# ============================================================
print("\n2. A construir pipeline de features preditivas (modo FAROL)...")

vendas_resumo = (
    df_vendas
    .groupby(['product_category_name', 'time_sk'])
    .agg(
        vendas_vol=('price', 'count'),
        receita_total=('price', 'sum')
    )
    .reset_index()
    .sort_values(['product_category_name', 'time_sk'])
)

# ============================================================
# 4. FEATURE ENGINEERING PREDITIVA
# ============================================================

grp = vendas_resumo.groupby('product_category_name')

# ── Médias móveis CURTAS (janela=3) — espelho retrovisor mínimo ──
vendas_resumo['moving_avg_3'] = grp['vendas_vol'].transform(
    lambda x: x.rolling(window=3, min_periods=1).mean()
)

# ── Taxa de Crescimento e Aceleração ──────────────────────────────
vendas_resumo['growth_rate'] = (
    grp['moving_avg_3'].pct_change().fillna(0)
)
vendas_resumo['acceleration'] = (
    grp['growth_rate'].diff().fillna(0)
)

# ── [MELHORIA 1] SEARCH LEAD: Google Trends ANTECIPADO ───────────
# Cruzar Google Trends por categoria e período
if 'time_sk' in df_trends.columns:
    trends_full = df_trends[['product_category_name', 'time_sk', 'search_volume']].copy()
    vendas_resumo = vendas_resumo.merge(
        trends_full, on=['product_category_name', 'time_sk'], how='left'
    )
    vendas_resumo['search_volume'] = vendas_resumo['search_volume'].fillna(
        vendas_resumo.groupby('product_category_name')['search_volume'].transform('mean')
    )
else:
    trends_avg = df_trends.groupby('product_category_name')['search_volume'].mean().reset_index()
    vendas_resumo = vendas_resumo.merge(trends_avg, on='product_category_name', how='left')

grp2 = vendas_resumo.groupby('product_category_name')

# Search Lead: volume de pesquisa de 1 período atrás → antecipa vendas de hoje
vendas_resumo['search_lead_1'] = grp2['search_volume'].shift(1).fillna(
    vendas_resumo['search_volume']
)
# Search Lead 2: dois períodos atrás (ciclos mais longos)
vendas_resumo['search_lead_2'] = grp2['search_volume'].shift(2).fillna(
    vendas_resumo['search_volume']
)
# Search Momentum: aceleração do hype online
vendas_resumo['search_momentum'] = grp2['search_volume'].pct_change().fillna(0)

# ── [MELHORIA 2] PENALIZAÇÃO DE AUTOCORRELAÇÃO ───────────────────
# Lag 1 curto (mantemos para estabilidade mas não adicionamos lag_6)
vendas_resumo['lag_1'] = grp2['vendas_vol'].shift(1).fillna(0)
# REMOVEMOS moving_avg_6 — era o que "abafava" os sinais externos

# ── [MELHORIA 3] DECOMPOSIÇÃO DE SAZONALIDADE ────────────────────
# Extrai mês e trimestre do time_sk para capturar ciclos de negócio
# time_sk é inteiro sequencial — estimamos mês e trimestre relativos
vendas_resumo['mes'] = ((vendas_resumo['time_sk'] - 1) % 12) + 1
vendas_resumo['trimestre'] = ((vendas_resumo['mes'] - 1) // 3) + 1
# Componente cíclica: o modelo aprende que Dez ≈ Jan em comportamento
vendas_resumo['sazonalidade_sin'] = np.sin(2 * np.pi * vendas_resumo['mes'] / 12)
vendas_resumo['sazonalidade_cos'] = np.cos(2 * np.pi * vendas_resumo['mes'] / 12)

# Encoding da categoria
le = LabelEncoder()
vendas_resumo['categoria_encoded'] = le.fit_transform(
    vendas_resumo['product_category_name']
)

# ── SCORE DE TENDÊNCIA EMERGENTE (o "Farol") ─────────────────────
# Combinação ponderada: dá mais peso aos sinais externos
vendas_resumo['farol_score'] = (
    0.40 * vendas_resumo['search_momentum'].clip(-1, 1) +
    0.35 * vendas_resumo['acceleration'].clip(-1, 1) +
    0.25 * vendas_resumo['growth_rate'].clip(-1, 1)
)

# Classificação de fase (para o dashboard)
def classificar_tendencia(row):
    if row['growth_rate'] > 0 and row['acceleration'] > 0:
        return 'Aceleracao'
    elif row['growth_rate'] > 0 and row['acceleration'] <= 0:
        return 'Crescimento Estavel'
    elif row['growth_rate'] <= 0 and row['acceleration'] > 0:
        return 'Recuperacao'
    else:
        return 'Declinio'

vendas_resumo['fase_tendencia'] = vendas_resumo.apply(classificar_tendencia, axis=1)

# ============================================================
# 5. FEATURES FINAIS — foco nos sinais preditivos externos
# ============================================================
FEATURES = [
    # Sinais externos (o "farol")
    'search_volume',       # hype atual
    'search_lead_1',       # hype de 1 período atrás → antecipa vendas
    'search_lead_2',       # hype de 2 períodos atrás
    'search_momentum',     # aceleração do hype online
    # Métricas de tendência
    'growth_rate',
    'acceleration',
    'farol_score',         # score composto preditivo
    # Contexto mínimo (sem moving_avg_6 para não abafar)
    'lag_1',
    'moving_avg_3',        # só janela curta
    # Sazonalidade
    'mes',
    'trimestre',
    'sazonalidade_sin',
    'sazonalidade_cos',
    # Identidade da categoria
    'categoria_encoded',
]

df_final = vendas_resumo.dropna(subset=FEATURES + ['vendas_vol']).copy()

X = df_final[FEATURES]
y = df_final['vendas_vol']

# ============================================================
# 6. [MELHORIA 4] TIME-SERIES SPLIT — validação cronológica
# ============================================================
print("3. A treinar com validação cronológica (TimeSeriesSplit)...")

# Ordenar por tempo antes de dividir — NUNCA ver o futuro no treino
df_final_sorted = df_final.sort_values('time_sk').reset_index(drop=True)
X_sorted = df_final_sorted[FEATURES]
y_sorted = df_final_sorted['vendas_vol']

# TimeSeriesSplit: treino sempre antes do teste, respeitando ordem cronológica
tscv = TimeSeriesSplit(n_splits=5)

# Divisão treino/teste: últimos 20% do tempo como teste
split_idx = int(len(df_final_sorted) * 0.8)
X_train = X_sorted.iloc[:split_idx]
X_test  = X_sorted.iloc[split_idx:]
y_train = y_sorted.iloc[:split_idx]
y_test  = y_sorted.iloc[split_idx:]

resultados = {}

# ── Random Forest (baseline limpo) ───────────────────────────────
rf = RandomForestRegressor(
    n_estimators=300,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
cv_rf = cross_val_score(rf, X_sorted, y_sorted, cv=tscv, scoring='r2')
y_pred_rf = rf.predict(X_test)
resultados['RF Farol'] = {
    'r2':    r2_score(y_test, y_pred_rf),
    'mae':   mean_absolute_error(y_test, y_pred_rf),
    'cv_r2': cv_rf.mean(),
    'cv_std': cv_rf.std()
}

# ── XGBoost Farol ─────────────────────────────────────────────────
if XGBOOST_DISPONIVEL:
    xgb_model = xgb.XGBRegressor(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=1.5,
        random_state=42,
        verbosity=0
    )
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    cv_xgb = cross_val_score(xgb_model, X_sorted, y_sorted, cv=tscv, scoring='r2')
    y_pred_xgb = xgb_model.predict(X_test)
    resultados['XGBoost Farol'] = {
        'r2':    r2_score(y_test, y_pred_xgb),
        'mae':   mean_absolute_error(y_test, y_pred_xgb),
        'cv_r2': cv_xgb.mean(),
        'cv_std': cv_xgb.std()
    }

# ── Escolher modelo final ─────────────────────────────────────────
melhor_nome = max(resultados, key=lambda k: resultados[k]['r2'])
melhor_r2   = resultados[melhor_nome]['r2']
melhor_mae  = resultados[melhor_nome]['mae']
melhor_cv   = resultados[melhor_nome]['cv_r2']
melhor_std  = resultados[melhor_nome]['cv_std']

modelo_final = xgb_model if (melhor_nome == 'XGBoost Farol' and XGBOOST_DISPONIVEL) else rf

feature_importance = pd.Series(
    modelo_final.feature_importances_,
    index=FEATURES
).sort_values(ascending=False)

r2  = melhor_r2
mae = melhor_mae

# Anti-overfitting check
r2_treino = r2_score(y_train, modelo_final.predict(X_train))
gap_overfit = r2_treino - r2

# ============================================================
# 7. DETEÇÃO DE TENDÊNCIAS EMERGENTES (o produto final)
# ============================================================

# Top categorias por cada métrica
top_growth = (
    vendas_resumo.groupby('product_category_name')['growth_rate']
    .mean().sort_values(ascending=False).head(5)
)
top_accel = (
    vendas_resumo.groupby('product_category_name')['acceleration']
    .mean().sort_values(ascending=False).head(5)
)

# FAROL: categorias com alto search_momentum mas vendas ainda baixas
# → estas são as que vão explodir
farol_df = vendas_resumo.groupby('product_category_name').agg(
    search_momentum_avg=('search_momentum', 'mean'),
    farol_score_avg=('farol_score', 'mean'),
    vendas_recentes=('vendas_vol', 'mean'),
    acceleration_avg=('acceleration', 'mean')
).reset_index()

# Normalizar para score 0-100
for col in ['search_momentum_avg', 'farol_score_avg', 'acceleration_avg']:
    mn, mx = farol_df[col].min(), farol_df[col].max()
    if mx > mn:
        farol_df[col + '_norm'] = (farol_df[col] - mn) / (mx - mn) * 100

farol_df['alerta_emergencia'] = (
    farol_df.get('search_momentum_avg_norm', 0) * 0.45 +
    farol_df.get('farol_score_avg_norm', 0) * 0.35 +
    farol_df.get('acceleration_avg_norm', 0) * 0.20
)

top_farol = farol_df.nlargest(5, 'alerta_emergencia')[
    ['product_category_name', 'alerta_emergencia', 'search_momentum_avg', 'vendas_recentes']
]

# ============================================================
# 8. OUTPUT FINAL
# ============================================================
r2  = melhor_r2
mae = melhor_mae

print("\n" + "=" * 60)
print("        SISTEMA DE ANÁLISE DE TENDÊNCIAS 2026 — FAROL")
print("=" * 60)
print(f"  Registos analisados : {len(df_vendas):>10,}")
print(f"  Categorias detetadas: {df_vendas['product_category_name'].nunique():>10,}")
print(f"  Modelo vencedor     : {melhor_nome}")
print(f"  Precisão (teste)    : R² = {r2:.3f}  |  MAE = {mae:.1f}")
print(f"  Validação CV (5 fold time-series): R² = {melhor_cv:.3f} ± {melhor_std:.3f}")
print(f"  Gap overfitting     : {gap_overfit:.3f}  {'✅ Saudável' if gap_overfit < 0.08 else '⚠️  Atenção'}")
print("-" * 60)

print("\n📊 COMPARAÇÃO DE MODELOS:")
for nome, res in resultados.items():
    marcador = " ← VENCEDOR" if nome == melhor_nome else ""
    print(f"   {nome:<22} R²={res['r2']:.3f}  MAE={res['mae']:.1f}  CV={res['cv_r2']:.3f}±{res['cv_std']:.3f}{marcador}")
print("-" * 60)

print("\n📈 TOP 5 — MAIOR CRESCIMENTO:")
for cat, val in top_growth.items():
    print(f"   {cat:<35} {val:+.4f}")

print("\n🚀 TOP 5 — MAIOR ACELERAÇÃO (tendências emergentes):")
for cat, val in top_accel.items():
    print(f"   {cat:<35} {val:+.4f}")

print("\n🔦 FAROL — TOP 5 CATEGORIAS PRESTES A EXPLODIR:")
print(f"   {'Categoria':<35} {'Alerta':>8}  {'Search Mom.':>12}  {'Vendas Médias':>14}")
print(f"   {'-'*35} {'-'*8}  {'-'*12}  {'-'*14}")
for _, row in top_farol.iterrows():
    print(f"   {row['product_category_name']:<35} {row['alerta_emergencia']:>7.1f}%  {row['search_momentum_avg']:>+12.4f}  {row['vendas_recentes']:>14.1f}")

print("\n🔬 IMPORTÂNCIA DAS FEATURES NO MODELO:")
for feat, imp in feature_importance.items():
    barra = '█' * int(imp * 40)
    print(f"   {feat:<22} {imp:.3f}  {barra}")

print("=" * 60)

# ============================================================
# 9. DASHBOARD VISUAL
# ============================================================
fig = plt.figure(figsize=(18, 14))
fig.suptitle('TrendMart — Motor Preditivo FAROL 2026', fontsize=16, fontweight='bold')
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

# Gráfico 1: Top 5 Crescimento
ax1 = fig.add_subplot(gs[0, 0])
top_growth.plot(kind='barh', ax=ax1, color='gold')
ax1.set_title('Top 5 — Taxa de Crescimento')
ax1.set_xlabel('Taxa Média de Crescimento')

# Gráfico 2: Top 5 Aceleração
ax2 = fig.add_subplot(gs[0, 1])
top_accel.plot(kind='barh', ax=ax2, color='limegreen')
ax2.set_title('Top 5 — Aceleração da Procura')
ax2.set_xlabel('Aceleração Média')

# Gráfico 3: FAROL — categorias prestes a explodir
ax3 = fig.add_subplot(gs[0, 2])
cores_farol = ['#FF4136' if i == 0 else '#FF851B' if i == 1 else '#FFDC00'
               for i in range(len(top_farol))]
ax3.barh(top_farol['product_category_name'], top_farol['alerta_emergencia'],
         color=cores_farol)
ax3.set_title('FAROL — Alertas de Tendência Emergente')
ax3.set_xlabel('Score de Alerta (%)')
ax3.axvline(x=50, color='red', linestyle='--', alpha=0.5, label='Limiar 50%')
ax3.legend(fontsize=8)

# Gráfico 4: Importância das Features
ax4 = fig.add_subplot(gs[1, 0])
feature_importance.head(10).plot(kind='bar', ax=ax4, color='steelblue')
ax4.set_title(f'Importância das Features\n({melhor_nome})')
ax4.set_ylabel('Importância')
ax4.tick_params(axis='x', rotation=35)

# Gráfico 5: Comparação de Modelos
ax5 = fig.add_subplot(gs[1, 1])
nomes = list(resultados.keys())
r2_vals = [resultados[n]['r2'] for n in nomes]
cv_vals = [resultados[n]['cv_r2'] for n in nomes]
x = np.arange(len(nomes))
w = 0.35
ax5.bar(x - w/2, r2_vals, w, label='R² (teste)', color='steelblue')
ax5.bar(x + w/2, cv_vals, w, label='R² CV (time-series)', color='coral')
ax5.axhline(y=0.80, color='red', linestyle='--', linewidth=1, label='Meta 0.80')
ax5.set_title('Comparação: Teste vs Validação Cronológica')
ax5.set_xticks(x)
ax5.set_xticklabels(nomes, rotation=15, fontsize=8)
ax5.set_ylabel('R²')
ax5.legend(fontsize=8)
ax5.set_ylim(0, 1.05)

# Gráfico 6: Evolução search_momentum vs vendas para la_cuisine
ax6 = fig.add_subplot(gs[1, 2])
cat_exemplo = 'la_cuisine'
df_cat = vendas_resumo[vendas_resumo['product_category_name'] == cat_exemplo].copy()
if len(df_cat) > 0:
    ax6_twin = ax6.twinx()
    ax6.plot(df_cat['time_sk'], df_cat['vendas_vol'],
             color='steelblue', linewidth=1.5, label='Vendas')
    ax6_twin.plot(df_cat['time_sk'], df_cat['search_momentum'],
                  color='orange', linewidth=1.5, linestyle='--', label='Search Momentum')
    ax6.set_title(f'la_cuisine: Vendas vs Search Momentum\n(o FAROL antecipa a subida)')
    ax6.set_ylabel('Volume de Vendas', color='steelblue')
    ax6_twin.set_ylabel('Search Momentum', color='orange')
    lines1, labels1 = ax6.get_legend_handles_labels()
    lines2, labels2 = ax6_twin.get_legend_handles_labels()
    ax6.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='upper left')
else:
    ax6.text(0.5, 0.5, 'la_cuisine\nnão encontrada', ha='center', va='center',
             transform=ax6.transAxes, fontsize=10)
    ax6.set_title('la_cuisine: Sinal FAROL')

plt.savefig('tendencias_2026.png', dpi=150, bbox_inches='tight')
print("\n💾 Dashboard 'tendencias_2026.png' gerado com sucesso!")