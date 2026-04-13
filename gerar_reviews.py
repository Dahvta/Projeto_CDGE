"""
╔══════════════════════════════════════════════════════════════════╗
║       GERADOR DE REVIEWS SINTÉTICAS — VERSÃO REALISTA           ║
║  Simula padrões reais: sentimento temporal, review momentum,    ║
║  correlação com tendências de mercado                           ║
╚══════════════════════════════════════════════════════════════════╝
"""

import pandas as pd
import numpy as np
import json
from sqlalchemy import create_engine
from datetime import timedelta

np.random.seed(42)

# ============================================================
# 1. LIGAÇÃO E EXTRAÇÃO
# ============================================================
motor_bd = create_engine('postgresql://admin:admin@localhost:5432/pl_cgde_store')

query_sql = """
    SELECT DISTINCT dt.full_date, dt.time_sk, dp.product_category_name
    FROM fact_sales fs
    JOIN dim_time dt ON fs.time_sk = dt.time_sk
    JOIN dim_product dp ON fs.product_sk = dp.product_sk
    WHERE dp.product_category_name IS NOT NULL
    ORDER BY dt.full_date, dp.product_category_name
"""

print("1. A ler categorias e datas do PostgreSQL...")
df_base = pd.read_sql(query_sql, motor_bd)
df_base['full_date'] = pd.to_datetime(df_base['full_date'])
df_base = df_base.sort_values(['product_category_name', 'full_date']).reset_index(drop=True)

categorias = df_base['product_category_name'].unique()
print(f"   ✅ {len(categorias)} categorias × {df_base['full_date'].nunique()} datas")

# ============================================================
# 2. PERFIS DE SENTIMENTO POR CATEGORIA
# ============================================================
# Categorias emergentes têm sentimento a SUBIR (sinal preditivo)
# Categorias em declínio têm sentimento a DESCER
# Categorias estáveis têm sentimento consistente

perfis_reviews = {
    # Produto bom, clientes satisfeitos e a aumentar
    'emergente_positivo': [
        'la_cuisine', 'alimentos_bebidas', 'cine_foto',
        'construcao_ferramentas_jardim', 'casa_conforto_2',
        'sinalizacao_e_seguranca'
    ],
    # Produto sólido, reviews consistentes
    'estavel_alto': [
        'moveis_escritorio', 'informatica_acessorios',
        'pcs', 'telefonia', 'papelaria'
    ],
    # Produto de moda — pico e queda rápida
    'moda_ciclo': [
        'fashion_bolsas_e_acessorios', 'cool_stuff',
        'fashion_roupa', 'fashion_calcados'
    ],
    # Produto sazonal — reviews sobem e descem com a época
    'sazonal': [
        'brinquedos', 'bebidas', 'esporte_lazer',
        'natal_decoracoes'
    ],
    # Produto com problemas de qualidade conhecidos
    'declinio': [
        'seguros_e_servicos', 'fraldas_higiene',
        'portateis_cozinha_e_banho'
    ],
}

def get_perfil_review(categoria):
    for perfil, lista in perfis_reviews.items():
        if categoria in lista:
            return perfil
    return 'padrao'

# ============================================================
# 3. GERAÇÃO DE REVIEWS REALISTAS
# ============================================================
print("2. A gerar padrões de sentimento realistas por categoria...")

todos_records = []

for categoria in categorias:
    df_cat = df_base[df_base['product_category_name'] == categoria].copy()
    n = len(df_cat)
    if n == 0:
        continue

    perfil = get_perfil_review(categoria)
    t = np.linspace(0, 1, n)
    meses = df_cat['full_date'].dt.month.values

    # ── Score base por perfil ────────────────────────────────────
    if perfil == 'emergente_positivo':
        # Sentimento sobe gradualmente — produto a ganhar qualidade/reputação
        score_base = 3.2 + 1.4 * (t ** 0.8)          # sobe de 3.2 → ~4.6
        volume_base = 20 + 180 * (t ** 1.2)           # reviews aumentam muito
        sentimento_trend = 0.6 * t                     # momentum positivo

    elif perfil == 'estavel_alto':
        # Consistentemente bom, pouca variação
        score_base = 4.1 + 0.3 * np.sin(2 * np.pi * t)
        volume_base = 80 + 40 * np.random.rand(n)
        sentimento_trend = np.zeros(n)

    elif perfil == 'moda_ciclo':
        # Pico a meio do período depois cai
        score_base = 3.5 + 1.2 * np.sin(np.pi * t)   # sobe e desce
        volume_base = 30 + 200 * np.sin(np.pi * t)    # mesmo padrão no volume
        sentimento_trend = np.cos(np.pi * t) * 0.4

    elif perfil == 'sazonal':
        # Pico em Dez (mês 12) e Verão (mês 7)
        sazon = np.sin(2 * np.pi * (meses - 3) / 12)
        score_base = 3.8 + 0.7 * sazon
        volume_base = 50 + 150 * np.clip(sazon, 0, 1)
        sentimento_trend = 0.2 * sazon

    elif perfil == 'declinio':
        # Sentimento a cair — produto com problemas
        score_base = 4.0 - 1.5 * t                    # cai de 4.0 → 2.5
        volume_base = 100 - 60 * t                     # menos reviews com o tempo
        sentimento_trend = -0.5 * t

    else:  # padrao
        score_base = 3.5 + 0.5 * np.sin(2 * np.pi * t)
        volume_base = 40 + 60 * np.random.rand(n)
        sentimento_trend = np.zeros(n)

    # ── Ruído suave (não puro) ───────────────────────────────────
    ruido_score  = np.convolve(np.random.normal(0, 0.15, n+4), np.ones(4)/4, 'valid')[:n]
    ruido_volume = np.convolve(np.random.normal(0, 8, n+4),    np.ones(4)/4, 'valid')[:n]

    # ── Score final (clampado entre 1.0 e 5.0) ──────────────────
    average_score = np.clip(score_base + ruido_score, 1.0, 5.0)

    # ── Volume de reviews ────────────────────────────────────────
    total_comments = np.clip(volume_base + ruido_volume, 1, 2000).astype(int)

    # ── Review Momentum (variação do sentimento) ─────────────────
    # Sinal preditivo: se o sentimento acelera, as vendas vão subir
    review_momentum = np.diff(average_score, prepend=average_score[0])

    # ── Score de sentimento normalizado 0-100 ────────────────────
    sentiment_score = (average_score - 1) / 4 * 100  # 1→0%, 5→100%

    # ── Construir records ────────────────────────────────────────
    for i, (_, row) in enumerate(df_cat.iterrows()):
        record = {
            "product_category_name": categoria,
            "full_date":             row['full_date'].strftime('%Y-%m-%d'),
            "time_sk":               int(row['time_sk']),
            "average_score":         round(float(average_score[i]), 2),
            "total_comments":        int(total_comments[i]),
            "sentiment_score":       round(float(sentiment_score[i]), 2),
            "review_momentum":       round(float(review_momentum[i]), 4),
            "perfil":                perfil
        }
        todos_records.append(record)

# ============================================================
# 4. GUARDAR EM JSON
# ============================================================
nome_ficheiro = 'reviews_sinteticas.json'
with open(nome_ficheiro, 'w', encoding='utf-8') as f:
    json.dump(todos_records, f, indent=2, ensure_ascii=False)

# ============================================================
# 5. VALIDAÇÃO
# ============================================================
df_val = pd.DataFrame(todos_records)

print("\n3. Validação dos padrões gerados:")
print(f"   {'Categoria':<35} {'Score Médio':>11}  {'Reviews Médias':>14}  {'Momentum':>10}  {'Perfil'}")
print(f"   {'-'*35} {'-'*11}  {'-'*14}  {'-'*10}  {'-'*20}")

cats_exemplo = [
    'la_cuisine', 'moveis_escritorio', 'brinquedos',
    'bebidas', 'sinalizacao_e_seguranca', 'pcs',
    'fashion_bolsas_e_acessorios'
]
for cat in cats_exemplo:
    sub = df_val[df_val['product_category_name'] == cat]
    if len(sub) > 0:
        perfil = get_perfil_review(cat)
        print(f"   {cat:<35} {sub['average_score'].mean():>11.2f}  "
              f"{sub['total_comments'].mean():>14.0f}  "
              f"{sub['review_momentum'].mean():>+10.4f}  {perfil}")

print(f"\n✅ Ficheiro '{nome_ficheiro}' gerado!")
print(f"   Registos  : {len(todos_records):,}")
print(f"   Categorias: {df_val['product_category_name'].nunique()}")
print(f"   Período   : {df_val['full_date'].min()} → {df_val['full_date'].max()}")
print(f"\n   Novos campos preditivos adicionados:")
print(f"   → sentiment_score  : sentimento normalizado 0-100")
print(f"   → review_momentum  : variação do sentimento (sinal antecipado)")
print(f"   → la_cuisine marcada como 'emergente_positivo' — score a subir! 🔦")