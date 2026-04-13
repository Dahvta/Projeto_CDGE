"""
╔══════════════════════════════════════════════════════════════════╗
║     GERADOR DE GOOGLE TRENDS SINTÉTICO — VERSÃO REALISTA        ║
║  Simula padrões reais: sazonalidade, tendências emergentes,     ║
║  hype antecipado (search ANTES das vendas subirem)              ║
╚══════════════════════════════════════════════════════════════════╝
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine

np.random.seed(42)  # Reprodutibilidade

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

categorias   = df_base['product_category_name'].unique()
datas        = sorted(df_base['full_date'].unique())
n_datas      = len(datas)
print(f"   ✅ {len(categorias)} categorias × {n_datas} datas = {len(df_base):,} linhas")

# ============================================================
# 2. PERFIS DE COMPORTAMENTO POR CATEGORIA
# ============================================================
# Cada categoria tem um perfil diferente de sazonalidade e tendência.
# Isto é o que torna os dados úteis para o modelo ML.

perfis = {
    # Categorias com pico de Verão (meses 6-8)
    'verao': [
        'esporte_lazer', 'fashion_bolsas_e_acessorios', 'cool_stuff',
        'bebidas', 'alimentos', 'moda_praia'
    ],
    # Categorias com pico de Natal/fim de ano (meses 11-12)
    'natal': [
        'brinquedos', 'eletronicos', 'pcs', 'informatica_acessorios',
        'consoles_games', 'telefonia', 'relogios_presentes',
        'musica', 'livros_interesse_geral'
    ],
    # Categorias com pico no início do ano (meses 1-3)
    'ano_novo': [
        'esporte_e_lazer', 'fitness_academia', 'saude_beleza',
        'suplementos', 'dieta_saude'
    ],
    # Categorias estáveis (B2B / escritório)
    'estavel': [
        'moveis_escritorio', 'papelaria', 'informatica_acessorios',
        'telefonia_fixa', 'industria_comercio_e_negocios'
    ],
    # Categorias em ASCENSÃO CONTÍNUA (tendências emergentes reais)
    # → search sobe ANTES das vendas — é aqui que o FAROL funciona
    'emergente': [
        'la_cuisine', 'alimentos_bebidas', 'cine_foto',
        'construcao_ferramentas_jardim', 'casa_conforto_2',
        'sinalizacao_e_seguranca'
    ],
}

def get_perfil(categoria):
    for perfil, lista in perfis.items():
        if categoria in lista:
            return perfil
    return 'padrao'

# ============================================================
# 3. GERAÇÃO DE SEARCH VOLUME REALISTA
# ============================================================
print("2. A gerar padrões de hype realistas por categoria...")

resultados = []

for categoria in categorias:
    df_cat = df_base[df_base['product_category_name'] == categoria].copy()
    n = len(df_cat)
    if n == 0:
        continue

    perfil = get_perfil(categoria)
    meses  = df_cat['full_date'].dt.month.values
    t      = np.linspace(0, 1, n)  # progresso temporal 0→1

    # ── Base: ruído suave (não aleatório puro) ──────────────────
    ruido = np.convolve(
        np.random.normal(0, 3, n + 10),
        np.ones(10) / 10, mode='valid'
    )[:n]

    # ── Componente de sazonalidade anual ────────────────────────
    if perfil == 'verao':
        # Pico nos meses 6-8
        sazon = 20 * np.sin(2 * np.pi * (meses - 3) / 12)
    elif perfil == 'natal':
        # Pico nos meses 10-12
        sazon = 25 * np.sin(2 * np.pi * (meses - 6) / 12)
    elif perfil == 'ano_novo':
        # Pico em Jan-Fev
        sazon = 15 * np.cos(2 * np.pi * meses / 12)
    else:
        sazon = 8 * np.sin(2 * np.pi * meses / 12)

    # ── Componente de tendência de longo prazo ──────────────────
    if perfil == 'emergente':
        # CRESCIMENTO ACELERADO — simula uma tendência a emergir
        # A curva sobe mais depressa no final do período
        tendencia = 30 * (t ** 1.5)  # crescimento acelerado
        base = 25
    elif perfil == 'estavel':
        tendencia = 5 * t            # crescimento lento e constante
        base = 45
    elif perfil == 'natal':
        tendencia = 10 * t
        base = 35
    else:
        tendencia = 8 * t
        base = 30

    # ── [CHAVE] LEAD ANTECIPADO: hype sobe 2-4 semanas antes das vendas
    # Simulamos isto deslocando o sinal de sazonalidade para a esquerda
    # → o Google Trends vai subir ANTES das vendas
    lead_periodos = 3  # hype antecipa 3 períodos as vendas
    if perfil in ('emergente', 'natal', 'verao'):
        sazon_avancada = np.roll(sazon, -lead_periodos)
        sazon_avancada[-lead_periodos:] = sazon[-lead_periodos:]
        sazon = sazon_avancada

    # ── Composição final ─────────────────────────────────────────
    search_raw = base + tendencia + sazon + ruido

    # Normalizar para escala 0-100 (como o Google Trends real)
    mn, mx = search_raw.min(), search_raw.max()
    if mx > mn:
        search_norm = (search_raw - mn) / (mx - mn) * 90 + 5  # entre 5 e 95
    else:
        search_norm = np.full(n, 50.0)

    df_cat = df_cat.copy()
    df_cat['search_volume'] = np.round(search_norm, 2)

    resultados.append(df_cat)

df_trends_final = pd.concat(resultados, ignore_index=True)

# ============================================================
# 4. VALIDAÇÃO: confirma que os padrões fazem sentido
# ============================================================
print("\n3. Validação dos padrões gerados:")
print(f"   {'Categoria':<35} {'Média':>6}  {'Std':>6}  {'Min':>5}  {'Max':>5}  {'Perfil'}")
print(f"   {'-'*35} {'-'*6}  {'-'*6}  {'-'*5}  {'-'*5}  {'-'*10}")

cats_exemplo = ['la_cuisine', 'moveis_escritorio', 'brinquedos',
                'bebidas', 'sinalizacao_e_seguranca', 'pcs']

for cat in cats_exemplo:
    sub = df_trends_final[df_trends_final['product_category_name'] == cat]
    if len(sub) > 0:
        perfil = get_perfil(cat)
        print(f"   {cat:<35} {sub['search_volume'].mean():>6.1f}  "
              f"{sub['search_volume'].std():>6.1f}  "
              f"{sub['search_volume'].min():>5.1f}  "
              f"{sub['search_volume'].max():>5.1f}  {perfil}")

# ============================================================
# 5. GUARDAR
# ============================================================
colunas_output = ['full_date', 'time_sk', 'product_category_name', 'search_volume']
df_trends_final[colunas_output].to_csv('google_trends_sintetico.csv', index=False)

print(f"\n✅ Ficheiro 'google_trends_sintetico.csv' gerado!")
print(f"   Linhas    : {len(df_trends_final):,}")
print(f"   Categorias: {df_trends_final['product_category_name'].nunique()}")
print(f"   Período   : {df_trends_final['full_date'].min().date()} → "
      f"{df_trends_final['full_date'].max().date()}")
print(f"\n   La_cuisine está marcada como 'emergente' — o FAROL deve detetá-la! 🔦")