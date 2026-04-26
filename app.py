"""
app.py — Simulador Fotovoltaico | Lucas do Rio Verde/MT
Sistema Web desenvolvido em Python + Streamlit
Projeto Integrador — BCT / UFMT

Disciplinas integradas:
  • Física III           — efeito fotovoltaico, eficiência dos módulos
  • Probabilidade e Est. — médias históricas de radiação com IC de 95%
  • Matemática Financeira — VPL, TIR, Payback, juros compostos, depreciação
  • Cálculo III          — otimização de ângulo de inclinação (função multivariável)
  • Gestão do Conhecimento — documentação e boas práticas
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Módulos internos do projeto
from data import (
    CIDADE, MESES, DIAS_POR_MES, IRRADIANCIA_MENSAL,
    TARIFA_ENERGIA_KWH, CUSTO_POR_KWP, VIDA_UTIL_ANOS,
    TAXA_DESCONTO, INFLACAO_ENERGIA_AA, HSP_MEDIA_ANUAL,
)
from calculations import (
    calcular_potencia_sistema,
    calcular_geracao_mensal,
    angulo_otimo,
    resumo_perdas,
    perda_por_temperatura,
)
from financial import (
    calcular_investimento,
    calcular_fluxo_caixa,
    calcular_payback,
    calcular_vpl,
    calcular_tir,
    economia_mensal_ano1,
    co2_evitado,
)

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURAÇÃO DA PÁGINA
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="SolarSim — Lucas do Rio Verde/MT",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS customizado ──────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;700&display=swap');

:root {
    --sol: #F5A623;
    --sol-light: #FFD580;
    --verde: #27AE60;
    --azul: #1A6B9A;
    --bg: #0D1117;
    --card: #161B22;
    --border: #30363D;
    --text: #E6EDF3;
    --muted: #8B949E;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

.stApp { background-color: var(--bg); }

h1, h2, h3 { font-family: 'Space Mono', monospace !important; }

.metric-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 12px;
    transition: border-color .2s;
}
.metric-card:hover { border-color: var(--sol); }
.metric-card .label { font-size: 0.75rem; color: var(--muted); letter-spacing:.1em; text-transform:uppercase; }
.metric-card .value { font-size: 2rem; font-weight: 700; color: var(--sol); font-family:'Space Mono',monospace; }
.metric-card .sub   { font-size: 0.8rem; color: var(--muted); margin-top:2px; }

.tag {
    display:inline-block; padding:2px 10px; border-radius:20px;
    font-size:.7rem; font-weight:700; letter-spacing:.05em;
    margin-bottom:4px;
}
.tag-fisica   { background:#1A3A5C; color:#64B5F6; }
.tag-prob     { background:#1A3A2A; color:#81C784; }
.tag-fin      { background:#3A2A0A; color:#FFD54F; }
.tag-calc     { background:#3A1A2A; color:#CE93D8; }

.section-header {
    font-family:'Space Mono',monospace;
    font-size:1.1rem; font-weight:700;
    border-left:4px solid var(--sol);
    padding-left:12px;
    margin: 28px 0 16px;
}

div[data-testid="stMetric"] > div { background:var(--card) !important; border-radius:10px; padding:16px !important; }

.stSidebar { background-color: #0D1117 !important; border-right: 1px solid var(--border); }

.highlight-box {
    background: linear-gradient(135deg, #1a2a1a, #1a3a0a);
    border: 1px solid var(--verde);
    border-radius:10px; padding:16px; margin:12px 0;
}
.warning-box {
    background: linear-gradient(135deg, #2a1a0a, #3a2a0a);
    border: 1px solid var(--sol);
    border-radius:10px; padding:16px; margin:12px 0;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PALETA PLOTLY
# ══════════════════════════════════════════════════════════════════════════════
PLOTLY_TEMPLATE = dict(
    layout=go.Layout(
        paper_bgcolor="#161B22",
        plot_bgcolor="#0D1117",
        font=dict(family="DM Sans", color="#E6EDF3"),
        xaxis=dict(gridcolor="#30363D", zerolinecolor="#30363D"),
        yaxis=dict(gridcolor="#30363D", zerolinecolor="#30363D"),
        legend=dict(bgcolor="#161B22", bordercolor="#30363D"),
    )
)

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — INPUTS
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("# ☀️ SolarSim")
    st.caption(f"📍 {CIDADE}")
    st.markdown("---")

    st.markdown("### 📊 Dados de Consumo")
    consumo_kwh = st.number_input(
        "Consumo médio mensal (kWh)",
        min_value=50, max_value=5000, value=350, step=10,
        help="Média dos últimos 12 meses da sua conta de energia"
    )

    valor_conta = st.number_input(
        "Valor médio da conta (R$)",
        min_value=50.0, max_value=5000.0, value=310.0, step=10.0,
        help="Usado para validar a tarifa local"
    )

    st.markdown("---")
    st.markdown("### ⚙️ Parâmetros Avançados")

    tarifa_custom = st.number_input(
        "Tarifa de energia (R$/kWh)",
        min_value=0.40, max_value=2.00,
        value=TARIFA_ENERGIA_KWH, step=0.01,
        format="%.2f"
    )

    inflacao_custom = st.slider(
        "Inflação da energia (% a.a.)",
        min_value=1.0, max_value=20.0,
        value=float(INFLACAO_ENERGIA_AA * 100), step=0.5
    ) / 100

    taxa_desconto_custom = st.slider(
        "Taxa de desconto / SELIC (% a.a.)",
        min_value=5.0, max_value=20.0,
        value=float(TAXA_DESCONTO * 100), step=0.5
    ) / 100

    st.markdown("---")
    st.caption("💡 Dados de radiação: CRESESB/INPE — série 1994–2020")
    st.caption("🔌 Tarifa: ENERGISA MT — Subgrupo B1 Residencial 2025")


# ══════════════════════════════════════════════════════════════════════════════
# CÁLCULOS PRINCIPAIS
# ══════════════════════════════════════════════════════════════════════════════
# Atualiza constantes financeiras com valores do sidebar
import financial as fin
import data as dados
dados.TARIFA_ENERGIA_KWH   = tarifa_custom
dados.INFLACAO_ENERGIA_AA  = inflacao_custom
dados.TAXA_DESCONTO        = taxa_desconto_custom
fin.TARIFA_ENERGIA_KWH     = tarifa_custom
fin.INFLACAO_ENERGIA_AA    = inflacao_custom
fin.TAXA_DESCONTO          = taxa_desconto_custom

# Executa todos os cálculos
pot = calcular_potencia_sistema(consumo_kwh)
ger = calcular_geracao_mensal(pot["potencia_real_kWp"])
inv = calcular_investimento(pot["potencia_real_kWp"])
fc  = calcular_fluxo_caixa(ger["media_anual"], consumo_kwh, inv["custo_total"])
pb  = calcular_payback(fc["acumulado"], fc["fluxo_descontado"], inv["custo_total"])
vpl = calcular_vpl(fc["fluxo_descontado"], inv["custo_total"])
tir = calcular_tir(fc["fluxo_liquido"], inv["custo_total"])
eco = economia_mensal_ano1(ger["geracao_kwh"], consumo_kwh)
co2 = co2_evitado(ger["media_anual"])
ang = angulo_otimo()
per = resumo_perdas()

# ══════════════════════════════════════════════════════════════════════════════
# CABEÇALHO
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="text-align:center; padding:32px 0 16px;">
    <div style="font-size:3rem;">☀️</div>
    <h1 style="font-size:2rem; margin:8px 0 4px; color:#F5A623;">
        Simulador Fotovoltaico
    </h1>
    <p style="color:#8B949E; font-size:1rem; margin:0;">
        Lucas do Rio Verde / MT &nbsp;·&nbsp; Projeto Integrador BCT/UFMT
    </p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SEÇÃO 1 — KPIs PRINCIPAIS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">📐 Dimensionamento do Sistema</div>', unsafe_allow_html=True)
st.caption('<span class="tag tag-fisica">Física III</span> Efeito fotovoltaico · Potência de pico · Performance Ratio', unsafe_allow_html=True)

c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="label">Potência do Sistema</div>
        <div class="value">{pot['potencia_real_kWp']:.2f}</div>
        <div class="sub">kWp instalados</div>
    </div>""", unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="label">Painéis Necessários</div>
        <div class="value">{pot['n_modulos']}</div>
        <div class="sub">módulos 550 Wp</div>
    </div>""", unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="label">Área de Telhado</div>
        <div class="value">{pot['area_total_m2']}</div>
        <div class="sub">m² necessários</div>
    </div>""", unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="label">Geração Anual</div>
        <div class="value">{ger['media_anual']:,.0f}</div>
        <div class="sub">kWh/ano estimados</div>
    </div>""", unsafe_allow_html=True)

with c5:
    cobertura = min(ger['media_anual'] / (consumo_kwh * 12) * 100, 100)
    st.markdown(f"""
    <div class="metric-card">
        <div class="label">Cobertura do Consumo</div>
        <div class="value">{cobertura:.0f}%</div>
        <div class="sub">da demanda anual</div>
    </div>""", unsafe_allow_html=True)

# Performance Ratio
st.markdown(f"""
<div class="warning-box">
    <b>⚡ Performance Ratio (PR) = {pot['fator_desempenho']*100:.1f}%</b><br>
    <small>Perdas: Inversor {per['Inversor']}% · Cabeamento {per['Cabeamento']}% · 
    Sombreamento {per['Sombreamento']}% · Sujeira {per['Sujeira']}% · 
    Temperatura {per['Temperatura']}% (ΔT = +{TEMP_OPERACAO_LOCAL - TEMP_REFERENCIA}°C acima do STC)</small>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SEÇÃO 2 — KPIs FINANCEIROS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">💰 Análise Financeira</div>', unsafe_allow_html=True)
st.caption('<span class="tag tag-fin">Matemática Financeira</span> Payback · VPL · TIR · Juros Compostos · Depreciação', unsafe_allow_html=True)

f1, f2, f3, f4, f5 = st.columns(5)

with f1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="label">Investimento Total</div>
        <div class="value">R${inv['custo_total']:,.0f}</div>
        <div class="sub">R$ {CUSTO_POR_KWP:,.0f}/kWp</div>
    </div>""", unsafe_allow_html=True)

with f2:
    pb_s = pb['payback_simples_anos']
    st.markdown(f"""
    <div class="metric-card">
        <div class="label">Payback Simples</div>
        <div class="value">{pb_s if pb_s else '>25'}</div>
        <div class="sub">{'anos' if pb_s else 'anos (fora do período)'}</div>
    </div>""", unsafe_allow_html=True)

with f3:
    pb_d = pb['payback_descontado_anos']
    st.markdown(f"""
    <div class="metric-card">
        <div class="label">Payback Descontado</div>
        <div class="value">{pb_d if pb_d else '>25'}</div>
        <div class="sub">anos (taxa {taxa_desconto_custom*100:.1f}% a.a.)</div>
    </div>""", unsafe_allow_html=True)

with f4:
    cor_vpl = "#27AE60" if vpl["viavel"] else "#E74C3C"
    st.markdown(f"""
    <div class="metric-card">
        <div class="label">VPL (25 anos)</div>
        <div class="value" style="color:{cor_vpl};">R${vpl['vpl']:,.0f}</div>
        <div class="sub">{'✅ Viável' if vpl['viavel'] else '❌ Inviável'}</div>
    </div>""", unsafe_allow_html=True)

with f5:
    st.markdown(f"""
    <div class="metric-card">
        <div class="label">TIR</div>
        <div class="value">{tir}%</div>
        <div class="sub">a.a. (vs SELIC {taxa_desconto_custom*100:.1f}%)</div>
    </div>""", unsafe_allow_html=True)

# Economia mensal média (ano 1)
eco_media = np.mean(list(eco.values()))
st.markdown(f"""
<div class="highlight-box">
    <b>💚 Economia média mensal no 1º ano: R$ {eco_media:,.2f}</b> &nbsp;·&nbsp; 
    Economia total em 25 anos: <b>R$ {sum(fc['economias']):,.0f}</b>
    &nbsp;·&nbsp; 🌿 CO₂ evitado: <b>{co2['ton_co2_25anos']:.1f} toneladas</b> 
    em 25 anos (~{co2['arvores_eq']:,} árvores)
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SEÇÃO 3 — GRÁFICOS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">📈 Dashboard de Geração & Análise</div>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "☀️ Geração vs Consumo",
    "💸 Fluxo de Caixa",
    "🔬 Física do Sistema",
    "📊 Estatística",
    "🌿 Ambiental",
])

# ── Tab 1: Geração vs Consumo ─────────────────────────────────────────────────
with tab1:
    fig = go.Figure()

    # Consumo (linha constante)
    fig.add_trace(go.Scatter(
        x=MESES, y=[consumo_kwh] * 12,
        name="Consumo Mensal",
        line=dict(color="#E74C3C", width=2, dash="dash"),
        mode="lines",
    ))

    # Geração com IC de 95%
    ger_vals = [ger["geracao_kwh"][m] for m in MESES]
    ic_lo    = [ger["ic_lower"][m] for m in MESES]
    ic_hi    = [ger["ic_upper"][m] for m in MESES]

    fig.add_trace(go.Scatter(
        x=MESES + MESES[::-1],
        y=ic_hi + ic_lo[::-1],
        fill='toself',
        fillcolor='rgba(245,166,35,0.15)',
        line=dict(color='rgba(0,0,0,0)'),
        name='IC 95% (Estatística)',
        hoverinfo='skip',
    ))

    fig.add_trace(go.Bar(
        x=MESES, y=ger_vals,
        name="Geração Estimada",
        marker_color="#F5A623",
        opacity=0.85,
    ))

    # Economia
    eco_vals = [eco[m] for m in MESES]
    fig.add_trace(go.Scatter(
        x=MESES, y=ger_vals,
        name="Geração (linha)",
        line=dict(color="#F5A623", width=3),
        mode="lines+markers",
        marker=dict(size=8),
    ))

    fig.update_layout(
        **PLOTLY_TEMPLATE["layout"].to_plotly_json(),
        title="Geração Estimada vs Consumo Mensal (kWh)",
        barmode="overlay",
        height=420,
        legend=dict(orientation="h", y=-0.2),
        yaxis_title="kWh",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Tabela detalhada
    df_ger = pd.DataFrame({
        "Mês": MESES,
        "Geração (kWh)": [ger["geracao_kwh"][m] for m in MESES],
        "IC Inf. 95%":   [ger["ic_lower"][m] for m in MESES],
        "IC Sup. 95%":   [ger["ic_upper"][m] for m in MESES],
        "Consumo (kWh)": [consumo_kwh] * 12,
        "Saldo (kWh)":   [round(ger["geracao_kwh"][m] - consumo_kwh, 1) for m in MESES],
        "Economia (R$)": [eco[m] for m in MESES],
    })
    st.dataframe(df_ger, use_container_width=True, hide_index=True)

# ── Tab 2: Fluxo de Caixa ─────────────────────────────────────────────────────
with tab2:
    anos = fc["anos"]
    acum = fc["acumulado"]

    fig2 = make_subplots(
        rows=2, cols=1,
        subplot_titles=["Fluxo de Caixa Líquido Anual (R$)", "Fluxo Acumulado — Payback (R$)"],
        vertical_spacing=0.12
    )

    # Barras do fluxo anual
    cores_fl = ["#E74C3C" if v < 0 else "#27AE60" for v in fc["fluxo_liquido"]]
    fig2.add_trace(go.Bar(
        x=[f"Ano {a}" for a in anos],
        y=fc["fluxo_liquido"],
        marker_color=cores_fl,
        name="Fluxo Líquido",
        showlegend=True,
    ), row=1, col=1)

    # Acumulado
    cores_ac = ["#E74C3C" if v < 0 else "#27AE60" for v in acum]
    fig2.add_trace(go.Scatter(
        x=[f"Ano {a}" for a in anos],
        y=acum,
        mode="lines+markers",
        line=dict(color="#F5A623", width=3),
        marker=dict(color=cores_ac, size=6),
        name="Acumulado",
    ), row=2, col=1)

    # Linha de payback
    fig2.add_hline(y=0, line_dash="dash", line_color="#E74C3C",
                   annotation_text="Ponto de Equilíbrio", row=2, col=1)

    if pb["payback_simples_anos"]:
        fig2.add_vline(
            x=f"Ano {pb['payback_simples_anos']}",
            line_dash="dot", line_color="#F5A623",
            annotation_text=f"Payback: Ano {pb['payback_simples_anos']}",
            row=2, col=1,
        )

    fig2.update_layout(
        **PLOTLY_TEMPLATE["layout"].to_plotly_json(),
        height=560,
        showlegend=True,
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Composição do investimento
    col_a, col_b = st.columns(2)
    with col_a:
        labels = ["Módulos", "Inversor", "Estrutura", "Instalação", "Outros"]
        values = [inv["custo_modulos"], inv["custo_inversor"],
                  inv["custo_estrutura"], inv["custo_instalacao"], inv["custo_outros"]]
        fig_pie = go.Figure(go.Pie(
            labels=labels, values=values,
            hole=0.4,
            marker_colors=["#F5A623", "#FFD580", "#F0A500", "#E8902A", "#D4770A"],
        ))
        fig_pie.update_layout(
            **PLOTLY_TEMPLATE["layout"].to_plotly_json(),
            title="Composição do Investimento",
            height=320,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_b:
        st.markdown("**Detalhamento do Investimento**")
        df_inv = pd.DataFrame({
            "Componente": labels,
            "Valor (R$)": [f"R$ {v:,.2f}" for v in values],
            "Proporção": [f"{v/inv['custo_total']*100:.1f}%" for v in values],
        })
        st.dataframe(df_inv, use_container_width=True, hide_index=True)

        st.markdown(f"""
        | Indicador | Valor |
        |-----------|-------|
        | **VPL** | R$ {vpl['vpl']:,.2f} |
        | **TIR** | {tir}% a.a. |
        | **SELIC ref.** | {taxa_desconto_custom*100:.1f}% a.a. |
        | **Payback simples** | {pb['payback_simples_anos']} anos |
        | **Payback descontado** | {pb['payback_descontado_anos']} anos |
        | **Vida útil** | {VIDA_UTIL_ANOS} anos |
        """)

# ── Tab 3: Física do Sistema ──────────────────────────────────────────────────
with tab3:
    st.caption('<span class="tag tag-fisica">Física III</span> Efeito Fotovoltaico · Temperatura · Otimização de Inclinação (Cálculo III)', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Diagrama de Perdas do Sistema (Sankey simplificado)**")
        perdas = resumo_perdas()
        labels_p = list(perdas.keys())[:-1]
        vals_p   = [perdas[k] for k in labels_p]

        fig_bar = go.Figure(go.Bar(
            x=labels_p,
            y=vals_p,
            marker_color=["#E74C3C", "#E67E22", "#E74C3C", "#C0392B", "#E74C3C"],
            text=[f"{v}%" for v in vals_p],
            textposition="outside",
        ))
        fig_bar.update_layout(
            **PLOTLY_TEMPLATE["layout"].to_plotly_json(),
            title=f"Perdas do Sistema (Total: {perdas['Total']}%)",
            yaxis_title="Perda (%)",
            height=320,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown("**Equação do Performance Ratio (PR):**")
        st.latex(r"PR = 1 - (\eta_{inv} + \eta_{cab} + \eta_{som} + \eta_{suj} + \alpha_T \cdot \Delta T)")
        st.latex(rf"PR = 1 - ({perdas['Total']/100:.4f}) = {pot['fator_desempenho']:.4f}")

    with col2:
        st.markdown("**Curva de Irradiância vs Ângulo de Inclinação (Cálculo III)**")
        betas = ang["betas"]
        irrs  = ang["irradiancias"]

        fig_ang = go.Figure()
        fig_ang.add_trace(go.Scatter(
            x=betas, y=irrs,
            mode="lines",
            line=dict(color="#F5A623", width=3),
            fill="tozeroy",
            fillcolor="rgba(245,166,35,0.15)",
        ))
        fig_ang.add_vline(
            x=ang["angulo_otimo_graus"],
            line_dash="dash", line_color="#27AE60",
            annotation_text=f"β* = {ang['angulo_otimo_graus']}°",
        )
        fig_ang.update_layout(
            **PLOTLY_TEMPLATE["layout"].to_plotly_json(),
            title="Otimização do Ângulo de Inclinação",
            xaxis_title="Ângulo β (graus)",
            yaxis_title="Irradiância média (kWh/m²/dia)",
            height=320,
        )
        st.plotly_chart(fig_ang, use_container_width=True)

        st.markdown("**Equação de otimização (Cálculo III):**")
        st.latex(r"I(\beta, \gamma) = I_{horiz} \cdot [\cos(\varphi - \beta) \cdot \cos(\gamma)]")
        st.latex(rf"\beta^* = {ang['angulo_otimo_graus']:.1f}° \approx |\varphi| = {abs(LATITUDE)}°")
        st.markdown(f"> Ângulo ótimo ≈ latitude local (|φ| = {abs(LATITUDE)}°) — validado pela teoria.")

    # Efeito da temperatura
    st.markdown("---")
    st.markdown("**Perda por Temperatura (Física III — Coeficiente de Temperatura)**")
    temps = np.arange(25, 80, 1)
    perdas_temp = [perda_por_temperatura(float(t)) for t in temps]
    fig_temp = go.Figure(go.Scatter(
        x=temps, y=perdas_temp,
        mode="lines",
        line=dict(color="#E74C3C", width=2),
        fill="tozeroy",
        fillcolor="rgba(231,76,60,0.1)",
    ))
    fig_temp.add_vline(x=TEMP_OPERACAO_LOCAL, line_dash="dash", line_color="#F5A623",
                       annotation_text=f"T_op = {TEMP_OPERACAO_LOCAL}°C ({per['Temperatura']}% perda)")
    fig_temp.update_layout(
        **PLOTLY_TEMPLATE["layout"].to_plotly_json(),
        title="Perda de Potência por Temperatura (αT = -0,35%/°C)",
        xaxis_title="Temperatura de Operação (°C)",
        yaxis_title="Perda (%)",
        height=280,
    )
    st.plotly_chart(fig_temp, use_container_width=True)
    st.latex(r"\Delta P = \alpha_T \cdot (T_{op} - T_{STC}) = -0.0035 \cdot (T_{op} - 25°C)")

# ── Tab 4: Estatística ────────────────────────────────────────────────────────
with tab4:
    st.caption('<span class="tag tag-prob">Probabilidade e Estatística</span> Médias Históricas · IC 95% · Distribuição de Geração', unsafe_allow_html=True)

    irr_vals = list(IRRADIANCIA_MENSAL.values())
    media_irr = np.mean(irr_vals)
    desvio_irr = np.std(irr_vals)

    col_s1, col_s2, col_s3 = st.columns(3)
    col_s1.metric("HSP Médio Anual", f"{media_irr:.2f} kWh/m²/dia")
    col_s2.metric("Desvio Padrão",   f"{desvio_irr:.2f} kWh/m²/dia")
    col_s3.metric("Coef. de Variação", f"{desvio_irr/media_irr*100:.1f}%")

    fig_stat = go.Figure()
    fig_stat.add_trace(go.Bar(
        x=MESES, y=irr_vals,
        name="HSP Médio",
        marker_color="#F5A623",
    ))
    fig_stat.add_hline(y=media_irr, line_dash="dash", line_color="#27AE60",
                       annotation_text=f"Média = {media_irr:.2f} kWh/m²/dia")
    fig_stat.add_hline(y=media_irr + desvio_irr, line_dash="dot",
                       line_color="#8B949E", annotation_text="+1σ")
    fig_stat.add_hline(y=media_irr - desvio_irr, line_dash="dot",
                       line_color="#8B949E", annotation_text="-1σ")
    fig_stat.update_layout(
        **PLOTLY_TEMPLATE["layout"].to_plotly_json(),
        title="Irradiância Solar Mensal — Lucas do Rio Verde/MT (CRESESB)",
        yaxis_title="HSP (kWh/m²/dia)",
        height=360,
    )
    st.plotly_chart(fig_stat, use_container_width=True)

    # Distribuição simulada de geração anual (TCL)
    st.markdown("**Distribuição da Geração Anual — Teorema Central do Limite (n=20 anos)**")
    media_anual_ger = ger["media_anual"]
    sigma_ger = media_anual_ger * 0.06  # ~6% CV
    x_dist = np.linspace(media_anual_ger - 3*sigma_ger, media_anual_ger + 3*sigma_ger, 300)
    y_dist = (1/(sigma_ger * np.sqrt(2*np.pi))) * np.exp(-0.5*((x_dist - media_anual_ger)/sigma_ger)**2)

    fig_dist = go.Figure()
    fig_dist.add_trace(go.Scatter(
        x=x_dist, y=y_dist,
        mode="lines", line=dict(color="#F5A623", width=2),
        fill="tozeroy", fillcolor="rgba(245,166,35,0.2)",
        name="Distribuição Normal"
    ))
    # IC 95%
    lo_95 = media_anual_ger - 1.96*sigma_ger
    hi_95 = media_anual_ger + 1.96*sigma_ger
    fig_dist.add_vrect(x0=lo_95, x1=hi_95, fillcolor="rgba(39,174,96,0.1)",
                       layer="below", line_width=0, annotation_text="IC 95%")
    fig_dist.add_vline(x=media_anual_ger, line_dash="dash", line_color="#27AE60",
                       annotation_text=f"μ = {media_anual_ger:.0f} kWh/ano")
    fig_dist.update_layout(
        **PLOTLY_TEMPLATE["layout"].to_plotly_json(),
        title=f"Distribuição de Geração Anual — IC 95%: [{lo_95:.0f}, {hi_95:.0f}] kWh",
        xaxis_title="Geração Anual (kWh)",
        yaxis_title="Densidade de Probabilidade",
        height=300,
    )
    st.plotly_chart(fig_dist, use_container_width=True)

    st.latex(r"\bar{X} \pm z_{0.025} \cdot \frac{\sigma}{\sqrt{n}} \quad \text{(Teorema Central do Limite)}")
    st.markdown(f"**Intervalo de Confiança 95%:** [{lo_95:,.0f} ; {hi_95:,.0f}] kWh/ano")

# ── Tab 5: Ambiental ──────────────────────────────────────────────────────────
with tab5:
    c_a, c_b, c_c = st.columns(3)
    c_a.metric("CO₂ evitado/ano",    f"{co2['kg_co2_ano']:,.0f} kg")
    c_b.metric("CO₂ evitado (25 anos)", f"{co2['ton_co2_25anos']:.1f} t")
    c_c.metric("Equivalente em árvores", f"{co2['arvores_eq']:,} 🌳")

    anos_lbl = [f"Ano {a}" for a in fc["anos"]]
    co2_acum = [co2["kg_co2_ano"] * a / 1000 for a in fc["anos"]]  # toneladas acumuladas

    fig_co2 = go.Figure(go.Scatter(
        x=anos_lbl, y=co2_acum,
        mode="lines+markers",
        line=dict(color="#27AE60", width=3),
        fill="tozeroy", fillcolor="rgba(39,174,96,0.15)",
        marker=dict(size=6),
        name="CO₂ evitado acumulado",
    ))
    fig_co2.update_layout(
        **PLOTLY_TEMPLATE["layout"].to_plotly_json(),
        title="CO₂ Evitado Acumulado ao Longo de 25 Anos (toneladas)",
        yaxis_title="CO₂ (t)",
        height=300,
    )
    st.plotly_chart(fig_co2, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# SEÇÃO 4 — FÓRMULAS E DOCUMENTAÇÃO (Gestão do Conhecimento)
# ══════════════════════════════════════════════════════════════════════════════
with st.expander("📚 Fundamentação Teórica e Fórmulas Utilizadas — (Gestão do Conhecimento)"):
    st.markdown("""
### 🔬 Física III — Efeito Fotovoltaico
""")
    st.latex(r"P_{pico} = \frac{E_{mes}}{HSP_{med} \times PR \times 30}")
    st.latex(r"PR = 1 - (\eta_{inv} + \eta_{cab} + \eta_{som} + \eta_{suj} + \alpha_T \cdot \Delta T)")
    st.latex(r"G_{mes} = P_{kWp} \times HSP_{mes} \times d_{mes} \times PR")

    st.markdown("### 📊 Probabilidade e Estatística")
    st.latex(r"\bar{X}_{HSP} = \frac{1}{12}\sum_{i=1}^{12} HSP_i \qquad \sigma = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(x_i-\bar{x})^2}")
    st.latex(r"IC_{95\%} = \bar{X} \pm z_{0.025} \cdot \frac{\sigma}{\sqrt{n}}, \quad z_{0.025}=1.96")

    st.markdown("### 💰 Matemática Financeira")
    st.latex(r"VPL = -C_0 + \sum_{t=1}^{N} \frac{FC_t}{(1+i)^t}")
    st.latex(r"TIR: \quad VPL(TIR) = 0 \implies \sum_{t=0}^{N} \frac{FC_t}{(1+TIR)^t} = 0")
    st.latex(r"E_t = G_t \times \tau_0 \times (1+\pi_e)^t, \quad G_t = G_0 \times (1-\delta)^t")

    st.markdown("### 📐 Cálculo III — Otimização")
    st.latex(r"I(\beta, \gamma) = I_{horiz} \cdot \cos(\varphi - \beta) \cdot \cos(\gamma)")
    st.latex(r"\frac{dI}{d\beta} = 0 \implies \beta^* = |\varphi| \approx 13°")

    st.markdown("""
### 🗂️ Gestão do Conhecimento
- **Dados de radiação:** CRESESB/INPE — Atlas Brasileiro de Energia Solar, 2ª ed. (série 1994–2020)
- **Tarifa:** ANEEL/ENERGISA-MT — Subgrupo B1 Residencial (2025)
- **Fator de emissão CO₂:** ONS — Fator Médio de Emissão da SIN (2023)
- **Código:** Modularizado em `data.py`, `calculations.py`, `financial.py` e `app.py`
- **Licença:** MIT — livre para uso acadêmico e extensão por futuros alunos do BCT/UFMT
""")

# ══════════════════════════════════════════════════════════════════════════════
# RODAPÉ
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.caption(
    "☀️ **SolarSim** · Projeto Integrador BCT/UFMT · Lucas do Rio Verde/MT · "
    "Física III · Prob. e Estatística · Mat. Financeira · Cálculo III · Gestão do Conhecimento"
)
