import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Dual Sourcing: Base-Surge Optimizer", layout="wide")

st.title("⚖️ Dual Sourcing: Base-Surge Newsvendor Optimizer")
st.markdown(
    "**Core concept:** A base supplier (cheap, slow) covers your *expected* demand. "
    "A surge supplier (expensive, fast) covers the *upside tail* — but only when the surge "
    "premium is cheaper than the cost of a lost sale. This tool finds that optimal split."
)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("📦 Product Economics")
    selling_price = st.number_input("Selling Price (£)", value=60.0, step=1.0)
    salvage_value = st.number_input("Salvage / Markdown Value (£)", value=15.0, step=1.0,
                                    help="What you recover per unit if forced to liquidate.")
    
    scenario = st.radio(
        "Lifecycle Scenario",
        options=["Shelf-Stable (Ongoing)", "FMCG (Risk of Obsolescence)", "End of Life (Sunset)"],
        index=1,
        help="Defines the penalty for leftover inventory."
    )

    st.markdown("---")
    st.header("📈 Demand Profile")
    mean_demand = st.number_input("Mean Seasonal/Period Demand", value=1000, step=50)
    volatility_pct = st.slider("Demand Volatility (CV %)", min_value=5, max_value=80, value=25,
                                help="Coefficient of Variation — higher means more uncertain demand.")
    sigma = mean_demand * (volatility_pct / 100.0)
    st.caption(f"Std Dev = **{int(sigma):,} units**")

    st.markdown("---")
    st.header("🏭 Base Supplier (Cheap, Slow)")
    base_cost = st.number_input("Unit Cost — Base (£)", value=20.0, step=0.5)
    base_lead_time = st.number_input("Lead Time — Base (weeks)", value=12, step=1)
    base_moq = st.number_input("MOQ — Base (units)", value=500, step=100)

    st.markdown("---")
    st.header("⚡ Surge Supplier (Expensive, Fast)")
    surge_cost = st.number_input("Unit Cost — Surge (£)", value=22.0, step=0.5)
    surge_lead_time = st.number_input("Lead Time — Surge (weeks)", value=2, step=1)
    surge_moq = st.number_input("MOQ — Surge (units)", value=100, step=50)

    st.markdown("---")
    st.header("💰 Holding Cost")
    holding_cost_pct = st.slider("Annual Holding Cost (% of unit cost)", 10, 40, 20) / 100.0
    holding_per_week = (base_cost * holding_cost_pct) / 52.0
    holding_per_period = holding_per_week * base_lead_time

    run = st.button("🚀 Run Optimizer", type="primary", use_container_width=True)


# ─────────────────────────────────────────────
# NEWSVENDOR ENGINE
# ─────────────────────────────────────────────
def expected_metrics(q, mu, sigma):
    if sigma <= 0:
        return float(min(q, mu)), float(max(0, q - mu)), float(max(0, mu - q))
    z     = (q - mu) / sigma
    phi_z = stats.norm.pdf(z)
    Phi_z = stats.norm.cdf(z)
    loss  = phi_z - z * (1 - Phi_z)
    exp_sales    = float(np.clip(mu - sigma * loss, 0, q))
    exp_leftover = float(max(0.0, q - exp_sales))
    exp_stockout = float(max(0.0, mu - exp_sales))
    return exp_sales, exp_leftover, exp_stockout

def calculate_overage_cost(cost, salvage, holding, scenario):
    if scenario == "End of Life (Sunset)":
        return (cost - salvage) + holding
    elif scenario == "FMCG (Risk of Obsolescence)":
        return ((cost - salvage) * 0.5) + holding
    else: # "Shelf-Stable (Ongoing)"
        return holding

def newsvendor_base(salvage, base_cost, surge_cost, mu, sigma, holding_per_period, scenario):
    cu = surge_cost - base_cost
    co = calculate_overage_cost(base_cost, salvage, holding_per_period, scenario)
    
    if (cu + co) <= 0:
        return None
    cr = cu / (cu + co)
    z  = stats.norm.ppf(cr)
    q  = max(0.0, mu + z * sigma)
    exp_sales, exp_leftover, exp_stockout = expected_metrics(q, mu, sigma)
    return dict(cu=cu, co=co, critical_ratio=cr, z_score=z, optimal_q=q,
                exp_sales=exp_sales, exp_leftover=exp_leftover, exp_stockout=exp_stockout)

def newsvendor_surge(price, salvage, base_cost, surge_cost, mu, sigma, holding_per_period, scenario):
    surge_hold = (surge_cost / base_cost) * holding_per_period
    cu = price - surge_cost
    co = calculate_overage_cost(surge_cost, salvage, surge_hold, scenario)
    
    if (cu + co) <= 0:
        return None
    cr = cu / (cu + co)
    z  = stats.norm.ppf(cr)
    q  = max(0.0, mu + z * sigma)
    exp_sales, exp_leftover, exp_stockout = expected_metrics(q, mu, sigma)
    exp_profit = (price * exp_sales) + (salvage * exp_leftover) - (surge_cost * q) - (holding_per_period * exp_leftover)
    return dict(cu=cu, co=co, critical_ratio=cr, z_score=z, optimal_q=q,
                exp_sales=exp_sales, exp_leftover=exp_leftover, exp_stockout=exp_stockout,
                exp_profit=exp_profit)

def dual_source_sweep(price, salvage, mu, sigma, base_cost, surge_cost,
                       base_moq, surge_moq, holding_per_period, q_base_fixed, scenario):
    results = []
    surge_holding_per_period = (surge_cost / base_cost) * holding_per_period

    for pct in np.arange(0.50, 0.999, 0.005):
        q_target    = mu + stats.norm.ppf(pct) * sigma
        q_surge_raw = max(0.0, q_target - q_base_fixed)
        q_surge     = max(float(surge_moq), q_surge_raw) if q_surge_raw >= surge_moq else 0.0
        q_total     = q_base_fixed + q_surge

        exp_sales, exp_leftover, exp_stockout = expected_metrics(q_total, mu, sigma)

        base_frac      = q_base_fixed / q_total if q_total > 0 else 1.0
        base_leftover  = exp_leftover * base_frac
        surge_leftover = exp_leftover * (1.0 - base_frac)

        revenue = price * exp_sales
        
        if scenario == "End of Life (Sunset)":
            salvage_rev = salvage * exp_leftover
        elif scenario == "FMCG (Risk of Obsolescence)":
            salvage_rev = (salvage * 0.5 * exp_leftover) + (base_cost * 0.5 * exp_leftover)
        else:
            salvage_rev = base_cost * exp_leftover
        
        base_spend  = base_cost * q_base_fixed
        surge_spend = surge_cost * q_surge
        hold_cost   = (holding_per_period * base_leftover) + (surge_holding_per_period * surge_leftover)
        exp_profit  = revenue + salvage_rev - base_spend - surge_spend - hold_cost

        results.append({
            "Service Level (%)":     round(pct * 100, 1),
            "Q Base":                int(q_base_fixed),
            "Q Surge":               int(q_surge),
            "Q Total":               int(q_total),
            "Surge %":               round(q_surge / q_total * 100, 1) if q_total > 0 else 0,
            "Base Spend (£)":        round(base_spend),
            "Surge Spend (£)":       round(surge_spend),
            "Holding Cost (£)":      round(hold_cost),
            "Exp. Leftover (units)": round(exp_leftover),
            "Exp. Stockout (units)": round(exp_stockout),
            "Exp. Profit (£)":       round(exp_profit),
        })

    return pd.DataFrame(results)

# ─────────────────────────────────────────────
# MAIN OUTPUT
# ─────────────────────────────────────────────
if run:
    if surge_cost <= base_cost:
        st.error("Surge unit cost must be higher than base unit cost.")
        st.stop()
    if base_cost <= salvage_value:
        st.error("Base unit cost must be above the salvage value.")
        st.stop()
    if selling_price <= surge_cost:
        st.error("Selling price must be above the surge unit cost.")
        st.stop()

    nv_base  = newsvendor_base(salvage_value, base_cost, surge_cost, mean_demand, sigma, holding_per_period, scenario)
    nv_surge = newsvendor_surge(selling_price, salvage_value, base_cost, surge_cost, mean_demand, sigma, holding_per_period, scenario)

    if not nv_base or not nv_surge:
        st.error("Check inputs — verify costs are valid relative to price and salvage.")
        st.stop()

    q_base_fixed = max(float(base_moq), nv_base["optimal_q"]) if nv_base["optimal_q"] >= base_moq else float(base_moq)

    df_sweep = dual_source_sweep(selling_price, salvage_value, mean_demand, sigma,
                                  base_cost, surge_cost, base_moq, surge_moq,
                                  holding_per_period, q_base_fixed, scenario)

    best_idx = df_sweep["Exp. Profit (£)"].idxmax()
    best     = df_sweep.loc[best_idx]

    # ── KPI HEADER ───────────────────────────────
    st.markdown("---")
    st.subheader(f"🎯 Optimal Dual-Source Split: {scenario}")
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total Service Level",   f"{best['Service Level (%)']:.1f}%")
    k2.metric("Base Order (units)",    f"{int(best['Q Base']):,}")
    k3.metric("Surge Order (units)",   f"{int(best['Q Surge']):,}", delta=f"{best['Surge %']:.0f}% of total", delta_color="off")
    k4.metric("Expected Profit",       f"£{int(best['Exp. Profit (£)']):,}")
    k5.metric("Expected Stockout",     f"{int(best['Exp. Stockout (units)']):,} units")

    # ── TABS ─────────────────────────────────────
    t1, t2, t3, t4 = st.tabs(["📊 Profit vs. Service Level", "🔬 Newsvendor Mechanics", "📐 Demand Distribution", "📋 Full Sweep Table"])

    with t1:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=df_sweep["Service Level (%)"], y=df_sweep["Exp. Profit (£)"], name="Expected Profit (£)", line=dict(color="#2563eb", width=3)), secondary_y=False)
        fig.add_trace(go.Scatter(x=df_sweep["Service Level (%)"], y=df_sweep["Q Surge"], name="Surge Units", line=dict(color="#f59e0b", width=2, dash="dot")), secondary_y=True)
        fig.add_trace(go.Scatter(x=df_sweep["Service Level (%)"], y=df_sweep["Q Base"], name="Base Units", line=dict(color="#10b981", width=2, dash="dot")), secondary_y=True)
        fig.add_vline(x=best["Service Level (%)"], line_dash="dash", line_color="red", annotation_text=f"Optimum: {best['Service Level (%)']:.1f}%", annotation_position="top right")
        fig.update_layout(height=450, template="plotly_white", title="Expected Profit vs. Target Service Level")
        st.plotly_chart(fig, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            if scenario == "End of Life (Sunset)":
                markdown_cost = max(0, int(best["Exp. Leftover (units)"]) * (base_cost - salvage_value))
            elif scenario == "FMCG (Risk of Obsolescence)":
                markdown_cost = max(0, int(best["Exp. Leftover (units)"]) * (base_cost - salvage_value) * 0.5)
            else:
                markdown_cost = 0

            fig_pie = go.Figure(go.Pie(
                labels=["Base Supplier Spend", "Surge Supplier Spend", "Expected Markdowns", "Holding Cost"],
                values=[best["Base Spend (£)"], best["Surge Spend (£)"], markdown_cost, best["Holding Cost (£)"]],
                marker_colors=["#10b981", "#f59e0b", "#ef4444", "#8b5cf6"], hole=0.4
            ))
            fig_pie.update_layout(title="Cost Breakdown", height=320, template="plotly_white")
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with c2:
            st.markdown("**What the surge order is buying you:**")
            surge_premium_total = int(best["Q Surge"]) * (surge_cost - base_cost)
            unhedged_exposure   = int(nv_base["exp_stockout"]) * (surge_cost - base_cost)
            net                 = unhedged_exposure - surge_premium_total
            st.metric("Surge Premium Paid", f"£{surge_premium_total:,}")
            st.metric("Unhedged Surge Exposure (base-only commit)", f"£{unhedged_exposure:,}")
            st.metric("Net Benefit of Pre-Ordering Surge", f"£{net:,}", delta="Pre-ordering surge is worth it" if net > 0 else "Surge order over-sized")

    with t2:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**🏭 Base Supplier (Scenario: {scenario})**")
            st.markdown(f"- **Cu** = surge cost − base cost = **£{nv_base['cu']:.2f}**/unit")
            st.markdown(f"- **Co** = **£{nv_base['co']:.2f}**/unit")
            st.markdown(f"- Critical Ratio: **{nv_base['critical_ratio']:.3f}**")
            st.markdown(f"- Newsvendor Optimal Q: **{int(nv_base['optimal_q']):,} units**")
        with col2:
            st.markdown(f"**⚡ Surge Supplier (Scenario: {scenario})**")
            st.markdown(f"- **Cu** = price − surge cost = **£{nv_surge['cu']:.2f}**/unit")
            st.markdown(f"- **Co** = **£{nv_surge['co']:.2f}**/unit")
            st.markdown(f"- Critical Ratio: **{nv_surge['critical_ratio']:.3f}**")
            st.markdown(f"- Standalone Optimal Q: **{int(nv_surge['optimal_q']):,} units**")

    with t3:
        x = np.linspace(mean_demand - 4 * sigma, mean_demand + 4 * sigma, 600)
        y_pdf = stats.norm.pdf(x, mean_demand, sigma)
        q_b_opt = int(best["Q Base"])
        q_t_opt = int(best["Q Total"])

        fig2 = go.Figure()
        mask_base = x <= q_b_opt
        fig2.add_trace(go.Scatter(x=np.concatenate([x[mask_base], [q_b_opt, x[mask_base][0]]]), y=np.concatenate([y_pdf[mask_base], [0, 0]]), fill='toself', fillcolor='rgba(16,185,129,0.2)', line=dict(color='rgba(0,0,0,0)'), name='Base Coverage'))
        mask_surge = (x > q_b_opt) & (x <= q_t_opt)
        if mask_surge.any():
            fig2.add_trace(go.Scatter(x=np.concatenate([[q_b_opt], x[mask_surge], [q_t_opt, q_b_opt]]), y=np.concatenate([[0], y_pdf[mask_surge], [0, 0]]), fill='toself', fillcolor='rgba(245,158,11,0.25)', line=dict(color='rgba(0,0,0,0)'), name='Surge Coverage'))
        mask_stock = x > q_t_opt
        if mask_stock.any():
            fig2.add_trace(go.Scatter(x=np.concatenate([[q_t_opt], x[mask_stock], [x[mask_stock][-1], q_t_opt]]), y=np.concatenate([[0], y_pdf[mask_stock], [0, 0]]), fill='toself', fillcolor='rgba(239,68,68,0.25)', line=dict(color='rgba(0,0,0,0)'), name='Residual Stockout'))
        
        fig2.add_trace(go.Scatter(x=x, y=y_pdf, mode='lines', line=dict(color='#1e3a5f', width=2.5), name='Demand Dist.'))
        fig2.update_layout(title="Demand Distribution with Base & Surge Coverage", xaxis_title="Demand", yaxis_title="Probability Density", height=420, template="plotly_white")
        st.plotly_chart(fig2, use_container_width=True)

    with t4:
        st.dataframe(df_sweep.style.format({"Service Level (%)": "{:.1f}%", "Exp. Profit (£)": "£{:,.0f}"}), use_container_width=True, hide_index=True)
else:
    st.info("👈 Configure your supplier parameters in the sidebar and click **Run Optimizer**.")
