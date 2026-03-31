import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

st.set_page_config(page_title="Dual Sourcing: Base-Surge Optimizer", layout="wide")

st.title("⚖️ Dual Sourcing: Base-Surge Newsvendor Optimizer")
st.markdown(
    "**Core concept:** A base supplier (cheap, slow) covers your *expected* demand. "
    "A surge supplier (expensive, fast) covers the *upside tail* — but only when the cost of "
    "the surge premium is less than the cost of a stockout. This tool finds that crossover point."
)

# ─────────────────────────────────────────────
# SIDEBAR: INPUTS
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("📦 Product Economics")
    selling_price = st.number_input("Selling Price (£)", value=60.0, step=1.0)
    salvage_value = st.number_input("Salvage / Markdown Value (£)", value=10.0, step=1.0,
                                    help="What you recover if you over-order and discount unsold units.")

    st.markdown("---")
    st.header("📈 Demand Profile")
    mean_demand = st.number_input("Mean Demand (units / period)", value=1000, step=50)
    volatility_pct = st.slider("Demand Volatility (CV %)", min_value=5, max_value=80, value=25,
                                help="Coefficient of Variation. Higher = more uncertain demand. This is the core driver of how much surge capacity you need.")
    st.caption(f"Std Dev = **{int(mean_demand * volatility_pct / 100):,} units**")

    st.markdown("---")
    st.header("🏭 Base Supplier (Cheap, Slow)")
    base_cost = st.number_input("Unit Cost — Base (£)", value=18.0, step=0.5)
    base_lead_time = st.number_input("Lead Time — Base (weeks)", value=10, step=1)
    base_moq = st.number_input("MOQ — Base (units)", value=500, step=100,
                                help="Minimum Order Quantity. Base suppliers often have high MOQs.")

    st.markdown("---")
    st.header("⚡ Surge Supplier (Expensive, Fast)")
    surge_cost = st.number_input("Unit Cost — Surge (£)", value=28.0, step=0.5)
    surge_lead_time = st.number_input("Lead Time — Surge (weeks)", value=2, step=1)
    surge_moq = st.number_input("MOQ — Surge (units)", value=100, step=50,
                                 help="Surge suppliers are typically more flexible on minimums.")

    st.markdown("---")
    st.header("💰 Financial Parameters")
    holding_cost_pct = st.slider("Annual Holding Cost (% of unit cost)", 10, 40, 20) / 100.0
    periods_per_year = st.number_input("Periods per Year", value=12, step=1,
                                       help="e.g. 12 = monthly ordering cycles, 52 = weekly.")

    run = st.button("🚀 Run Optimizer", type="primary", use_container_width=True)

# ─────────────────────────────────────────────
# CORE NEWSVENDOR ENGINE
# ─────────────────────────────────────────────
def newsvendor_analysis(price, salvage, cost, mu, sigma):
    """Returns critical ratio, optimal Q, and expected metrics."""
    underage_cost  = price - cost       # cost of stocking out (lost margin)
    overage_cost   = cost - salvage     # cost of over-ordering (markdown loss)

    if (underage_cost + overage_cost) <= 0:
        return None

    critical_ratio = underage_cost / (underage_cost + overage_cost)
    z = stats.norm.ppf(critical_ratio)
    optimal_q = max(0, mu + z * sigma)

    # Expected sales, leftover, stockout
    exp_sales    = mu - sigma * stats.norm.pdf(z) + optimal_q * (1 - stats.norm.cdf(z))
    exp_sales    = min(exp_sales, optimal_q)
    exp_leftover = max(0, optimal_q - exp_sales)
    exp_stockout = max(0, mu - exp_sales)

    exp_profit = (price * exp_sales) + (salvage * exp_leftover) - (cost * optimal_q)

    return {
        "critical_ratio": critical_ratio,
        "z_score": z,
        "optimal_q": optimal_q,
        "underage_cost": underage_cost,
        "overage_cost": overage_cost,
        "exp_sales": exp_sales,
        "exp_leftover": exp_leftover,
        "exp_stockout": exp_stockout,
        "exp_profit": exp_profit,
    }


def dual_source_split(price, salvage, mu, sigma, base_cost, surge_cost, base_moq, surge_moq):
    """
    Dual sourcing logic:
    - Base order covers the newsvendor Q computed at base supplier's cost.
    - For each additional unit above that, we check: is the surge premium cheaper than the stockout cost?
    - The surge quantity covers the demand between the base Q and the surge-adjusted optimal Q.
    Returns a sweep of results across percentile thresholds so we can plot the tradeoff.
    """
    results = []
    percentiles = np.arange(0.50, 0.999, 0.005)

    for pct in percentiles:
        q_total = mu + stats.norm.ppf(pct) * sigma

        # Base covers what newsvendor says at base cost; can't go below MOQ if ordering
        base_nv     = newsvendor_analysis(price, salvage, base_cost, mu, sigma)
        q_base_raw  = base_nv["optimal_q"] if base_nv else mu
        q_base      = max(base_moq, q_base_raw) if q_base_raw >= base_moq else 0

        # Surge fills the gap between base commitment and total target Q
        q_surge_raw = max(0, q_total - q_base)
        q_surge     = max(surge_moq, q_surge_raw) if q_surge_raw >= surge_moq else 0

        q_total_actual = q_base + q_surge

        # Expected financials
        sigma_eff = sigma
        exp_sales    = mu - sigma_eff * stats.norm.pdf(stats.norm.ppf(pct)) + q_total_actual * (1 - pct)
        exp_sales    = min(exp_sales, q_total_actual)
        exp_leftover = max(0, q_total_actual - exp_sales)
        exp_stockout = max(0, mu - exp_sales)

        revenue        = price * exp_sales
        salvage_rev    = salvage * exp_leftover
        base_spend     = base_cost * q_base
        surge_spend    = surge_cost * q_surge
        stockout_cost  = (price - base_cost) * exp_stockout  # opportunity cost
        exp_profit     = revenue + salvage_rev - base_spend - surge_spend

        results.append({
            "Service Level (%)": round(pct * 100, 1),
            "Q Base": int(q_base),
            "Q Surge": int(q_surge),
            "Q Total": int(q_total_actual),
            "Surge %": round(q_surge / q_total_actual * 100, 1) if q_total_actual > 0 else 0,
            "Base Spend (£)": round(base_spend),
            "Surge Spend (£)": round(surge_spend),
            "Surge Premium (£)": round(surge_spend - (surge_cost - base_cost) * 0),  # incremental vs all-base
            "Exp. Stockout (units)": round(exp_stockout),
            "Exp. Leftover (units)": round(exp_leftover),
            "Exp. Profit (£)": round(exp_profit),
        })

    return pd.DataFrame(results)


# ─────────────────────────────────────────────
# MAIN OUTPUT
# ─────────────────────────────────────────────
if run:
    sigma = mean_demand * (volatility_pct / 100.0)
    holding_per_unit_per_period = (base_cost * holding_cost_pct) / periods_per_year

    # Run individual newsvendor for each supplier
    nv_base  = newsvendor_analysis(selling_price, salvage_value, base_cost,  mean_demand, sigma)
    nv_surge = newsvendor_analysis(selling_price, salvage_value, surge_cost, mean_demand, sigma)

    if not nv_base or not nv_surge:
        st.error("Check inputs — cost must be between salvage value and selling price.")
        st.stop()

    df_sweep = dual_source_split(selling_price, salvage_value, mean_demand, sigma,
                                  base_cost, surge_cost, base_moq, surge_moq)

    # Find the profit-maximizing row
    best_idx   = df_sweep["Exp. Profit (£)"].idxmax()
    best_row   = df_sweep.loc[best_idx]

    # ── KPI HEADER ──────────────────────────────
    st.markdown("---")
    st.subheader("🎯 Optimal Dual-Source Split")

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Optimal Service Level",  f"{best_row['Service Level (%)']:.0f}%")
    k2.metric("Base Order (units)",     f"{int(best_row['Q Base']):,}")
    k3.metric("Surge Order (units)",    f"{int(best_row['Q Surge']):,}",
              delta=f"{best_row['Surge %']:.0f}% of total",
              delta_color="off")
    k4.metric("Expected Profit",        f"£{int(best_row['Exp. Profit (£)']):,}")
    k5.metric("Exp. Stockout",          f"{int(best_row['Exp. Stockout (units)']):,} units")

    # ── TABS ────────────────────────────────────
    t1, t2, t3, t4 = st.tabs([
        "📊 Profit vs. Service Level",
        "🔬 Newsvendor Mechanics",
        "📐 Demand Distribution",
        "📋 Full Sweep Table"
    ])

    # ── TAB 1: PROFIT CURVE ─────────────────────
    with t1:
        st.markdown(
            "The chart below shows how expected profit changes as you increase the target service level "
            "by ordering more from the surge supplier. The **peak is the optimal split** — past that point, "
            "the surge premium costs more than the additional stockout risk you're hedging."
        )

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(go.Scatter(
            x=df_sweep["Service Level (%)"], y=df_sweep["Exp. Profit (£)"],
            name="Expected Profit (£)", line=dict(color="#2563eb", width=3)
        ), secondary_y=False)

        fig.add_trace(go.Scatter(
            x=df_sweep["Service Level (%)"], y=df_sweep["Q Surge"],
            name="Surge Units Ordered", line=dict(color="#f59e0b", width=2, dash="dot")
        ), secondary_y=True)

        fig.add_trace(go.Scatter(
            x=df_sweep["Service Level (%)"], y=df_sweep["Q Base"],
            name="Base Units Ordered", line=dict(color="#10b981", width=2, dash="dot")
        ), secondary_y=True)

        # Mark optimum
        fig.add_vline(x=best_row["Service Level (%)"], line_dash="dash", line_color="red",
                      annotation_text=f"Optimum: {best_row['Service Level (%)']:.0f}%",
                      annotation_position="top right")

        fig.update_layout(
            title="Expected Profit vs. Target Service Level",
            xaxis_title="Target Service Level (%)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=450, template="plotly_white"
        )
        fig.update_yaxes(title_text="Expected Profit (£)", secondary_y=False)
        fig.update_yaxes(title_text="Units Ordered", secondary_y=True)

        st.plotly_chart(fig, use_container_width=True)

        # Cost decomposition at optimum
        st.markdown("#### Cost Decomposition at Optimal Split")
        c1, c2 = st.columns(2)
        with c1:
            pie_labels = ["Base Supplier Spend", "Surge Supplier Spend", "Expected Markdowns"]
            pie_values = [
                best_row["Base Spend (£)"],
                best_row["Surge Spend (£)"],
                int(best_row["Exp. Leftover (units)"]) * (base_cost - salvage_value),
            ]
            fig_pie = go.Figure(go.Pie(
                labels=pie_labels, values=pie_values,
                marker_colors=["#10b981", "#f59e0b", "#ef4444"],
                hole=0.4
            ))
            fig_pie.update_layout(title="Total Cost Breakdown", height=320, template="plotly_white")
            st.plotly_chart(fig_pie, use_container_width=True)

        with c2:
            st.markdown("**What the surge premium is buying you:**")
            surge_units = int(best_row["Q Surge"])
            surge_premium_per_unit = surge_cost - base_cost
            total_surge_premium = surge_units * surge_premium_per_unit
            avoided_stockout_val = int(best_row["Exp. Stockout (units)"]) * (selling_price - base_cost)

            st.metric("Surge Premium Paid", f"£{total_surge_premium:,}",
                      help="Extra spend vs. ordering everything from base.")
            st.metric("Stockout Cost at Base-Only", 
                      f"£{int(nv_base['exp_stockout'] * (selling_price - base_cost)):,}",
                      help="Lost margin if you only used the base supplier at its optimal Q.")
            net_benefit = int(nv_base['exp_stockout'] * (selling_price - base_cost)) - total_surge_premium
            st.metric("Net Benefit of Surge", f"£{net_benefit:,}",
                      delta="Surge pays off" if net_benefit > 0 else "Surge too expensive",
                      delta_color="normal" if net_benefit > 0 else "inverse")

    # ── TAB 2: NEWSVENDOR MECHANICS ─────────────
    with t2:
        st.markdown(
            "The newsvendor model finds the order quantity where the **marginal cost of over-ordering "
            "equals the marginal cost of under-ordering**. The critical ratio is the service level that "
            "maximises expected profit — not revenue, not fill rate."
        )

        col1, col2 = st.columns(2)

        def nv_card(label, nv, cost, color):
            with st.container(border=True):
                st.markdown(f"**{label}**")
                st.markdown(f"- Unit Cost: **£{cost:.2f}**")
                st.markdown(f"- Underage Cost (Cu): **£{nv['underage_cost']:.2f}** *(lost margin per stockout unit)*")
                st.markdown(f"- Overage Cost (Co): **£{nv['overage_cost']:.2f}** *(markdown loss per leftover unit)*")
                st.markdown(f"- Critical Ratio: **{nv['critical_ratio']:.3f}** → order up to **{nv['critical_ratio']*100:.1f}th percentile**")
                st.markdown(f"- Z-score: **{nv['z_score']:.3f}**")
                st.markdown(f"- Standalone Optimal Q: **{int(nv['optimal_q']):,} units**")
                st.markdown(f"- Expected Stockout: **{int(nv['exp_stockout']):,} units**")
                st.markdown(f"- Expected Leftover: **{int(nv['exp_leftover']):,} units**")
                st.markdown(f"- Expected Profit: **£{int(nv['exp_profit']):,}**")

        with col1:
            nv_card("🏭 Base Supplier (Standalone)", nv_base, base_cost, "#10b981")
        with col2:
            nv_card("⚡ Surge Supplier (Standalone)", nv_surge, surge_cost, "#f59e0b")

        st.markdown("---")
        st.markdown(
            "**The key insight:** The base supplier's lower cost raises its critical ratio "
            f"({nv_base['critical_ratio']:.3f}) vs. the surge supplier "
            f"({nv_surge['critical_ratio']:.3f}). "
            "In isolation, you'd order more from base because each unit is cheaper to over-stock. "
            "But in a dual-source model, you *commit* to the base order early and use surge to "
            "flexibly cover the uncertain tail — so you're not paying surge cost on every unit, "
            "only on the incremental upside."
        )

    # ── TAB 3: DEMAND DISTRIBUTION ──────────────
    with t3:
        st.markdown(
            "This shows the demand distribution and where the base and surge commitments sit on it. "
            "Units to the right of the total order quantity are unmet demand (stockout). "
            "Units to the left of the base order are the over-order risk covered by markdown."
        )

        x = np.linspace(mean_demand - 4*sigma, mean_demand + 4*sigma, 500)
        y_pdf = stats.norm.pdf(x, mean_demand, sigma)

        q_base_opt  = int(best_row["Q Base"])
        q_total_opt = int(best_row["Q Total"])

        fig2 = go.Figure()

        # Shade overage zone (leftover risk)
        mask_over = x <= q_base_opt
        fig2.add_trace(go.Scatter(
            x=np.concatenate([x[mask_over], [q_base_opt, x[mask_over][0]]]),
            y=np.concatenate([y_pdf[mask_over], [0, 0]]),
            fill='toself', fillcolor='rgba(16,185,129,0.2)',
            line=dict(color='rgba(0,0,0,0)'), name='Base Coverage (over-order risk)'
        ))

        # Shade surge zone
        mask_surge = (x > q_base_opt) & (x <= q_total_opt)
        if mask_surge.any():
            fig2.add_trace(go.Scatter(
                x=np.concatenate([[q_base_opt], x[mask_surge], [q_total_opt, q_base_opt]]),
                y=np.concatenate([[0], y_pdf[mask_surge], [0, 0]]),
                fill='toself', fillcolor='rgba(245,158,11,0.25)',
                line=dict(color='rgba(0,0,0,0)'), name='Surge Coverage'
            ))

        # Shade stockout zone
        mask_stock = x > q_total_opt
        fig2.add_trace(go.Scatter(
            x=np.concatenate([[q_total_opt], x[mask_stock], [x[mask_stock][-1], q_total_opt]]),
            y=np.concatenate([[0], y_pdf[mask_stock], [0, 0]]),
            fill='toself', fillcolor='rgba(239,68,68,0.25)',
            line=dict(color='rgba(0,0,0,0)'), name='Residual Stockout Risk'
        ))

        # PDF line
        fig2.add_trace(go.Scatter(
            x=x, y=y_pdf, mode='lines',
            line=dict(color='#1e3a5f', width=2.5), name='Demand Distribution'
        ))

        fig2.add_vline(x=mean_demand, line_dash="dot", line_color="grey",
                       annotation_text="Mean Demand", annotation_position="top left")
        fig2.add_vline(x=q_base_opt, line_dash="dash", line_color="#10b981",
                       annotation_text=f"Base Commit: {q_base_opt:,}", annotation_position="top right")
        fig2.add_vline(x=q_total_opt, line_dash="dash", line_color="#f59e0b",
                       annotation_text=f"Total Q: {q_total_opt:,}", annotation_position="top right")

        fig2.update_layout(
            title="Demand Distribution with Base & Surge Coverage",
            xaxis_title="Demand (units)", yaxis_title="Probability Density",
            height=420, template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown(
            f"🟢 **Green zone** — demand covered by base order alone. Any demand here means you may have leftover stock at markdown value (£{salvage_value:.0f}/unit).  \n"
            f"🟡 **Amber zone** — demand covered only because you placed the surge order. This is what the surge premium buys.  \n"
            f"🔴 **Red zone** — demand scenarios above your total order. These result in a stockout at lost margin of £{selling_price - base_cost:.0f}/unit."
        )

    # ── TAB 4: FULL SWEEP TABLE ──────────────────
    with t4:
        st.markdown("Full sensitivity sweep across all target service levels. Highlighted row = profit-maximising optimum.")

        def highlight_best(row):
            if abs(row["Service Level (%)"] - best_row["Service Level (%)"]) < 0.1:
                return ['background-color: #dbeafe'] * len(row)
            return [''] * len(row)

        st.dataframe(
            df_sweep.style
                .apply(highlight_best, axis=1)
                .format({
                    "Service Level (%)": "{:.1f}%",
                    "Surge %": "{:.1f}%",
                    "Base Spend (£)": "£{:,.0f}",
                    "Surge Spend (£)": "£{:,.0f}",
                    "Exp. Profit (£)": "£{:,.0f}",
                }),
            use_container_width=True, hide_index=True
        )

        csv = df_sweep.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Sweep (.CSV)", data=csv,
                           file_name="dual_source_sweep.csv", mime="text/csv")

else:
    st.info("👈 Configure your supplier parameters in the sidebar and click **Run Optimizer**.")
    st.markdown("""
    ### How this works
    
    **Step 1 — Newsvendor baseline:** For each supplier in isolation, the model finds the profit-maximising 
    order quantity using the critical ratio: `Cu / (Cu + Co)` where Cu = cost of under-ordering (lost margin) 
    and Co = cost of over-ordering (markdown loss).

    **Step 2 — Base commitment:** You lock in the base order early (long lead time). This covers expected 
    demand at low cost, but you're exposed to demand volatility.

    **Step 3 — Surge fill:** As the selling season approaches, demand uncertainty resolves. The surge supplier 
    covers the gap between your base commitment and your target service level — but at a premium. The model 
    sweeps every possible service level target and finds where the surge premium is no longer worth the 
    additional stockout risk it hedges.

    **Step 4 — Output:** The optimal split, a profit curve, and the full demand distribution showing exactly 
    what each supplier is covering.
    """)
