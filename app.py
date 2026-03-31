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
    salvage_value = st.number_input("Salvage / Markdown Value (£)", value=10.0, step=1.0,
                                    help="What you recover per unit if you over-order and discount unsold stock.")
    alpha = st.slider("Inventory Carry-Forward (%)", 0, 100, 70,
                      help="Percent of leftover inventory expected to sell later at full price (Shelf Life Factor). Higher % = Lower overage cost.") / 100.0

    st.markdown("---")
    st.header("📈 Demand Profile")
    mean_demand = st.number_input("Mean Weekly Demand (units)", value=1000, step=50)
    volatility_pct = st.slider("Demand Volatility (CV %)", min_value=5, max_value=80, value=25,
                                help="Coefficient of Variation — higher means more uncertain demand. "
                                     "This is the primary driver of how much surge capacity you need.")
    sigma = mean_demand * (volatility_pct / 100.0)
    st.caption(f"Std Dev = **{int(sigma):,} units/week**")

    st.markdown("---")
    st.header("🏭 Base Supplier (Cheap, Slow)")
    base_cost = st.number_input("Unit Cost — Base (£)", value=18.0, step=0.5)
    base_lead_time = st.number_input("Lead Time — Base (weeks)", value=10, step=1)
    base_moq = st.number_input("MOQ — Base (units)", value=500, step=100,
                                help="Minimum Order Quantity. Base suppliers often carry high MOQs.")

    st.markdown("---")
    st.header("⚡ Surge Supplier (Expensive, Fast)")
    surge_cost = st.number_input("Unit Cost — Surge (£)", value=28.0, step=0.5)
    surge_lead_time = st.number_input("Lead Time — Surge (weeks)", value=2, step=1)
    surge_moq = st.number_input("MOQ — Surge (units)", value=100, step=50,
                                 help="Surge suppliers are typically more flexible on minimums.")

    st.markdown("---")
    st.header("💰 Holding Cost")
    holding_cost_pct = st.slider("Annual Holding Cost (% of unit cost)", 10, 40, 20,
                                  help="Includes warehousing, insurance, obsolescence, and cost of capital.") / 100.0
    # All time units are weeks throughout
    holding_per_week = (base_cost * holding_cost_pct) / 52.0
    holding_per_period = holding_per_week * base_lead_time
    st.caption(
        f"£{holding_per_week:.3f}/unit/week × {base_lead_time} week lead time "
        f"= **£{holding_per_period:.2f}/unit** applied to expected leftover inventory"
    )

    run = st.button("🚀 Run Optimizer", type="primary", use_container_width=True)


# ─────────────────────────────────────────────
# NEWSVENDOR ENGINE
# ─────────────────────────────────────────────
def expected_metrics(q, mu, sigma):
    """
    Exact expected sales, leftover, and stockout using the normal loss function.
    L(z) = phi(z) - z*(1 - Phi(z))
    E[sales] = mu - sigma * L(z)
    """
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


def newsvendor_base(salvage, base_cost, surge_cost, mu, sigma, holding_per_period, price, alpha):
    """
    Base supplier in a dual-source model.
    Cu = surge_cost - base_cost
         Under-ordering from base just triggers the surge premium — the sale is not lost.
    Co = (base_cost - effective_salvage) + holding_per_period
         Overage considers carry-forward (shelf life).
    """
    effective_salvage = salvage + alpha * (price - salvage)
    cu = surge_cost - base_cost
    co = (base_cost - effective_salvage) + holding_per_period
    co = max(co, 1e-5) # Prevent division by zero if Co goes non-positive due to high alpha
    
    if (cu + co) <= 0:
        return None
    cr = cu / (cu + co)
    z  = stats.norm.ppf(cr)
    q  = max(0.0, mu + z * sigma)
    exp_sales, exp_leftover, exp_stockout = expected_metrics(q, mu, sigma)
    return dict(cu=cu, co=co, critical_ratio=cr, z_score=z, optimal_q=q,
                exp_sales=exp_sales, exp_leftover=exp_leftover, exp_stockout=exp_stockout)


def newsvendor_surge(price, salvage, surge_cost, mu, sigma, holding_per_period, alpha):
    """
    Surge supplier — last resort. A stockout here is a genuine lost sale.
    Cu = price - surge_cost   (full lost margin)
    Co = (surge_cost - effective_salvage) + holding_per_period
    """
    effective_salvage = salvage + alpha * (price - salvage)
    cu = price - surge_cost
    co = (surge_cost - effective_salvage) + holding_per_period
    co = max(co, 1e-5) # Prevent division by zero
    
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
                       base_moq, surge_moq, holding_per_period, q_base_fixed):
    """
    Sweep target service levels 50–99.9%. Base Q is fixed. Surge fills the gap.
    Expected sales computed via exact normal loss function at each total Q.
    Holding cost applied proportionally to expected leftover from each supplier.
    """
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

        revenue     = price * exp_sales
        salvage_rev = salvage * exp_leftover
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

    nv_base  = newsvendor_base(salvage_value, base_cost, surge_cost, mean_demand, sigma, holding_per_period, selling_price, alpha)
    nv_surge = newsvendor_surge(selling_price, salvage_value, surge_cost, mean_demand, sigma, holding_per_period, alpha)

    if not nv_base or not nv_surge:
        st.error("Check inputs — verify costs are valid relative to price and salvage.")
        st.stop()

    q_base_fixed = max(float(base_moq), nv_base["optimal_q"]) if nv_base["optimal_q"] >= base_moq else float(base_moq)

    df_sweep = dual_source_sweep(selling_price, salvage_value, mean_demand, sigma,
                                  base_cost, surge_cost, base_moq, surge_moq,
                                  holding_per_period, q_base_fixed)

    best_idx = df_sweep["Exp. Profit (£)"].idxmax()
    best     = df_sweep.loc[best_idx]

    # ── KPI HEADER ───────────────────────────────
    st.markdown("---")
    st.subheader("🎯 Optimal Dual-Source Split")
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total Service Level",   f"{best['Service Level (%)']:.0f}%",
              help="Combined coverage from base + surge against the demand distribution.")
    k2.metric("Base Order (units)",    f"{int(best['Q Base']):,}")
    k3.metric("Surge Order (units)",   f"{int(best['Q Surge']):,}",
              delta=f"{best['Surge %']:.0f}% of total", delta_color="off")
    k4.metric("Expected Profit",       f"£{int(best['Exp. Profit (£)']):,}")
    k5.metric("Expected Stockout",     f"{int(best['Exp. Stockout (units)']):,} units")

    # ── TABS ─────────────────────────────────────
    t1, t2, t3, t4 = st.tabs([
        "📊 Profit vs. Service Level",
        "🔬 Newsvendor Mechanics",
        "📐 Demand Distribution",
        "📋 Full Sweep Table"
    ])

    # ── TAB 1 ────────────────────────────────────
    with t1:
        st.markdown(
            "As the target service level increases, more units are ordered from the surge supplier. "
            "Profit rises until the surge premium exceeds the value of additional stockout risk hedged — "
            "the **peak of the curve is the optimal split**."
        )

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(
            x=df_sweep["Service Level (%)"], y=df_sweep["Exp. Profit (£)"],
            name="Expected Profit (£)", line=dict(color="#2563eb", width=3)
        ), secondary_y=False)
        fig.add_trace(go.Scatter(
            x=df_sweep["Service Level (%)"], y=df_sweep["Q Surge"],
            name="Surge Units", line=dict(color="#f59e0b", width=2, dash="dot")
        ), secondary_y=True)
        fig.add_trace(go.Scatter(
            x=df_sweep["Service Level (%)"], y=df_sweep["Q Base"],
            name="Base Units (fixed)", line=dict(color="#10b981", width=2, dash="dot")
        ), secondary_y=True)
        fig.add_vline(x=best["Service Level (%)"], line_dash="dash", line_color="red",
                      annotation_text=f"Optimum: {best['Service Level (%)']:.0f}%",
                      annotation_position="top right")
        fig.update_layout(
            title="Expected Profit vs. Total Target Service Level (base + surge combined)",
            xaxis_title="Target Service Level (%)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=450, template="plotly_white"
        )
        fig.update_yaxes(title_text="Expected Profit (£)", secondary_y=False)
        fig.update_yaxes(title_text="Units Ordered", secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Cost Decomposition at Optimal Split")
        c1, c2 = st.columns(2)
        with c1:
            markdown_cost = max(0, int(best["Exp. Leftover (units)"]) * (base_cost - salvage_value))
            fig_pie = go.Figure(go.Pie(
                labels=["Base Supplier Spend", "Surge Supplier Spend", "Expected Markdowns", "Holding Cost"],
                values=[best["Base Spend (£)"], best["Surge Spend (£)"], markdown_cost, best["Holding Cost (£)"]],
                marker_colors=["#10b981", "#f59e0b", "#ef4444", "#8b5cf6"],
                hole=0.4
            ))
            fig_pie.update_layout(title="Cost Breakdown at Optimal Split", height=320, template="plotly_white")
            st.plotly_chart(fig_pie, use_container_width=True)

        with c2:
            st.markdown("**What the surge order is buying you:**")
            surge_premium_total = int(best["Q Surge"]) * (surge_cost - base_cost)
            unhedged_exposure   = int(nv_base["exp_stockout"]) * (surge_cost - base_cost)
            net                 = unhedged_exposure - surge_premium_total
            st.metric("Surge Premium Paid",
                      f"£{surge_premium_total:,}",
                      help="Incremental cost vs. ordering all units from base.")
            st.metric("Unhedged Surge Exposure (base-only commit)",
                      f"£{unhedged_exposure:,}",
                      help="What you'd pay in last-minute surge premiums if you only placed the base order and had to escalate stockouts.")
            st.metric("Net Benefit of Pre-Ordering Surge", f"£{net:,}",
                      delta="Pre-ordering surge is worth it" if net > 0 else "Surge order over-sized",
                      delta_color="normal" if net > 0 else "inverse")

    # ── TAB 2 ────────────────────────────────────
    with t2:
        st.markdown(
            "The newsvendor model finds the order quantity where the marginal cost of "
            "**over-ordering (Co)** equals the marginal cost of **under-ordering (Cu)**. "
            "The critical ratio `Cu / (Cu + Co)` gives the optimal service level for each supplier's role."
        )

        col1, col2 = st.columns(2)
        with col1:
            with st.container(border=True):
                st.markdown("**🏭 Base Supplier**")
                st.markdown(
                    f"- Unit Cost: **£{base_cost:.2f}**\n"
                    f"- **Cu** = surge cost − base cost = **£{nv_base['cu']:.2f}**/unit  \n"
                    f"  *(under-ordering from base only triggers the surge premium — the sale is not lost)*\n"
                    f"- **Co** = base cost − effective salvage + holding = **£{nv_base['co']:.2f}**/unit  \n"
                    f"  *(Includes £{holding_per_period:.2f} holding over {base_lead_time} weeks and shelf-life carry-forward)*\n"
                    f"- Critical Ratio: **{nv_base['critical_ratio']:.3f}** → commit to **{nv_base['critical_ratio']*100:.1f}th percentile**\n"
                    f"- Z-score: **{nv_base['z_score']:.3f}**\n"
                    f"- Newsvendor Optimal Q: **{int(nv_base['optimal_q']):,} units**\n"
                    f"- After MOQ floor: **{int(q_base_fixed):,} units**"
                )
        with col2:
            with st.container(border=True):
                st.markdown("**⚡ Surge Supplier**")
                surge_hold = (surge_cost / base_cost) * holding_per_period
                st.markdown(
                    f"- Unit Cost: **£{surge_cost:.2f}**\n"
                    f"- **Cu** = price − surge cost = **£{nv_surge['cu']:.2f}**/unit  \n"
                    f"  *(surge is last resort — a stockout here is a genuine lost sale)*\n"
                    f"- **Co** = surge cost − effective salvage + holding = **£{nv_surge['co']:.2f}**/unit  \n"
                    f"  *(Includes holding cost and shelf-life carry-forward)*\n"
                    f"- Critical Ratio: **{nv_surge['critical_ratio']:.3f}** → target **{nv_surge['critical_ratio']*100:.1f}th percentile**\n"
                    f"- Z-score: **{nv_surge['z_score']:.3f}**\n"
                    f"- Standalone Optimal Q: **{int(nv_surge['optimal_q']):,} units**\n"
                    f"- Expected Stockout (standalone): **{int(nv_surge['exp_stockout']):,} units**\n"
                    f"- Expected Leftover (standalone): **{int(nv_surge['exp_leftover']):,} units**"
                )

        st.markdown("---")
        st.markdown(
            f"Because under-ordering from base only costs the surge premium (£{nv_base['cu']:.2f}/unit) "
            f"rather than the full lost margin (£{nv_surge['cu']:.2f}/unit), the base critical ratio "
            f"({nv_base['critical_ratio']:.3f}) is intentionally **lower** than the surge critical ratio "
            f"({nv_surge['critical_ratio']:.3f}). This keeps the early base commitment conservative — "
            f"the surge supplier handles demand above that threshold."
        )

    # ── TAB 3 ────────────────────────────────────
    with t3:
        st.markdown(
            "The bell curve shows the probability distribution of demand. "
            "Green is covered by base alone, amber by surge, red is residual stockout risk."
        )

        x       = np.linspace(mean_demand - 4 * sigma, mean_demand + 4 * sigma, 600)
        y_pdf   = stats.norm.pdf(x, mean_demand, sigma)
        q_b_opt = int(best["Q Base"])
        q_t_opt = int(best["Q Total"])

        fig2 = go.Figure()

        mask_base = x <= q_b_opt
        fig2.add_trace(go.Scatter(
            x=np.concatenate([x[mask_base], [q_b_opt, x[mask_base][0]]]),
            y=np.concatenate([y_pdf[mask_base], [0, 0]]),
            fill='toself', fillcolor='rgba(16,185,129,0.2)',
            line=dict(color='rgba(0,0,0,0)'), name='Base Coverage'
        ))

        mask_surge = (x > q_b_opt) & (x <= q_t_opt)
        if mask_surge.any():
            fig2.add_trace(go.Scatter(
                x=np.concatenate([[q_b_opt], x[mask_surge], [q_t_opt, q_b_opt]]),
                y=np.concatenate([[0], y_pdf[mask_surge], [0, 0]]),
                fill='toself', fillcolor='rgba(245,158,11,0.25)',
                line=dict(color='rgba(0,0,0,0)'), name='Surge Coverage'
            ))

        mask_stock = x > q_t_opt
        if mask_stock.any():
            fig2.add_trace(go.Scatter(
                x=np.concatenate([[q_t_opt], x[mask_stock], [x[mask_stock][-1], q_t_opt]]),
                y=np.concatenate([[0], y_pdf[mask_stock], [0, 0]]),
                fill='toself', fillcolor='rgba(239,68,68,0.25)',
                line=dict(color='rgba(0,0,0,0)'), name='Residual Stockout Risk'
            ))

        fig2.add_trace(go.Scatter(
            x=x, y=y_pdf, mode='lines',
            line=dict(color='#1e3a5f', width=2.5), name='Demand Distribution'
        ))
        fig2.add_vline(x=mean_demand, line_dash="dot", line_color="grey",
                       annotation_text="Mean Demand", annotation_position="top left")
        fig2.add_vline(x=q_b_opt, line_dash="dash", line_color="#10b981",
                       annotation_text=f"Base Commit: {q_b_opt:,}", annotation_position="top right")
        fig2.add_vline(x=q_t_opt, line_dash="dash", line_color="#f59e0b",
                       annotation_text=f"Total Q: {q_t_opt:,}", annotation_position="top right")
        fig2.update_layout(
            title="Demand Distribution with Base & Surge Coverage",
            xaxis_title="Demand (units)", yaxis_title="Probability Density",
            height=420, template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown(
            f"🟢 **Green** — covered by base order. Over-order risk: leftover sold at £{salvage_value:.0f}/unit "
            f"after £{holding_per_period:.2f}/unit holding cost over the {base_lead_time}-week base lead time.  \n"
            f"🟡 **Amber** — covered by surge order. Each unit costs £{surge_cost - base_cost:.0f} more than base.  \n"
            f"🔴 **Red** — above total order quantity. Genuine lost sale at £{selling_price - surge_cost:.0f}/unit margin."
        )

    # ── TAB 4 ────────────────────────────────────
    with t4:
        st.markdown("Full sweep across all target service levels. Highlighted row = profit-maximising optimum.")

        def highlight_best(row):
            return (['background-color: #dbeafe'] * len(row)
                    if abs(row["Service Level (%)"] - best["Service Level (%)"]) < 0.1
                    else [''] * len(row))

        st.dataframe(
            df_sweep.style
                .apply(highlight_best, axis=1)
                .format({
                    "Service Level (%)":     "{:.1f}%",
                    "Surge %":               "{:.1f}%",
                    "Base Spend (£)":        "£{:,.0f}",
                    "Surge Spend (£)":       "£{:,.0f}",
                    "Holding Cost (£)":      "£{:,.0f}",
                    "Exp. Profit (£)":       "£{:,.0f}",
                }),
            use_container_width=True, hide_index=True
        )
        st.download_button("📥 Download Sweep (.CSV)",
                           data=df_sweep.to_csv(index=False).encode("utf-8"),
                           file_name="dual_source_sweep.csv", mime="text/csv")

else:
    st.info("👈 Configure your supplier parameters in the sidebar and click **Run Optimizer**.")
    st.markdown("""
    ### How this works

    **Base supplier:** Under-ordering from base doesn't lose the sale — the surge supplier catches it.
    - Cu (base) = `surge cost − base cost` — just the premium paid per unit escalated to surge
    - Co (base) = `base cost − salvage + holding cost` — markdown loss plus carrying cost over the base lead time

    This deliberately produces a conservative base commitment. If holding cost is low, Co falls,
    the critical ratio rises, and the base order increases — exactly as intuition suggests.

    **Surge supplier:** Surge is the last resort — a stockout there is a genuine lost sale.
    - Cu (surge) = `price − surge cost` — full lost margin
    - Co (surge) = `surge cost − salvage + holding cost`

    **The sweep:** The base Q is fixed at its newsvendor optimal. The model tests every surge top-up
    from the 50th to 99.9th demand percentile, computes expected profit at each level using the exact
    normal loss function, and identifies the total quantity where profit peaks.

    **Holding costs:** Calculated weekly (annual % ÷ 52) and applied over the base lead time in weeks.
    A longer lead time increases holding cost per unit, raising Co and pulling the base critical ratio
    — and the optimal service level — down.
    """)
