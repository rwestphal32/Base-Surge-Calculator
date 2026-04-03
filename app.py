import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math

st.set_page_config(page_title="Dual Sourcing Optimizer", layout="wide")

# ─────────────────────────────────────────────
# MATH CORE
# ─────────────────────────────────────────────

def expected_metrics(q, mu, sigma):
    """Normal loss function for exact newsvendor expected values."""
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


def overage_cost(unit_cost, salvage, pipeline_hold, scenario):
    """
    Co = cost of committing one unit too many.
    Pipeline holding is always included (you paid WACC regardless of what happens to the unit).
    Markdown risk depends on lifecycle scenario.
      Shelf-Stable : unsold inventory carries at cost — no markdown, just hold cost
      FMCG         : 50% chance of full markdown; 50% re-enters next cycle at cost
      End of Life  : full markdown to salvage value
    """
    hold = pipeline_hold
    if scenario == "End of Life (Sunset)":
        return (unit_cost - salvage) + hold
    elif scenario == "FMCG (Risk of Obsolescence)":
        return (unit_cost - salvage) * 0.5 + hold
    else:  # Shelf-Stable
        return hold


def salvage_recovery(exp_leftover, unit_cost, salvage, scenario):
    """
    Expected cash recovered from leftover units under each scenario.
    Shelf-Stable: carry at cost (no loss, no gain beyond cost recovery)
    FMCG:        50% at salvage, 50% at cost
    EOL:         all at salvage
    """
    if scenario == "End of Life (Sunset)":
        return salvage * exp_leftover
    elif scenario == "FMCG (Risk of Obsolescence)":
        return (salvage * 0.5 + unit_cost * 0.5) * exp_leftover
    else:
        return unit_cost * exp_leftover


def invested_capital(unit_cost, q, lead_time_weeks):
    """
    Capital tied up in the pipeline at the moment of order.
    = unit_cost × quantity × (lead_time / 52)
    This is the denominator for ROIC — the cash deployed before a single unit is sold.
    """
    return unit_cost * q * (lead_time_weeks / 52.0)


def strategy_metrics(q, unit_cost, tlc, sigma, lt_weeks, mu, price, salvage, pipeline_hold, scenario):
    """
    Compute profit and ROIC for a single-source strategy.
    Spend = TLC × committed quantity (not just expected sales — you pay for what you order).
    """
    exp_sales, exp_leftover, exp_stockout = expected_metrics(q, mu, sigma)
    revenue    = price * exp_sales
    salvage_rv = salvage_recovery(exp_leftover, unit_cost, salvage, scenario)
    spend      = tlc * q                          # full committed quantity
    profit     = revenue + salvage_rv - spend
    inv_cap    = invested_capital(unit_cost, q, lt_weeks)
    roic       = profit / inv_cap if inv_cap > 0 else 0.0
    return dict(q=q, exp_sales=exp_sales, exp_leftover=exp_leftover, exp_stockout=exp_stockout,
                revenue=revenue, salvage_rv=salvage_rv, spend=spend,
                profit=profit, inv_cap=inv_cap, roic=roic)


def newsvendor_q(cu, co, mu, sigma):
    if (cu + co) <= 0:
        return None
    cr = cu / (cu + co)
    z  = stats.norm.ppf(cr)
    return dict(cu=cu, co=co, cr=cr, z=z, q=max(0.0, mu + z * sigma))


def dual_sweep(price, salvage, mu, sigma_surge,
               base_cost, base_tlc, base_lt,
               surge_cost, surge_tlc, surge_lt,
               q_base_fixed, scenario):
    """
    Sweep target service levels 50–99.9%.
    Base Q is fixed. Surge quantity fills to each target.
    Both profit and ROIC are computed at every point so the user can choose the objective.
    Spend = TLC × COMMITTED quantities for both suppliers.
    Salvage recovery applied to expected leftover, allocated proportionally.
    """
    results = []
    for pct in np.arange(0.50, 0.999, 0.005):
        q_target   = mu + stats.norm.ppf(pct) * sigma_surge
        q_surge    = max(0.0, q_target - q_base_fixed)
        q_total    = q_base_fixed + q_surge

        exp_sales, exp_leftover, exp_stockout = expected_metrics(q_total, mu, sigma_surge)

        # Allocate leftover proportionally between suppliers
        base_frac      = q_base_fixed / q_total if q_total > 0 else 1.0
        base_leftover  = exp_leftover * base_frac
        surge_leftover = exp_leftover * (1.0 - base_frac)

        revenue       = price * exp_sales
        salvage_base  = salvage_recovery(base_leftover,  base_cost,  salvage, scenario)
        salvage_surge = salvage_recovery(surge_leftover, surge_cost, salvage, scenario)
        base_spend    = base_tlc  * q_base_fixed   # committed quantity
        surge_spend   = surge_tlc * q_surge         # committed quantity
        profit        = revenue + salvage_base + salvage_surge - base_spend - surge_spend

        # Invested capital: both pipelines running simultaneously
        ic_base  = invested_capital(base_cost,  q_base_fixed, base_lt)
        ic_surge = invested_capital(surge_cost, q_surge,      surge_lt)
        inv_cap  = ic_base + ic_surge
        roic     = profit / inv_cap if inv_cap > 0 else 0.0

        results.append({
            "Service Level (%)":     round(pct * 100, 1),
            "Q Base":                int(q_base_fixed),
            "Q Surge":               int(q_surge),
            "Q Total":               int(q_total),
            "Base Spend (£)":        round(base_spend),
            "Surge Spend (£)":       round(surge_spend),
            "Exp. Sales":            round(exp_sales),
            "Exp. Leftover (units)": round(exp_leftover),
            "Exp. Stockout (units)": round(exp_stockout),
            "Exp. Profit (£)":       round(profit),
            "Invested Capital (£)":  round(inv_cap),
            "ROIC":                  round(roic, 4),
        })

    return pd.DataFrame(results)


# ─────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────

app_mode = st.sidebar.radio("Mode", ["🚀 Pro Mode (Dashboard)", "🎓 Learning Mode"])
st.sidebar.markdown("---")

# ══════════════════════════════════════════════
# PRO MODE
# ══════════════════════════════════════════════
if app_mode == "🚀 Pro Mode (Dashboard)":
    st.title("⚖️ Dual Sourcing: Profit & ROIC Optimizer")
    st.markdown(
        "**The core problem with naive analysis:** Raw profit always favors the cheap base supplier "
        "because its invoice cost is lower. But that ignores the working capital tied up in a long "
        "pipeline. This tool compares strategies on both **expected profit** and **ROIC** — "
        "capital efficiency that a base-cost-only view completely misses."
    )

    with st.sidebar:
        st.header("📦 Product Economics")
        selling_price = st.number_input("Selling Price (£)", value=60.0, step=1.0)
        salvage_value = st.number_input("Salvage / Markdown Value (£)", value=15.0, step=1.0,
                                        help="Recovery value per unsold unit.")
        scenario = st.radio("Lifecycle Scenario", [
            "Shelf-Stable (Ongoing)",
            "FMCG (Risk of Obsolescence)",
            "End of Life (Sunset)"
        ], index=2)

        st.markdown("---")
        st.header("📈 Demand Profile")
        mean_demand    = st.number_input("Mean Period Demand (units)", value=1000, step=50)
        volatility_pct = st.slider("Demand Volatility (CV %)", 5, 80, 25,
                                   help="Coefficient of Variation of demand.")
        sigma_base     = mean_demand * (volatility_pct / 100.0)
        st.caption(f"Std Dev at order time: **{int(sigma_base):,} units**")

        st.markdown("---")
        st.header("🏭 Base Supplier (Cheap, Slow)")
        base_cost     = st.number_input("Invoice Cost — Base (£)", value=20.0, step=0.5)
        base_lead_time = st.number_input("Lead Time (weeks)", value=12, step=1, min_value=1)
        base_moq      = st.number_input("MOQ — Base (units)", value=500, step=100)

        st.markdown("---")
        st.header("⚡ Surge Supplier (Expensive, Fast)")
        surge_cost     = st.number_input("Invoice Cost — Surge (£)", value=22.0, step=0.5)
        surge_lead_time = st.number_input("Lead Time (weeks)", value=2, step=1, min_value=1)

        st.markdown("---")
        st.header("💰 Cost of Capital")
        wacc = st.slider("Annual WACC (%)", 5, 40, 20,
                         help="Used to compute pipeline holding cost and invested capital for ROIC.") / 100.0

        st.markdown("---")
        st.header("🎯 Optimization Objective")
        obj = st.radio("Optimize the dual-source split to maximize:",
                       ["Expected Profit", "ROIC"],
                       help="Profit-max finds the split with the highest absolute return. "
                            "ROIC-max finds the split that generates the best return per £ of capital deployed.")

        run = st.button("🚀 Run Optimizer", type="primary", use_container_width=True)

    if run:
        # ── DERIVED PARAMETERS ─────────────────────
        if surge_cost <= base_cost:
            st.error("Surge invoice cost must exceed base invoice cost.")
            st.stop()
        if selling_price <= surge_cost:
            st.error("Selling price must exceed surge cost.")
            st.stop()

        # TLC = invoice + pipeline holding (WACC × weeks)
        base_hold  = (base_cost  * wacc / 52.0) * base_lead_time
        surge_hold = (surge_cost * wacc / 52.0) * surge_lead_time
        base_tlc   = base_cost  + base_hold
        surge_tlc  = surge_cost + surge_hold

        # Variance reduction: surge is ordered closer to the selling window
        sigma_surge = sigma_base * math.sqrt(surge_lead_time / base_lead_time)

        # ── NEWSVENDOR PARAMETERS ──────────────────
        cu_base  = surge_tlc - base_tlc    # under-ordering base = pay surge premium, not lose sale
        co_base  = overage_cost(base_cost,  salvage_value, base_hold,  scenario)
        cu_surge = selling_price - surge_tlc
        co_surge = overage_cost(surge_cost, salvage_value, surge_hold, scenario)

        nv_base  = newsvendor_q(cu_base,  co_base,  mean_demand, sigma_base)
        nv_surge = newsvendor_q(cu_surge, co_surge, mean_demand, sigma_surge)

        if not nv_base or not nv_surge:
            st.error("Invalid cost parameters — check that costs sit between salvage and price.")
            st.stop()

        # ── SINGLE-SOURCE BASELINES ────────────────
        m_base_only  = strategy_metrics(nv_base["q"],  base_cost,  base_tlc,  sigma_base,
                                         base_lead_time,  mean_demand, selling_price,
                                         salvage_value, base_hold,  scenario)
        m_surge_only = strategy_metrics(nv_surge["q"], surge_cost, surge_tlc, sigma_surge,
                                         surge_lead_time, mean_demand, selling_price,
                                         salvage_value, surge_hold, scenario)

        # Apply MOQ floor to base commit
        q_base_fixed = max(float(base_moq), nv_base["q"])

        # ── SWEEP ──────────────────────────────────
        df = dual_sweep(selling_price, salvage_value, mean_demand, sigma_surge,
                        base_cost, base_tlc, base_lead_time,
                        surge_cost, surge_tlc, surge_lead_time,
                        q_base_fixed, scenario)

        if obj == "Expected Profit":
            best = df.loc[df["Exp. Profit (£)"].idxmax()]
        else:
            best = df.loc[df["ROIC"].idxmax()]

        # ── KPI HEADER ─────────────────────────────
        st.markdown("---")
        st.subheader(f"🎯 Optimal Split — maximizing {obj} | Scenario: {scenario}")

        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Target Service Level",  f"{best['Service Level (%)']:.1f}%")
        k2.metric("Base Order",            f"{int(best['Q Base']):,} units")
        k3.metric("Surge Order",           f"{int(best['Q Surge']):,} units")
        k4.metric("Expected Profit",       f"£{int(best['Exp. Profit (£)']):,}")
        k5.metric("ROIC",                  f"{best['ROIC']:.1%}")

        # ── TABS ───────────────────────────────────
        t1, t2, t3, t4 = st.tabs([
            "📊 Strategy Comparison",
            "📈 Profit & ROIC Curves",
            "⚙️ TLC & Mechanics",
            "📐 Demand Distribution"
        ])

        # ── TAB 1: STRATEGY COMPARISON ─────────────
        with t1:
            st.markdown(
                "Three strategies evaluated on a **level playing field**: same demand distribution, "
                "same salvage treatment, same period. The ROIC column is where base-supplier bias "
                "disappears — a longer pipeline ties up proportionally more capital."
            )

            # Build comparison table
            comp = pd.DataFrame([
                {
                    "Strategy":          "🏭 Base Only",
                    "Order Qty":         int(m_base_only["q"]),
                    "Sigma Used":        f"{int(sigma_base):,}",
                    "Exp. Profit (£)":   int(m_base_only["profit"]),
                    "Invested Capital (£)": int(m_base_only["inv_cap"]),
                    "ROIC":              m_base_only["roic"],
                    "Exp. Stockout":     int(m_base_only["exp_stockout"]),
                    "Exp. Leftover":     int(m_base_only["exp_leftover"]),
                },
                {
                    "Strategy":          "⚡ Surge Only",
                    "Order Qty":         int(m_surge_only["q"]),
                    "Sigma Used":        f"{int(sigma_surge):,}",
                    "Exp. Profit (£)":   int(m_surge_only["profit"]),
                    "Invested Capital (£)": int(m_surge_only["inv_cap"]),
                    "ROIC":              m_surge_only["roic"],
                    "Exp. Stockout":     int(m_surge_only["exp_stockout"]),
                    "Exp. Leftover":     int(m_surge_only["exp_leftover"]),
                },
                {
                    "Strategy":          "⚖️ Dual Sourcing",
                    "Order Qty":         int(best["Q Total"]),
                    "Sigma Used":        f"{int(sigma_surge):,}",
                    "Exp. Profit (£)":   int(best["Exp. Profit (£)"]),
                    "Invested Capital (£)": int(best["Invested Capital (£)"]),
                    "ROIC":              float(best["ROIC"]),
                    "Exp. Stockout":     int(best["Exp. Stockout (units)"]),
                    "Exp. Leftover":     int(best["Exp. Leftover (units)"]),
                },
            ])

            st.dataframe(
                comp.style.format({
                    "Exp. Profit (£)":      "£{:,.0f}",
                    "Invested Capital (£)": "£{:,.0f}",
                    "ROIC":                 "{:.1%}",
                }),
                use_container_width=True, hide_index=True
            )

            st.markdown("---")
            st.markdown("#### Why ROIC diverges from Profit")
            st.markdown(
                f"- **Base supplier** commits **{int(m_base_only['q']):,} units** at **{base_lead_time} weeks** lead time, "
                f"tying up **£{int(m_base_only['inv_cap']):,}** in pipeline capital before a unit is sold.\n"
                f"- **Surge supplier** commits **{int(m_surge_only['q']):,} units** at **{surge_lead_time} weeks**, "
                f"deploying only **£{int(m_surge_only['inv_cap']):,}** — a **{m_surge_only['inv_cap']/m_base_only['inv_cap']:.0%}** "
                f"smaller capital footprint.\n"
                f"- The surge supplier's higher invoice cost (£{surge_cost} vs £{base_cost}) is partially offset by "
                f"**£{base_hold - surge_hold:.2f}/unit** less in pipeline WACC charges "
                f"(TLC gap: £{surge_tlc - base_tlc:.2f} vs invoice gap: £{surge_cost - base_cost:.2f}).\n"
                f"- Under **{scenario}**, leftover inventory is worth "
                f"{'salvage value only' if scenario == 'End of Life (Sunset)' else 'partially recoverable'}, "
                f"which {'heavily penalises' if scenario == 'End of Life (Sunset)' else 'moderately affects'} "
                f"over-ordering from the base supplier."
            )

            # Bar chart: Profit vs ROIC side by side
            fig_comp = make_subplots(rows=1, cols=2,
                                     subplot_titles=["Expected Profit (£)", "ROIC (%)"])
            strategies = ["Base Only", "Surge Only", "Dual Sourcing"]
            profits    = [m_base_only["profit"], m_surge_only["profit"], best["Exp. Profit (£)"]]
            roics      = [m_base_only["roic"]*100, m_surge_only["roic"]*100, float(best["ROIC"])*100]
            colors     = ["#10b981", "#f59e0b", "#2563eb"]

            fig_comp.add_trace(go.Bar(x=strategies, y=profits, marker_color=colors, showlegend=False), row=1, col=1)
            fig_comp.add_trace(go.Bar(x=strategies, y=roics,   marker_color=colors, showlegend=False), row=1, col=2)
            fig_comp.update_layout(height=350, template="plotly_white")
            fig_comp.update_yaxes(title_text="£", row=1, col=1)
            fig_comp.update_yaxes(title_text="%", row=1, col=2)
            st.plotly_chart(fig_comp, use_container_width=True)

        # ── TAB 2: CURVES ──────────────────────────
        with t2:
            st.markdown(
                "Both curves are plotted across the full service level sweep. "
                "The two objectives **diverge** — the profit-maximising split is not the same "
                "as the ROIC-maximising split. Which matters depends on whether you're capital-constrained."
            )

            best_profit_row = df.loc[df["Exp. Profit (£)"].idxmax()]
            best_roic_row   = df.loc[df["ROIC"].idxmax()]

            fig2 = make_subplots(specs=[[{"secondary_y": True}]])
            fig2.add_trace(go.Scatter(
                x=df["Service Level (%)"], y=df["Exp. Profit (£)"],
                name="Expected Profit (£)", line=dict(color="#2563eb", width=3)
            ), secondary_y=False)
            fig2.add_trace(go.Scatter(
                x=df["Service Level (%)"], y=df["ROIC"] * 100,
                name="ROIC (%)", line=dict(color="#f59e0b", width=2, dash="dot")
            ), secondary_y=True)

            fig2.add_vline(x=best_profit_row["Service Level (%)"], line_dash="dash", line_color="#2563eb",
                           annotation_text=f"Profit-max: {best_profit_row['Service Level (%)']:.1f}%",
                           annotation_position="top left")
            fig2.add_vline(x=best_roic_row["Service Level (%)"], line_dash="dash", line_color="#f59e0b",
                           annotation_text=f"ROIC-max: {best_roic_row['Service Level (%)']:.1f}%",
                           annotation_position="top right")

            fig2.update_layout(
                title="Expected Profit & ROIC vs Target Service Level",
                xaxis_title="Target Service Level (%) — combined base + surge",
                height=420, template="plotly_white",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            fig2.update_yaxes(title_text="Expected Profit (£)", secondary_y=False)
            fig2.update_yaxes(title_text="ROIC (%)", secondary_y=True)
            st.plotly_chart(fig2, use_container_width=True)

            c1, c2 = st.columns(2)
            c1.metric("Profit-maximising split",
                      f"£{int(best_profit_row['Exp. Profit (£)']):,}",
                      delta=f"ROIC: {best_profit_row['ROIC']:.1%} | SL: {best_profit_row['Service Level (%)']:.1f}%",
                      delta_color="off")
            c2.metric("ROIC-maximising split",
                      f"{best_roic_row['ROIC']:.1%}",
                      delta=f"Profit: £{int(best_roic_row['Exp. Profit (£)']):,} | SL: {best_roic_row['Service Level (%)']:.1f}%",
                      delta_color="off")

            # Invested capital curve
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(
                x=df["Service Level (%)"], y=df["Invested Capital (£)"],
                name="Invested Capital", fill='tozeroy',
                fillcolor='rgba(239,68,68,0.1)', line=dict(color="#ef4444", width=2)
            ))
            fig3.update_layout(
                title="Invested Capital vs Service Level (more surge = less capital)",
                xaxis_title="Target Service Level (%)",
                yaxis_title="Invested Capital (£)",
                height=300, template="plotly_white"
            )
            st.plotly_chart(fig3, use_container_width=True)
            st.caption(
                "As you increase the surge allocation, the base Q stays fixed but surge pipeline capital "
                f"({surge_lead_time} weeks) grows slowly vs base ({base_lead_time} weeks). "
                "Total invested capital rises, but more slowly than profit — until the surge premium dominates."
            )

        # ── TAB 3: TLC & MECHANICS ─────────────────
        with t3:
            st.markdown("### Total Landed Cost & Newsvendor Parameters")
            st.markdown(
                "TLC converts invoice prices into economically comparable costs by including "
                "the WACC charge on capital tied up in transit. This narrows — but rarely closes — "
                "the raw invoice gap between base and surge."
            )

            m1, m2, m3 = st.columns(3)
            m1.metric("Base TLC",         f"£{base_tlc:.2f}",
                      help=f"£{base_cost} invoice + £{base_hold:.2f} WACC over {base_lead_time} wks")
            m2.metric("Surge TLC",        f"£{surge_tlc:.2f}",
                      help=f"£{surge_cost} invoice + £{surge_hold:.2f} WACC over {surge_lead_time} wks")
            m3.metric("True TLC Premium", f"£{surge_tlc - base_tlc:.2f}",
                      delta=f"vs invoice gap £{surge_cost - base_cost:.2f}",
                      delta_color="off")

            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                with st.container(border=True):
                    st.markdown("**🏭 Base Supplier — Newsvendor**")
                    st.markdown(
                        f"- **Cu** = Surge TLC − Base TLC = **£{nv_base['cu']:.2f}**/unit  \n"
                        f"  *(under-ordering from base means paying the surge premium, not losing the sale)*\n"
                        f"- **Co** = **£{nv_base['co']:.2f}**/unit  \n"
                        f"  *(pipeline hold £{base_hold:.2f} + "
                        f"{'full markdown £' + str(round(base_cost - salvage_value, 2)) if scenario == 'End of Life (Sunset)' else 'scenario markdown'})*\n"
                        f"- **Critical Ratio:** {nv_base['cr']:.3f} → {nv_base['cr']*100:.1f}th percentile\n"
                        f"- **Newsvendor Q:** {int(nv_base['q']):,} units (MOQ floor: {int(q_base_fixed):,})\n"
                        f"- **Sigma used:** {int(sigma_base):,} (full forecast uncertainty — ordered now)"
                    )
            with col2:
                with st.container(border=True):
                    st.markdown("**⚡ Surge Supplier — Newsvendor**")
                    st.markdown(
                        f"- **Cu** = Price − Surge TLC = **£{nv_surge['cu']:.2f}**/unit  \n"
                        f"  *(surge is last resort — a stockout here is a genuine lost sale)*\n"
                        f"- **Co** = **£{nv_surge['co']:.2f}**/unit  \n"
                        f"  *(pipeline hold £{surge_hold:.2f} + "
                        f"{'full markdown £' + str(round(surge_cost - salvage_value, 2)) if scenario == 'End of Life (Sunset)' else 'scenario markdown'})*\n"
                        f"- **Critical Ratio:** {nv_surge['cr']:.3f} → {nv_surge['cr']*100:.1f}th percentile\n"
                        f"- **Newsvendor Q:** {int(nv_surge['q']):,} units\n"
                        f"- **Sigma used:** {int(sigma_surge):,} (reduced — ordered closer to selling season)"
                    )

            st.markdown("---")
            st.markdown("#### Lifecycle Scenario Impact on Co")
            sc_table = pd.DataFrame([
                {"Scenario": "Shelf-Stable",      "Base Co (£)": round(overage_cost(base_cost,  salvage_value, base_hold,  "Shelf-Stable (Ongoing)"), 2),
                                                   "Surge Co (£)": round(overage_cost(surge_cost, salvage_value, surge_hold, "Shelf-Stable (Ongoing)"), 2)},
                {"Scenario": "FMCG",              "Base Co (£)": round(overage_cost(base_cost,  salvage_value, base_hold,  "FMCG (Risk of Obsolescence)"), 2),
                                                   "Surge Co (£)": round(overage_cost(surge_cost, salvage_value, surge_hold, "FMCG (Risk of Obsolescence)"), 2)},
                {"Scenario": "End of Life",        "Base Co (£)": round(overage_cost(base_cost,  salvage_value, base_hold,  "End of Life (Sunset)"), 2),
                                                   "Surge Co (£)": round(overage_cost(surge_cost, salvage_value, surge_hold, "End of Life (Sunset)"), 2)},
            ])
            st.dataframe(sc_table, use_container_width=True, hide_index=True)
            st.caption(
                "Higher Co → lower critical ratio → smaller order. EOL scenario produces the most "
                "conservative base order and the strongest case for shifting volume to surge."
            )

        # ── TAB 4: DEMAND DISTRIBUTION ─────────────
        with t4:
            st.markdown(
                "The base order is placed now, against the wide (uncertain) forecast. "
                "The surge order is placed later, against a much tighter forecast — "
                "this is the variance reduction benefit of a shorter lead time."
            )

            x = np.linspace(mean_demand - 4 * sigma_base, mean_demand + 4 * sigma_base, 600)
            y_base  = stats.norm.pdf(x, mean_demand, sigma_base)
            y_surge = stats.norm.pdf(x, mean_demand, sigma_surge)

            q_b = int(best["Q Base"])
            q_t = int(best["Q Total"])

            fig4 = go.Figure()
            fig4.add_trace(go.Scatter(
                x=x, y=y_base, mode='lines',
                line=dict(color='#10b981', width=2, dash='dash'),
                name=f'Base forecast σ={int(sigma_base):,} (order now)'
            ))
            fig4.add_trace(go.Scatter(
                x=x, y=y_surge, fill='tozeroy',
                fillcolor='rgba(37,99,235,0.1)', mode='lines',
                line=dict(color='#2563eb', width=2),
                name=f'Surge forecast σ={int(sigma_surge):,} (order later)'
            ))

            # Shade zones against surge distribution
            mask_base  = x <= q_b
            mask_surge = (x > q_b) & (x <= q_t)
            mask_stock = x > q_t

            fig4.add_trace(go.Scatter(
                x=np.concatenate([x[mask_base], [q_b, x[mask_base][0]]]),
                y=np.concatenate([y_surge[mask_base], [0, 0]]),
                fill='toself', fillcolor='rgba(16,185,129,0.2)',
                line=dict(color='rgba(0,0,0,0)'), name='Base Coverage'
            ))
            if mask_surge.any():
                fig4.add_trace(go.Scatter(
                    x=np.concatenate([[q_b], x[mask_surge], [q_t, q_b]]),
                    y=np.concatenate([[0], y_surge[mask_surge], [0, 0]]),
                    fill='toself', fillcolor='rgba(245,158,11,0.2)',
                    line=dict(color='rgba(0,0,0,0)'), name='Surge Coverage'
                ))
            if mask_stock.any():
                fig4.add_trace(go.Scatter(
                    x=np.concatenate([[q_t], x[mask_stock], [x[mask_stock][-1], q_t]]),
                    y=np.concatenate([[0], y_surge[mask_stock], [0, 0]]),
                    fill='toself', fillcolor='rgba(239,68,68,0.2)',
                    line=dict(color='rgba(0,0,0,0)'), name='Residual Stockout Risk'
                ))

            fig4.add_vline(x=mean_demand, line_dash="dot", line_color="grey", annotation_text="Mean")
            fig4.add_vline(x=q_b, line_dash="dash", line_color="#10b981",
                           annotation_text=f"Base Q: {q_b:,}")
            fig4.add_vline(x=q_t, line_dash="dash", line_color="#f59e0b",
                           annotation_text=f"Total Q: {q_t:,}")
            fig4.update_layout(
                height=450, template="plotly_white",
                xaxis_title="Demand (units)", yaxis_title="Probability Density",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig4, use_container_width=True)

            st.markdown(
                f"🟢 **Green** — demand covered by base order. Overage risk: Co = £{nv_base['co']:.2f}/unit.  \n"
                f"🟡 **Amber** — demand covered by surge. Each unit costs £{surge_tlc - base_tlc:.2f} more (TLC gap).  \n"
                f"🔴 **Red** — above total order. Lost sale at £{selling_price - surge_tlc:.2f}/unit net margin."
            )

    else:
        st.info("👈 Configure parameters in the sidebar and click **Run Optimizer**.")
        st.markdown("""
        ### Why raw profit comparisons mislead

        Ordering from a cheap base supplier with a 12-week lead time looks profitable on paper.
        But that capital is locked in a ship for 12 weeks before a single unit is sold.

        **Total Landed Cost (TLC)** corrects for this:
        `TLC = invoice cost + (invoice × WACC/52 × lead_time_weeks)`

        This narrows the gap between base and surge — but the bigger fix is **ROIC**:
        `ROIC = Expected Profit / Invested Capital`

        where Invested Capital = `unit_cost × quantity × (lead_time / 52)`.

        A surge supplier with a 2-week lead time deploys a fraction of the capital for a similar
        profit, producing dramatically higher ROIC. This tool shows both objectives and lets you
        choose which one to optimize the dual-source split against.
        """)


# ══════════════════════════════════════════════
# LEARNING MODE
# ══════════════════════════════════════════════
else:
    st.sidebar.info("🎓 **Learning Mode Active**")
    st.title("🎓 Dual Sourcing Masterclass")

    l_price, l_salvage, l_mu = 60.0, 15.0, 1000
    l_base_cost, l_wacc = 20.0, 0.20

    t1, t2, t3, t4, t5 = st.tabs([
        "1. The Dilemma",
        "2. Total Landed Cost",
        "3. The Hedge (Cu vs Co)",
        "4. Lead Time & Variance",
        "5. Profit vs ROIC"
    ])

    with t1:
        st.subheader("The Problem with Forecasting")
        st.markdown(
            "Demand is uncertain. If you order too little, you stockout and lose margin. "
            "If you order too much, you're stuck with inventory you have to mark down. "
            "The base supplier is cheap but forces you to commit early — against a wide, uncertain forecast."
        )
        l_vol = st.slider("Demand Volatility (CV %)", 5, 80, 25, key="l_vol")
        l_sigma = l_mu * (l_vol / 100.0)
        x = np.linspace(200, 1800, 600)
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=x, y=stats.norm.pdf(x, l_mu, l_sigma),
            fill='tozeroy', fillcolor='rgba(37,99,235,0.15)',
            line=dict(color='#2563eb', width=3)
        ))
        fig1.add_vline(x=l_mu, line_dash="dash", line_color="black", annotation_text="Mean Demand")
        fig1.update_layout(height=320, template="plotly_white",
                           xaxis_title="Demand", yaxis_title="Probability Density")
        st.plotly_chart(fig1, use_container_width=True)
        st.markdown(
            f"At **{l_vol}% CV**, one standard deviation of demand is **±{int(l_sigma):,} units**. "
            "Ordering to the mean leaves you exposed to both tails."
        )

    with t2:
        st.subheader("Total Landed Cost: The Working Capital Secret")
        st.markdown(
            "Comparing invoice prices is misleading. Every week inventory spends in transit, "
            "you've paid for it but can't sell it — that's an opportunity cost at your WACC rate."
        )
        c1, c2 = st.columns(2)
        l_surge_cost = c1.slider("Surge Invoice Price (£)", 20.5, 40.0, 22.0, step=0.5)
        l_base_lt    = c2.slider("Base Lead Time (weeks)", 4, 24, 12)

        l_base_hold  = (l_base_cost   * l_wacc / 52) * l_base_lt
        l_surge_hold = (l_surge_cost  * l_wacc / 52) * 2
        l_base_tlc   = l_base_cost  + l_base_hold
        l_surge_tlc  = l_surge_cost + l_surge_hold

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Base Invoice",    f"£{l_base_cost:.2f}")
        m2.metric("Base TLC",        f"£{l_base_tlc:.2f}", delta=f"+£{l_base_hold:.2f} holding")
        m3.metric("Surge Invoice",   f"£{l_surge_cost:.2f}")
        m4.metric("Surge TLC",       f"£{l_surge_tlc:.2f}", delta=f"+£{l_surge_hold:.2f} holding")

        st.metric("True TLC Premium (Surge − Base)", f"£{l_surge_tlc - l_base_tlc:.2f}",
                  delta=f"vs invoice gap £{l_surge_cost - l_base_cost:.2f}", delta_color="off")
        st.info(
            f"The invoice gap is £{l_surge_cost - l_base_cost:.2f} but the **true economic gap is only "
            f"£{l_surge_tlc - l_base_tlc:.2f}** once pipeline holding cost is included. "
            f"Longer base lead times compress this further."
        )

    with t3:
        st.subheader("The Cost of Being Wrong (Cu vs Co)")
        st.markdown(
            "The newsvendor model optimises based on the **cost of ordering one unit too few (Cu)** "
            "vs the **cost of ordering one unit too many (Co)**."
        )
        l_scenario = st.radio("Lifecycle Scenario", [
            "Shelf-Stable (Ongoing)", "FMCG (Risk of Obsolescence)", "End of Life (Sunset)"
        ], index=2, key="l_scen")

        l_surge_cost_t3 = 22.0
        l_base_lt_t3    = 12
        l_bh = (l_base_cost     * l_wacc / 52) * l_base_lt_t3
        l_sh = (l_surge_cost_t3 * l_wacc / 52) * 2
        l_btlc = l_base_cost     + l_bh
        l_stlc = l_surge_cost_t3 + l_sh

        cu_b = l_stlc - l_btlc
        co_b = overage_cost(l_base_cost, l_salvage, l_bh, l_scenario)
        cr_b = cu_b / (cu_b + co_b) if (cu_b + co_b) > 0 else 0

        cu_s = l_price - l_stlc
        co_s = overage_cost(l_surge_cost_t3, l_salvage, l_sh, l_scenario)
        cr_s = cu_s / (cu_s + co_s) if (cu_s + co_s) > 0 else 0

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**🏭 Base Supplier**")
            st.metric("Cu (under-order cost)", f"£{cu_b:.2f}",
                      help="Pay surge TLC instead of base TLC — sale is not lost")
            st.metric("Co (over-order cost)", f"£{co_b:.2f}",
                      help="Pipeline hold + scenario markdown penalty")
            st.markdown(f"**Critical Ratio:** `{cu_b:.2f} / ({cu_b:.2f}+{co_b:.2f})` = **{cr_b:.3f}**")
        with c2:
            st.markdown("**⚡ Surge Supplier**")
            st.metric("Cu (under-order cost)", f"£{cu_s:.2f}",
                      help="Full lost margin — no fallback")
            st.metric("Co (over-order cost)", f"£{co_s:.2f}")
            st.markdown(f"**Critical Ratio:** `{cu_s:.2f} / ({cu_s:.2f}+{co_s:.2f})` = **{cr_s:.3f}**")

        st.markdown(
            f"Under **{l_scenario}**, the base CR is **{cr_b:.3f}** and the surge CR is **{cr_s:.3f}**. "
            "The base order is deliberately conservative because under-ordering from base just escalates to surge — it's not a lost sale."
        )

    with t4:
        st.subheader("Lead Time & Variance Reduction")
        st.markdown(
            "The surge order is placed much closer to the selling season. By then, "
            "promotional data, early POS reads, and retailer call-offs have resolved much of the uncertainty. "
            "Mathematically: σ scales with √(lead_time), so a shorter lead time = a tighter forecast."
        )

        l_vol2   = st.slider("Base Volatility (CV %)", 5, 80, 25, key="l_vol2")
        l_base_lt2 = st.slider("Base Lead Time (weeks)", 4, 24, 12, key="l_blt2")
        l_sigma2 = l_mu * (l_vol2 / 100.0)
        l_sigma_surge2 = l_sigma2 * math.sqrt(2 / l_base_lt2)

        x5 = np.linspace(0, 2000, 600)
        fig5 = go.Figure()
        fig5.add_trace(go.Scatter(
            x=x5, y=stats.norm.pdf(x5, l_mu, l_sigma2),
            mode='lines', line=dict(color='#ef4444', width=2, dash='dash'),
            name=f'Base forecast σ={int(l_sigma2):,}'
        ))
        fig5.add_trace(go.Scatter(
            x=x5, y=stats.norm.pdf(x5, l_mu, l_sigma_surge2),
            fill='tozeroy', fillcolor='rgba(16,185,129,0.2)',
            line=dict(color='#10b981', width=3),
            name=f'Surge forecast σ={int(l_sigma_surge2):,}'
        ))
        fig5.add_vline(x=l_mu, line_dash="dash", line_color="black")
        fig5.update_layout(height=300, template="plotly_white",
                           xaxis_title="Demand", yaxis_title="Probability Density")
        st.plotly_chart(fig5, use_container_width=True)
        st.info(
            f"At {l_base_lt2}-week base lead time and 2-week surge lead time, "
            f"the surge forecast is **{l_sigma_surge2/l_sigma2:.0%}** as uncertain as the base forecast. "
            "This means the surge supplier can cover the uncertain tail with far less overstock risk."
        )

    with t5:
        st.subheader("Profit vs ROIC: Why They Tell Different Stories")
        st.markdown(
            "The base supplier almost always wins on raw profit in isolation — its unit cost is lower. "
            "But ROIC penalises it for the capital it locks up. This tab shows the divergence."
        )

        l_scenario5    = "End of Life (Sunset)"
        l_surge_cost5  = 22.0
        l_base_lt5     = 12
        l_bh5 = (l_base_cost    * l_wacc / 52) * l_base_lt5
        l_sh5 = (l_surge_cost5  * l_wacc / 52) * 2
        l_btlc5 = l_base_cost   + l_bh5
        l_stlc5 = l_surge_cost5 + l_sh5
        l_sigma5       = l_mu * 0.25
        l_sigma_surge5 = l_sigma5 * math.sqrt(2 / l_base_lt5)

        nv_b5 = newsvendor_q(l_stlc5 - l_btlc5,
                             overage_cost(l_base_cost,   l_salvage, l_bh5, l_scenario5),
                             l_mu, l_sigma5)
        nv_s5 = newsvendor_q(l_price - l_stlc5,
                             overage_cost(l_surge_cost5, l_salvage, l_sh5, l_scenario5),
                             l_mu, l_sigma_surge5)

        m_b5 = strategy_metrics(nv_b5["q"], l_base_cost,   l_btlc5, l_sigma5,       l_base_lt5, l_mu, l_price, l_salvage, l_bh5, l_scenario5)
        m_s5 = strategy_metrics(nv_s5["q"], l_surge_cost5, l_stlc5, l_sigma_surge5, 2,           l_mu, l_price, l_salvage, l_sh5, l_scenario5)

        comp5 = pd.DataFrame([
            {"Strategy": "🏭 Base Only",    "Order Qty": int(m_b5["q"]), "Exp. Profit (£)": int(m_b5["profit"]),
             "Invested Capital (£)": int(m_b5["inv_cap"]), "ROIC": m_b5["roic"]},
            {"Strategy": "⚡ Surge Only",   "Order Qty": int(m_s5["q"]), "Exp. Profit (£)": int(m_s5["profit"]),
             "Invested Capital (£)": int(m_s5["inv_cap"]), "ROIC": m_s5["roic"]},
        ])
        st.dataframe(comp5.style.format({
            "Exp. Profit (£)": "£{:,.0f}", "Invested Capital (£)": "£{:,.0f}", "ROIC": "{:.1%}"
        }), use_container_width=True, hide_index=True)

        st.markdown(
            f"Base supplier: **£{int(m_b5['profit']):,}** profit on **£{int(m_b5['inv_cap']):,}** capital = **{m_b5['roic']:.1%} ROIC**.  \n"
            f"Surge supplier: **£{int(m_s5['profit']):,}** profit on **£{int(m_s5['inv_cap']):,}** capital = **{m_s5['roic']:.1%} ROIC**.  \n\n"
            "The surge supplier deploys a fraction of the capital. In a capital-constrained business, "
            "that freed-up cash can be redeployed elsewhere — making ROIC the right lens for strategic decisions."
        )
