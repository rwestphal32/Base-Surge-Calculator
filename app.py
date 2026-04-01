import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math

st.set_page_config(page_title="Dual Sourcing Optimizer", layout="wide")

# ─────────────────────────────────────────────
# CORE NEWSVENDOR MATH (Variance Reduction Enabled)
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

def calc_standalone(price, salvage, cost, mu, sigma, holding_per_period, scenario):
    cu = price - cost
    co = calculate_overage_cost(cost, salvage, holding_per_period, scenario)
    if (cu + co) <= 0: return 0, 0
    cr = cu / (cu + co)
    z = stats.norm.ppf(cr)
    q = max(0.0, mu + z * sigma)
    exp_sales, exp_leftover, exp_stockout = expected_metrics(q, mu, sigma)
    
    revenue = price * exp_sales
    if scenario == "End of Life (Sunset)": salvage_rev = salvage * exp_leftover
    elif scenario == "FMCG (Risk of Obsolescence)": salvage_rev = (salvage * 0.5 * exp_leftover) + (cost * 0.5 * exp_leftover)
    else: salvage_rev = cost * exp_leftover
    
    spend = cost * q
    hold_cost = holding_per_period * exp_leftover
    exp_profit = revenue + salvage_rev - spend - hold_cost
    return q, exp_profit

def newsvendor_base(salvage, base_cost, surge_cost, mu, sigma_base, holding_per_period, scenario):
    cu = surge_cost - base_cost
    co = calculate_overage_cost(base_cost, salvage, holding_per_period, scenario)
    if (cu + co) <= 0: return None
    cr = cu / (cu + co)
    z  = stats.norm.ppf(cr)
    q  = max(0.0, mu + z * sigma_base)
    return dict(cu=cu, co=co, critical_ratio=cr, z_score=z, optimal_q=q)

def newsvendor_surge(price, salvage, base_cost, surge_cost, mu, sigma_surge, surge_hold, scenario):
    cu = price - surge_cost
    co = calculate_overage_cost(surge_cost, salvage, surge_hold, scenario)
    if (cu + co) <= 0: return None
    cr = cu / (cu + co)
    z  = stats.norm.ppf(cr)
    q  = max(0.0, mu + z * sigma_surge)
    return dict(cu=cu, co=co, critical_ratio=cr, z_score=z, optimal_q=q)

def dual_source_sweep(price, salvage, mu, sigma_surge, base_cost, surge_cost, base_hold, surge_hold, q_base_fixed, scenario):
    results = []
    for pct in np.arange(0.50, 0.999, 0.005):
        # Target Q is generated against the HIGHLY ACCURATE Surge forecast
        q_target = mu + stats.norm.ppf(pct) * sigma_surge
        q_surge = max(0.0, q_target - q_base_fixed)
        q_total = q_base_fixed + q_surge

        # Outcomes are evaluated against the Surge forecast because that is when the final capacity decision is locked in
        exp_sales, exp_leftover, exp_stockout = expected_metrics(q_total, mu, sigma_surge)
        
        base_frac = q_base_fixed / q_total if q_total > 0 else 1.0
        base_leftover = exp_leftover * base_frac
        surge_leftover = exp_leftover * (1.0 - base_frac)

        revenue = price * exp_sales
        if scenario == "End of Life (Sunset)": salvage_rev = salvage * exp_leftover
        elif scenario == "FMCG (Risk of Obsolescence)": salvage_rev = (salvage * 0.5 * exp_leftover) + (base_cost * 0.5 * exp_leftover)
        else: salvage_rev = base_cost * exp_leftover
        
        base_spend = base_cost * q_base_fixed
        surge_spend = surge_cost * q_surge
        hold_cost = (base_hold * base_leftover) + (surge_hold * surge_leftover)
        exp_profit = revenue + salvage_rev - base_spend - surge_spend - hold_cost

        results.append({
            "Service Level (%)": round(pct * 100, 1), "Q Base": int(q_base_fixed), "Q Surge": int(q_surge),
            "Q Total": int(q_total), "Surge %": round(q_surge / q_total * 100, 1) if q_total > 0 else 0,
            "Base Spend (£)": round(base_spend), "Surge Spend (£)": round(surge_spend),
            "Holding Cost (£)": round(hold_cost), "Exp. Leftover (units)": round(exp_leftover),
            "Exp. Stockout (units)": round(exp_stockout), "Exp. Profit (£)": round(exp_profit),
            "Exp. Sales Total": exp_sales
        })
    return pd.DataFrame(results)


# ─────────────────────────────────────────────
# APP MODE SELECTOR
# ─────────────────────────────────────────────
app_mode = st.sidebar.radio("Select App Mode", ["🎓 Learning Mode (Concepts)", "🚀 Pro Mode (Dashboard)"])
st.sidebar.markdown("---")

if app_mode == "🚀 Pro Mode (Dashboard)":
    st.title("⚖️ Dual Sourcing: Operational Optimizer")
    st.markdown("Configure your supplier economics in the sidebar to generate your optimal procurement split.")

    with st.sidebar:
        st.header("📦 Product Economics")
        selling_price = st.number_input("Selling Price (£)", value=60.0, step=1.0)
        salvage_value = st.number_input("Salvage / Markdown Value (£)", value=15.0, step=1.0)
        scenario = st.radio("Lifecycle Scenario", ["Shelf-Stable (Ongoing)", "FMCG (Risk of Obsolescence)", "End of Life (Sunset)"], index=2)
        
        st.markdown("---")
        st.header("📈 Demand Profile")
        mean_demand = st.number_input("Mean Seasonal/Period Demand", value=1000, step=50)
        volatility_pct = st.slider("Base Demand Volatility (CV %)", 5, 80, 25)
        sigma_base = mean_demand * (volatility_pct / 100.0)
        
        st.markdown("---")
        st.header("🏭 Base Supplier (Cheap, Slow)")
        base_cost = st.number_input("Unit Cost — Base (£)", value=20.0, step=0.5)
        base_lead_time = st.number_input("Lead Time (weeks)", value=12, step=1, min_value=1)
        base_moq = st.number_input("MOQ — Base (units)", value=500, step=100)
        
        st.markdown("---")
        st.header("⚡ Surge Supplier (Expensive, Fast)")
        surge_cost = st.number_input("Unit Cost — Surge (£)", value=22.0, step=0.5)
        surge_lead_time = st.number_input("Surge Lead Time (weeks)", value=2, step=1, min_value=1)
        surge_moq = st.number_input("MOQ — Surge (units)", value=100, step=50)
        
        st.markdown("---")
        st.header("💰 Holding Cost")
        holding_cost_pct = st.slider("Annual Holding Cost / WACC (%)", 10, 40, 20) / 100.0
        
        # Calculate holding costs per period based on lead times
        base_holding_per_period = ((base_cost * holding_cost_pct) / 52.0) * base_lead_time
        surge_holding_per_period = ((surge_cost * holding_cost_pct) / 52.0) * surge_lead_time
        
        # APPLY THE SQUARE ROOT LAW OF LEAD TIME: Volatility shrinks as lead time shortens.
        sigma_surge = sigma_base * math.sqrt(surge_lead_time / base_lead_time)
        
        run = st.button("🚀 Run Optimizer", type="primary", use_container_width=True)

    if run:
        # Standalone comparisons evaluated against their respective uncertainties
        base_q_only, base_profit_only = calc_standalone(selling_price, salvage_value, base_cost, mean_demand, sigma_base, base_holding_per_period, scenario)
        surge_q_only, surge_profit_only = calc_standalone(selling_price, salvage_value, surge_cost, mean_demand, sigma_surge, surge_holding_per_period, scenario)

        nv_base = newsvendor_base(salvage_value, base_cost, surge_cost, mean_demand, sigma_base, base_holding_per_period, scenario)
        nv_surge = newsvendor_surge(selling_price, salvage_value, base_cost, surge_cost, mean_demand, sigma_surge, surge_holding_per_period, scenario)
        
        q_base_fixed = max(float(base_moq), nv_base["optimal_q"])
            
        df_sweep = dual_source_sweep(selling_price, salvage_value, mean_demand, sigma_surge, base_cost, surge_cost, base_holding_per_period, surge_holding_per_period, q_base_fixed, scenario)
        best = df_sweep.loc[df_sweep["Exp. Profit (£)"].idxmax()]

        st.subheader(f"🎯 Optimal Split: {scenario}")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Target Service Level", f"{best['Service Level (%)']:.1f}%")
        k2.metric("Base Order", f"{int(best['Q Base']):,}")
        k3.metric("Surge Order", f"{int(best['Q Surge']):,}")
        k4.metric("Dual-Source Expected Profit", f"£{int(best['Exp. Profit (£)']):,}")

        t1, t2, t3, t4 = st.tabs(["📈 Financials & Strategy", "⚙️ Newsvendor Mechanics", "📐 Demand Distribution", "📋 Raw Data"])
        
        with t1:
            st.markdown("#### Strategy Comparison: Expected Profit")
            st.markdown("By placing the Surge order later (with a shorter lead time), forecast volatility shrinks. This massive reduction in overage risk is what makes Dual Sourcing so profitable.")
            
            sc1, sc2, sc3 = st.columns(3)
            best_single = max(base_profit_only, surge_profit_only)
            value_add = best['Exp. Profit (£)'] - best_single
            sc1.metric("Base Supplier Only", f"£{int(base_profit_only):,}", help=f"Optimal Q if only using Base: {int(base_q_only)}")
            sc2.metric("Surge Supplier Only", f"£{int(surge_profit_only):,}", help=f"Optimal Q if only using Surge: {int(surge_q_only)}")
            sc3.metric("Dual Sourcing (Optimal)", f"£{int(best['Exp. Profit (£)']):,}", delta=f"+£{int(value_add):,} value generated" if value_add > 1 else "0 value generated", delta_color="normal")

            st.markdown("---")
            c_chart, c_pie = st.columns([2, 1])
            with c_chart:
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                fig.add_trace(go.Scatter(x=df_sweep["Service Level (%)"], y=df_sweep["Exp. Profit (£)"], name="Profit", line=dict(color="#2563eb", width=3)), secondary_y=False)
                fig.add_trace(go.Scatter(x=df_sweep["Service Level (%)"], y=df_sweep["Q Surge"], name="Surge Units", line=dict(color="#f59e0b", dash="dot")), secondary_y=True)
                fig.add_vline(x=best["Service Level (%)"], line_dash="dash", line_color="red", annotation_text="Optimum Profit")
                fig.update_layout(height=400, template="plotly_white", margin=dict(l=0,r=0,t=30,b=0), title="Profit vs Target Service Level")
                st.plotly_chart(fig, use_container_width=True)
            
            with c_pie:
                if scenario == "End of Life (Sunset)": md_cost = max(0, int(best["Exp. Leftover (units)"]) * (base_cost - salvage_value))
                elif scenario == "FMCG (Risk of Obsolescence)": md_cost = max(0, int(best["Exp. Leftover (units)"]) * (base_cost - salvage_value) * 0.5)
                else: md_cost = 0
                
                fig_pie = go.Figure(go.Pie(
                    labels=["Base Spend", "Surge Spend", "Exp. Markdowns", "Holding Cost"],
                    values=[best["Base Spend (£)"], best["Surge Spend (£)"], md_cost, best["Holding Cost (£)"]],
                    marker_colors=["#10b981", "#f59e0b", "#ef4444", "#8b5cf6"], hole=0.4
                ))
                fig_pie.update_layout(title="Cost Breakdown at Optimum", height=400, margin=dict(l=0,r=0,t=30,b=0), template="plotly_white")
                st.plotly_chart(fig_pie, use_container_width=True)
                
            st.markdown("#### Show Your Work: Financial Value of the Surge Hedge")
            c_f1, c_f2, c_f3 = st.columns(3)
            
            if int(best["Q Surge"]) == 0:
                expected_surge_sales, surge_investment, revenue_protected, net_surge_margin = 0, 0, 0, 0
            else:
                base_only_sales = expected_metrics(best["Q Base"], mean_demand, sigma_base)[0]
                expected_surge_sales = best["Exp. Sales Total"] - base_only_sales
                surge_investment = int(best["Q Surge"]) * surge_cost
                revenue_protected = int(expected_surge_sales) * selling_price
                net_surge_margin = revenue_protected - surge_investment
            
            c_f1.metric("Surge Investment (Cost)", f"£{surge_investment:,}", help="Total capital spent on the fast supplier.")
            c_f2.metric("Revenue Protected (Benefit)", f"£{revenue_protected:,}", help="Expected top-line revenue captured that would have been lost without the Surge supplier.")
            c_f3.metric("Net Margin Added", f"£{net_surge_margin:,}", help="The bottom-line profit generated purely by utilizing the Surge supplier.")

        with t2:
            st.markdown("The underlying math calculates the optimal service level (Critical Ratio) by balancing the **Cost of Under-ordering ($C_u$)** against the **Cost of Over-ordering ($C_o$)**.")
            col1, col2 = st.columns(2)
            with col1:
                with st.container(border=True):
                    st.markdown("### 🏭 Base Supplier")
                    st.markdown(f"*(Evaluated against 12-week forecast volatility: {volatility_pct:.1f}%)*")
                    st.markdown(f"- **$C_u$:** Surge Cost (£{surge_cost}) - Base Cost (£{base_cost}) = **£{nv_base['cu']:.2f}**/unit")
                    st.markdown(f"- **$C_o$:** Scenario Penalty + Holding = **£{nv_base['co']:.2f}**/unit")
                    st.markdown("---")
                    st.markdown(f"**Critical Ratio ($CR_1$):** `Cu / (Cu + Co)` = **{nv_base['critical_ratio']:.3f}**")
                    st.markdown(f"**Target Z-Score:** **{nv_base['z_score']:.3f}** std deviations")
                    st.markdown(f"**Optimal Base $Q_1$:** **{int(nv_base['optimal_q']):,} units**")
            with col2:
                with st.container(border=True):
                    st.markdown("### ⚡ Surge Supplier")
                    surge_vol = volatility_pct * math.sqrt(surge_lead_time / base_lead_time)
                    st.markdown(f"*(Evaluated against accurate 2-week forecast: {surge_vol:.1f}%)*")
                    st.markdown(f"- **$C_u$:** Price (£{selling_price}) - Surge Cost (£{surge_cost}) = **£{nv_surge['cu']:.2f}**/unit")
                    st.markdown(f"- **$C_o$:** Scenario Penalty + Holding = **£{nv_surge['co']:.2f}**/unit")
                    st.markdown("---")
                    st.markdown(f"**Critical Ratio ($CR_2$):** `Cu / (Cu + Co)` = **{nv_surge['critical_ratio']:.3f}**")
                    st.markdown(f"**Target Z-Score:** **{nv_surge['z_score']:.3f}** std deviations")
                    st.markdown(f"**Total Optimal Target $Q_2$:** **{int(nv_surge['optimal_q']):,} units**")

        with t3:
            x = np.linspace(mean_demand - 4*sigma_base, mean_demand + 4*sigma_base, 600)
            y_pdf_base = stats.norm.pdf(x, mean_demand, sigma_base)
            y_pdf_surge = stats.norm.pdf(x, mean_demand, sigma_surge)
            
            fig2 = go.Figure()
            # Base distribution
            fig2.add_trace(go.Scatter(x=x, y=y_pdf_base, mode='lines', line=dict(color='#10b981', dash="dash"), name='Base Forecast (High Uncertainty)'))
            # Surge distribution
            fig2.add_trace(go.Scatter(x=x, y=y_pdf_surge, fill='tozeroy', fillcolor='rgba(37,99,235,0.1)', mode='lines', line=dict(color='#2563eb', width=2), name='Surge Forecast (Low Uncertainty)'))
            
            fig2.add_vline(x=int(best["Q Base"]), line_dash="dash", line_color="#10b981", annotation_text="Base Order")
            fig2.add_vline(x=int(best["Q Total"]), line_dash="dash", line_color="#f59e0b", annotation_text="Total Inventory Target")
            fig2.update_layout(height=450, template="plotly_white", xaxis_title="Demand Level", yaxis_title="Probability Density", title="The Value of Postponement (Shrinking the Bell Curve)")
            st.plotly_chart(fig2, use_container_width=True)

        with t4:
            st.dataframe(df_sweep.style.format({"Service Level (%)": "{:.1f}%", "Exp. Profit (£)": "£{:,.0f}"}), use_container_width=True, hide_index=True)

else:
    # =========================================================================
    # LEARNING MODE
    # =========================================================================
    st.sidebar.info("🎓 **Learning Mode Active**\n\nThe complex controls have been hidden. Follow the tabs on the right to learn the mechanics of Dual Sourcing step-by-step.")
    st.title("🎓 Dual Sourcing Masterclass")
    
    l_price, l_salvage, l_mean, l_base_cost = 60.0, 15.0, 1000, 20.0
    l_holding_pct = 0.20 

    t1, t2, t3, t4, t5 = st.tabs(["1. The Dilemma", "2. The Hedge", "3. The Lead Time Trap 🚨", "4. The Lifecycle", "5. Strategy & Option Value"])

    with t1:
        st.subheader("Step 1: The Problem with Forecasting")
        l_volatility = st.slider("Demand Volatility (CV %)", 5, 80, 25, key="l_vol")
        l_sigma = l_mean * (l_volatility / 100.0)
        x = np.linspace(200, 1800, 600)
        y_pdf = stats.norm.pdf(x, l_mean, l_sigma)
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=x, y=y_pdf, fill='tozeroy', fillcolor='rgba(37, 99, 235, 0.2)', line=dict(color='#2563eb', width=3)))
        fig1.add_vline(x=l_mean, line_dash="dash", line_color="black", annotation_text=f"Expected: {l_mean}")
        fig1.update_layout(height=350, template="plotly_white", xaxis_title="Possible Demand", yaxis_title="Probability")
        st.plotly_chart(fig1, use_container_width=True)

    with t2:
        st.subheader("Step 2: The Cost of Being Wrong (Cu vs Co)")
        l_surge_cost = st.slider("Surge Supplier Premium (Unit Cost)", 20.5, 40.0, 22.0, step=0.5, key="l_surge")
        c1, c2 = st.columns(2)
        base_cu = l_surge_cost - l_base_cost
        base_co = (l_base_cost - l_salvage) + 1.85
        base_cr = base_cu / (base_cu + base_co)
        with c1:
            st.markdown("### 🏭 Base Supplier")
            st.metric("Cost of Under-ordering (Cu)", f"£{base_cu:.2f}")
            st.metric("Cost of Over-ordering (Co)", f"£{base_co:.2f}")
            st.markdown(f"**Target Probability:** `Cu / (Cu + Co)` = **{base_cr*100:.1f}%**")
        surge_cu = l_price - l_surge_cost
        surge_co = (l_surge_cost - l_salvage) + 0.30
        surge_cr = surge_cu / (surge_cu + surge_co)
        with c2:
            st.markdown("### ⚡ Surge Supplier")
            st.metric("Cost of Under-ordering (Cu)", f"£{surge_cu:.2f}")
            st.metric("Cost of Over-ordering (Co)", f"£{surge_co:.2f}")
            st.markdown(f"**Target Probability:** `Cu / (Cu + Co)` = **{surge_cr*100:.1f}%**")

    with t3:
        st.subheader("Step 3: The Lead Time Trap (Variance Reduction)")
        st.markdown("Why do companies care about fast shipping? Because of the **Square Root Law of Lead Time**. Predicting demand 2 weeks from now is easy. Predicting demand 12 weeks from now is guessing.")
        
        l_base_lead_time = st.slider("Base Supplier Lead Time (Weeks)", 4, 24, 12, key="l_lt_mba")
        l_surge_lead_time = 2
        
        dynamic_sigma_surge = l_sigma * math.sqrt(l_surge_lead_time / l_base_lead_time)
        dynamic_cv_surge = (dynamic_sigma_surge / l_mean) * 100
        
        col_f1, col_f2 = st.columns(2)
        col_f1.metric("Base Forecast Volatility (12 weeks out)", f"{l_volatility:.1f}%")
        col_f2.metric("Surge Forecast Volatility (2 weeks out)", f"{dynamic_cv_surge:.1f}%", delta=f"-{l_volatility - dynamic_cv_surge:.1f}% accuracy gained", delta_color="normal")
        
        x5 = np.linspace(0, 2000, 600)
        y5_base = stats.norm.pdf(x5, l_mean, l_sigma)
        y5_surge = stats.norm.pdf(x5, l_mean, dynamic_sigma_surge)
        
        fig5 = go.Figure()
        fig5.add_trace(go.Scatter(x=x5, y=y5_base, mode='lines', line=dict(color='#ef4444', width=2, dash='dash'), name='Base Forecast (Highly Uncertain)'))
        fig5.add_trace(go.Scatter(x=x5, y=y5_surge, fill='tozeroy', fillcolor='rgba(16, 185, 129, 0.2)', line=dict(color='#10b981', width=3), name='Surge Forecast (Highly Accurate)'))
        fig5.add_vline(x=l_mean, line_dash="dash", line_color="black")
        fig5.update_layout(height=350, template="plotly_white", xaxis_title="Demand", yaxis_title="Probability Density", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig5, use_container_width=True)

    with t4:
        st.subheader("Step 4: The Lifecycle Reality")
        l_scenario = st.radio("Product Lifecycle Scenario", ["End of Life (Sunset)", "FMCG (Risk of Obsolescence)", "Shelf-Stable (Ongoing)"], index=0, key="l_scen")
        nv_b = newsvendor_base(l_salvage, l_base_cost, l_surge_cost, l_mean, l_sigma, 1.85, l_scenario)
        q_b = int(nv_b['optimal_q'])
        
        c3, c4 = st.columns([1, 2])
        with c3:
            st.metric("Base Co (Overage Penalty)", f"£{nv_b['co']:.2f}")
            st.metric("Optimal Base Order", f"{int(q_b):,} units")
        with c4:
            x2 = np.linspace(200, 1800, 600)
            y2 = stats.norm.pdf(x2, l_mean, l_sigma)
            fig3 = go.Figure()
            m_base = x2 <= q_b
            fig3.add_trace(go.Scatter(x=np.concatenate([x2[m_base], [q_b, x2[m_base][0]]]), y=np.concatenate([y2[m_base], [0, 0]]), fill='toself', fillcolor='rgba(16,185,129,0.3)', line=dict(color='rgba(0,0,0,0)'), name='Base Coverage'))
            fig3.add_trace(go.Scatter(x=x2, y=y2, mode='lines', line=dict(color='#1e3a5f'), showlegend=False))
            fig3.update_layout(height=300, margin=dict(l=0,r=0,t=0,b=0), template="plotly_white")
            st.plotly_chart(fig3, use_container_width=True)

    with t5:
        st.subheader("Step 5: The Value of Postponement")
        st.markdown("If you place both orders blindly on Day 1, single-sourcing from China will almost always win mathematically. **The secret to Dual Sourcing is Option Value.** By waiting to place the Surge order until the forecast is highly accurate, you eliminate the massive overage risk. *That* is why it's profitable.")
        
        l_surge_sigma = l_sigma * math.sqrt(2 / 12)
        df_l_sweep = dual_source_sweep(l_price, l_salvage, l_mean, l_surge_sigma, l_base_cost, l_surge_cost, 1.85, 0.30, q_b, l_scenario)
        l_best = df_l_sweep.loc[df_l_sweep["Exp. Profit (£)"].idxmax()]
        
        l_base_q_only, l_base_profit = calc_standalone(l_price, l_salvage, l_base_cost, l_mean, l_sigma, 1.85, l_scenario)
        l_surge_q_only, l_surge_profit = calc_standalone(l_price, l_salvage, l_surge_cost, l_mean, l_surge_sigma, 0.30, l_scenario)
        
        sc1, sc2, sc3 = st.columns(3)
        l_best_single = max(l_base_profit, l_surge_profit)
        l_value_add = l_best['Exp. Profit (£)'] - l_best_single
        sc1.metric("Base Supplier Only", f"£{int(l_base_profit):,}")
        sc2.metric("Surge Supplier Only", f"£{int(l_surge_profit):,}")
        sc3.metric("Dual Sourcing", f"£{int(l_best['Exp. Profit (£)']):,}", delta=f"+£{int(l_value_add):,} value added" if l_value_add > 1 else "0 value added", delta_color="normal")

        fig4 = make_subplots(specs=[[{"secondary_y": True}]])
        fig4.add_trace(go.Scatter(x=df_l_sweep["Service Level (%)"], y=df_l_sweep["Exp. Profit (£)"], name="Expected Profit (£)", line=dict(color="#2563eb", width=3)), secondary_y=False)
        fig4.add_vline(x=l_best["Service Level (%)"], line_dash="dash", line_color="red", annotation_text=f"Max Profit at {l_best['Service Level (%)']:.1f}%")
        fig4.update_layout(height=350, template="plotly_white", xaxis_title="Target Service Level (%)", yaxis_title="Expected Profit (£)")
        st.plotly_chart(fig4, use_container_width=True)
