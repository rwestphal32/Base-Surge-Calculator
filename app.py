import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math

st.set_page_config(page_title="Dual Sourcing Optimizer", layout="wide")

# ─────────────────────────────────────────────
# CORE NEWSVENDOR MATH (TLC & Variance Reduction)
# ─────────────────────────────────────────────
def expected_metrics(q, mu, sigma):
    if sigma <= 0: return float(min(q, mu)), float(max(0, q - mu)), float(max(0, mu - q))
    z     = (q - mu) / sigma
    phi_z = stats.norm.pdf(z)
    Phi_z = stats.norm.cdf(z)
    loss  = phi_z - z * (1 - Phi_z)
    exp_sales    = float(np.clip(mu - sigma * loss, 0, q))
    exp_leftover = float(max(0.0, q - exp_sales))
    exp_stockout = float(max(0.0, mu - exp_sales))
    return exp_sales, exp_leftover, exp_stockout

def calculate_overage_cost(unit_cost, salvage, pipeline_holding, scenario):
    # Overage penalty = Pipeline capital cost + scenario-specific markdown risk
    if scenario == "End of Life (Sunset)": return (unit_cost - salvage) + pipeline_holding
    elif scenario == "FMCG (Risk of Obsolescence)": return ((unit_cost - salvage) * 0.5) + pipeline_holding
    else: return pipeline_holding # Shelf-Stable

def calc_standalone(price, salvage, unit_cost, tlc, mu, sigma, pipeline_holding, scenario):
    cu = price - tlc
    co = calculate_overage_cost(unit_cost, salvage, pipeline_holding, scenario)
    if (cu + co) <= 0: return 0, 0
    cr = cu / (cu + co)
    z = stats.norm.ppf(cr)
    q = max(0.0, mu + z * sigma)
    exp_sales, exp_leftover, exp_stockout = expected_metrics(q, mu, sigma)
    
    revenue = price * exp_sales
    if scenario == "End of Life (Sunset)": salvage_rev = salvage * exp_leftover
    elif scenario == "FMCG (Risk of Obsolescence)": salvage_rev = (salvage * 0.5 * exp_leftover) + (unit_cost * 0.5 * exp_leftover)
    else: salvage_rev = unit_cost * exp_leftover
    
    spend = tlc * q # Spend includes pipeline WACC
    exp_profit = revenue + salvage_rev - spend
    return q, exp_profit

def newsvendor_base(salvage, base_cost, base_tlc, surge_tlc, mu, sigma_base, base_hold, scenario):
    cu = surge_tlc - base_tlc # True premium factors in working capital savings
    co = calculate_overage_cost(base_cost, salvage, base_hold, scenario)
    if (cu + co) <= 0: return None
    cr = cu / (cu + co)
    z  = stats.norm.ppf(cr)
    q  = max(0.0, mu + z * sigma_base)
    return dict(cu=cu, co=co, critical_ratio=cr, z_score=z, optimal_q=q)

def newsvendor_surge(price, salvage, surge_cost, surge_tlc, mu, sigma_surge, surge_hold, scenario):
    cu = price - surge_tlc
    co = calculate_overage_cost(surge_cost, salvage, surge_hold, scenario)
    if (cu + co) <= 0: return None
    cr = cu / (cu + co)
    z  = stats.norm.ppf(cr)
    q  = max(0.0, mu + z * sigma_surge)
    return dict(cu=cu, co=co, critical_ratio=cr, z_score=z, optimal_q=q)

def dual_source_sweep(price, salvage, mu, sigma_surge, base_cost, base_tlc, surge_cost, surge_tlc, q_base_fixed, scenario):
    results = []
    exp_sales_base_only = expected_metrics(q_base_fixed, mu, sigma_surge)[0]

    for pct in np.arange(0.50, 0.999, 0.005):
        q_target = mu + stats.norm.ppf(pct) * sigma_surge
        q_surge = max(0.0, q_target - q_base_fixed)
        q_total = q_base_fixed + q_surge

        exp_sales, exp_leftover, exp_stockout = expected_metrics(q_total, mu, sigma_surge)
        exp_surge_usage = exp_sales - exp_sales_base_only

        base_frac = q_base_fixed / q_total if q_total > 0 else 1.0
        base_leftover = exp_leftover * base_frac

        revenue = price * exp_sales
        if scenario == "End of Life (Sunset)": salvage_rev = salvage * exp_leftover
        elif scenario == "FMCG (Risk of Obsolescence)": salvage_rev = (salvage * 0.5 * exp_leftover) + (base_cost * 0.5 * exp_leftover)
        else: salvage_rev = base_cost * exp_leftover
        
        base_spend = base_tlc * q_base_fixed
        surge_spend = surge_tlc * exp_surge_usage 
        
        exp_profit = revenue + salvage_rev - base_spend - surge_spend

        results.append({
            "Service Level (%)": round(pct * 100, 1), "Q Base": int(q_base_fixed), "Q Surge": int(q_surge),
            "Q Total Target": int(q_total), "Expected Surge Usage": exp_surge_usage,
            "Base Spend (£)": round(base_spend), "Expected Surge Spend (£)": round(surge_spend),
            "Exp. Leftover (units)": round(exp_leftover), "Exp. Stockout (units)": round(exp_stockout), 
            "Exp. Profit (£)": round(exp_profit), "Exp. Sales Total": exp_sales
        })
    return pd.DataFrame(results)

# ─────────────────────────────────────────────
# APP MODE SELECTOR & DASHBOARDS
# ─────────────────────────────────────────────
app_mode = st.sidebar.radio("Select App Mode", ["🎓 Learning Mode (Concepts)", "🚀 Pro Mode (Dashboard)"])
st.sidebar.markdown("---")

if app_mode == "🚀 Pro Mode (Dashboard)":
    st.title("⚖️ Dual Sourcing: Operational Optimizer")

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
        
        st.markdown("---")
        st.header("💰 Cost of Capital")
        wacc = st.slider("Annual WACC (%)", 10, 40, 20) / 100.0
        
        # TOTAL LANDED COST (TLC) MATH
        base_hold = ((base_cost * wacc) / 52.0) * base_lead_time
        surge_hold = ((surge_cost * wacc) / 52.0) * surge_lead_time
        base_tlc = base_cost + base_hold
        surge_tlc = surge_cost + surge_hold
        
        # VARIANCE REDUCTION
        sigma_surge = sigma_base * math.sqrt(surge_lead_time / base_lead_time)
        
        run = st.button("🚀 Run Optimizer", type="primary", use_container_width=True)

    if run:
        base_q_only, base_profit_only = calc_standalone(selling_price, salvage_value, base_cost, base_tlc, mean_demand, sigma_base, base_hold, scenario)
        surge_q_only, surge_profit_only = calc_standalone(selling_price, salvage_value, surge_cost, surge_tlc, mean_demand, sigma_surge, surge_hold, scenario)

        nv_base = newsvendor_base(salvage_value, base_cost, base_tlc, surge_tlc, mean_demand, sigma_base, base_hold, scenario)
        nv_surge = newsvendor_surge(selling_price, salvage_value, surge_cost, surge_tlc, mean_demand, sigma_surge, surge_hold, scenario)
        
        # LOGIC GATE: Is the TLC Premium higher than the TLC Overage Risk?
        true_premium = surge_tlc - base_tlc
        base_overage_penalty = calculate_overage_cost(base_cost, salvage_value, base_hold, scenario)
        
        if true_premium >= base_overage_penalty:
            q_base_fixed = max(float(base_moq), base_q_only)
        else:
            q_base_fixed = max(float(base_moq), nv_base["optimal_q"])
            
        df_sweep = dual_source_sweep(selling_price, salvage_value, mean_demand, sigma_surge, base_cost, base_tlc, surge_cost, surge_tlc, q_base_fixed, scenario)
        best = df_sweep.loc[df_sweep["Exp. Profit (£)"].idxmax()]

        st.subheader(f"🎯 Optimal Split: {scenario}")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Target Service Level", f"{best['Service Level (%)']:.1f}%")
        k2.metric("Base Order", f"{int(best['Q Base']):,}")
        k3.metric("Surge Capacity Target", f"{int(best['Q Surge']):,}", help=f"Expected actual usage: {int(best['Expected Surge Usage'])} units.")
        k4.metric("Optimal Expected Profit", f"£{int(best['Exp. Profit (£)']):,}")

        t1, t2, t3 = st.tabs(["📈 Financials & Strategy", "⚙️ TLC & Mechanics", "📐 Demand Distribution"])
        
        with t1:
            if true_premium >= base_overage_penalty:
                st.warning("⚠️ **Surge Premium exceeds Base Overage Penalty.** The Base Order has been set to the Single-Source maximum. The Surge supplier acts purely as catastrophic stockout insurance.")
            
            st.markdown("#### Strategy Comparison: Expected Profit")
            sc1, sc2, sc3 = st.columns(3)
            best_single = max(base_profit_only, surge_profit_only)
            value_add = best['Exp. Profit (£)'] - base_profit_only
            sc1.metric("Base Supplier Only", f"£{int(base_profit_only):,}", help=f"Order: {int(base_q_only)}")
            sc2.metric("Surge Supplier Only", f"£{int(surge_profit_only):,}", help=f"Order: {int(surge_q_only)}")
            sc3.metric("Dual Sourcing (Optimal)", f"£{int(best['Exp. Profit (£)']):,}", delta=f"+£{int(value_add):,} vs Base Only", delta_color="normal")

            st.markdown("---")
            st.markdown("#### Bridging the Gap: Where the Value Comes From")
            c_f1, c_f2, c_f3 = st.columns(3)
            
            if int(best["Q Surge"]) == 0:
                surge_profit_captured, base_optimization = 0, 0
            else:
                surge_profit_captured = best["Expected Surge Usage"] * (selling_price - surge_tlc)
                base_optimization = value_add - surge_profit_captured
            
            c_f1.metric("Surge Profit Captured", f"£{int(surge_profit_captured):,}", help="Margin captured from sales that would have stocked out.")
            c_f2.metric("Base Optimization", f"£{int(base_optimization):,}", help="Savings from reduced overage risk minus margin sacrificed to Surge.")
            c_f3.metric("Total Strategy Value Add", f"£{int(value_add):,}")

            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Scatter(x=df_sweep["Service Level (%)"], y=df_sweep["Exp. Profit (£)"], name="Expected Profit", line=dict(color="#2563eb", width=3)), secondary_y=False)
            fig.add_vline(x=best["Service Level (%)"], line_dash="dash", line_color="red", annotation_text="Optimum Profit")
            fig.update_layout(height=350, template="plotly_white", title="Profit vs Target Service Level")
            st.plotly_chart(fig, use_container_width=True)

        with t2:
            st.markdown("### Total Landed Cost (TLC) Accounting")
            st.markdown("Comparing invoice prices is misleading. Surge suppliers subsidize their premium by freeing up working capital earlier.")
            
            t_c1, t_c2 = st.columns(2)
            t_c1.metric("Base TLC", f"£{base_tlc:.2f}", help=f"£{base_cost} invoice + £{base_hold:.2f} WACC over {base_lead_time} wks")
            t_c2.metric("Surge TLC", f"£{surge_tlc:.2f}", help=f"£{surge_cost} invoice + £{surge_hold:.2f} WACC over {surge_lead_time} wks")
            
            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                with st.container(border=True):
                    st.markdown("### 🏭 Base Supplier")
                    st.markdown(f"- **$C_u$:** Surge TLC - Base TLC = **£{nv_base['cu']:.2f}**/unit")
                    st.markdown(f"- **$C_o$:** Scenario Penalty + Base WACC = **£{nv_base['co']:.2f}**/unit")
                    st.markdown(f"**Critical Ratio ($CR_1$):** `Cu / (Cu + Co)` = **{nv_base['critical_ratio']:.3f}**")
                    st.markdown(f"**Optimal Base $Q_1$:** **{int(nv_base['optimal_q']):,} units**")
            with col2:
                with st.container(border=True):
                    st.markdown("### ⚡ Surge Supplier")
                    st.markdown(f"- **$C_u$:** Price - Surge TLC = **£{nv_surge['cu']:.2f}**/unit")
                    st.markdown(f"- **$C_o$:** Scenario Penalty + Surge WACC = **£{nv_surge['co']:.2f}**/unit")
                    st.markdown(f"**Critical Ratio ($CR_2$):** `Cu / (Cu + Co)` = **{nv_surge['critical_ratio']:.3f}**")
                    st.markdown(f"**Total Optimal Target $Q_2$:** **{int(nv_surge['optimal_q']):,} units**")

        with t3:
            x = np.linspace(mean_demand - 4*sigma_base, mean_demand + 4*sigma_base, 600)
            y_pdf_base = stats.norm.pdf(x, mean_demand, sigma_base)
            y_pdf_surge = stats.norm.pdf(x, mean_demand, sigma_surge)
            
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=x, y=y_pdf_base, mode='lines', line=dict(color='#10b981', dash="dash"), name='Base Forecast (High Uncertainty)'))
            fig2.add_trace(go.Scatter(x=x, y=y_pdf_surge, fill='tozeroy', fillcolor='rgba(37,99,235,0.1)', mode='lines', line=dict(color='#2563eb', width=2), name='Surge Forecast (Low Uncertainty)'))
            fig2.add_vline(x=int(best["Q Base"]), line_dash="dash", line_color="#10b981", annotation_text="Base Order")
            fig2.add_vline(x=int(best["Q Total Target"]), line_dash="dash", line_color="#f59e0b", annotation_text="Total Target")
            fig2.update_layout(height=450, template="plotly_white", xaxis_title="Demand Level", yaxis_title="Probability Density")
            st.plotly_chart(fig2, use_container_width=True)

else:
    # LEARNING MODE (Truncated slightly to save character limits, follows same TLC logic)
    st.sidebar.info("🎓 **Learning Mode Active**")
    st.title("🎓 Dual Sourcing Masterclass")
    
    l_price, l_salvage, l_mean, l_base_cost = 60.0, 15.0, 1000, 20.0
    l_wacc = 0.20 

    t1, t2, t3, t4, t5 = st.tabs(["1. The Dilemma", "2. Total Landed Cost 🚨", "3. The Hedge", "4. Lead Time & Variance", "5. The Bottom Line"])

    with t1:
        st.subheader("Step 1: The Problem with Forecasting")
        l_volatility = st.slider("Demand Volatility (CV %)", 5, 80, 25, key="l_vol")
        l_sigma = l_mean * (l_volatility / 100.0)
        x = np.linspace(200, 1800, 600)
        y_pdf = stats.norm.pdf(x, l_mean, l_sigma)
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=x, y=y_pdf, fill='tozeroy', fillcolor='rgba(37, 99, 235, 0.2)', line=dict(color='#2563eb', width=3)))
        fig1.add_vline(x=l_mean, line_dash="dash", line_color="black")
        fig1.update_layout(height=350, template="plotly_white")
        st.plotly_chart(fig1, use_container_width=True)

    with t2:
        st.subheader("Step 2: Total Landed Cost (The Working Capital Secret)")
        st.markdown("Comparing invoice prices is misleading. A fast supplier subsidizes their higher unit cost by freeing up your working capital earlier.")
        
        c_tlc1, c_tlc2 = st.columns(2)
        l_surge_cost = c_tlc1.slider("Surge Supplier Invoice Price", 20.5, 40.0, 22.0, step=0.5)
        l_base_lt = c_tlc2.slider("Base Lead Time (Weeks)", 4, 24, 12)
        
        l_base_hold = ((l_base_cost * l_wacc) / 52) * l_base_lt
        l_surge_hold = ((l_surge_cost * l_wacc) / 52) * 2 # Fixed 2 wk surge
        
        l_base_tlc = l_base_cost + l_base_hold
        l_surge_tlc = l_surge_cost + l_surge_hold
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Base TLC", f"£{l_base_tlc:.2f}", help=f"£20 + £{l_base_hold:.2f} WACC")
        m2.metric("Surge TLC", f"£{l_surge_tlc:.2f}", help=f"£{l_surge_cost} + £{l_surge_hold:.2f} WACC")
        m3.metric("True Surge Premium", f"£{l_surge_tlc - l_base_tlc:.2f}")

    with t3:
        st.subheader("Step 3: The Cost of Being Wrong (Cu vs Co)")
        l_scenario = st.radio("Product Lifecycle Scenario", ["End of Life (Sunset)", "FMCG (Risk of Obsolescence)", "Shelf-Stable (Ongoing)"], index=0, key="l_scen")
        
        c1, c2 = st.columns(2)
        base_cu = l_surge_tlc - l_base_tlc
        base_co = calculate_overage_cost(l_base_cost, l_salvage, l_base_hold, l_scenario)
        base_cr = base_cu / (base_cu + base_co) if (base_cu+base_co) > 0 else 0
        with c1:
            st.markdown("### 🏭 Base Supplier")
            st.metric("Cost of Under-ordering (Cu)", f"£{base_cu:.2f}")
            st.metric("Cost of Over-ordering (Co)", f"£{base_co:.2f}")
            st.markdown(f"**Target Probability:** `Cu / (Cu + Co)` = **{base_cr*100:.1f}%**")
            
        surge_cu = l_price - l_surge_tlc
        surge_co = calculate_overage_cost(l_surge_cost, l_salvage, l_surge_hold, l_scenario)
        surge_cr = surge_cu / (surge_cu + surge_co)
        with c2:
            st.markdown("### ⚡ Surge Supplier")
            st.metric("Cost of Under-ordering (Cu)", f"£{surge_cu:.2f}")
            st.metric("Cost of Over-ordering (Co)", f"£{surge_co:.2f}")
            st.markdown(f"**Target Probability:** `Cu / (Cu + Co)` = **{surge_cr*100:.1f}%**")

    with t4:
        st.subheader("Step 4: Variance Reduction")
        st.markdown("Because you place the Surge order closer to the selling season, your forecast becomes highly accurate.")
        dynamic_sigma_surge = l_sigma * math.sqrt(2 / l_base_lt)
        
        x5 = np.linspace(0, 2000, 600)
        y5_base = stats.norm.pdf(x5, l_mean, l_sigma)
        y5_surge = stats.norm.pdf(x5, l_mean, dynamic_sigma_surge)
        
        fig5 = go.Figure()
        fig5.add_trace(go.Scatter(x=x5, y=y5_base, mode='lines', line=dict(color='#ef4444', width=2, dash='dash'), name='Base Forecast (Highly Uncertain)'))
        fig5.add_trace(go.Scatter(x=x5, y=y5_surge, fill='tozeroy', fillcolor='rgba(16, 185, 129, 0.2)', line=dict(color='#10b981', width=3), name='Surge Forecast (Highly Accurate)'))
        fig5.add_vline(x=l_mean, line_dash="dash", line_color="black")
        fig5.update_layout(height=300, template="plotly_white")
        st.plotly_chart(fig5, use_container_width=True)

    with t5:
        st.subheader("Step 5: Bridging the Profit Gap")
        
        nv_b = newsvendor_base(l_salvage, l_base_cost, l_base_tlc, l_surge_tlc, l_mean, l_sigma, l_base_hold, l_scenario)
        l_base_q_only, l_base_profit = calc_standalone(l_price, l_salvage, l_base_cost, l_base_tlc, l_mean, l_sigma, l_base_hold, l_scenario)
        l_surge_q_only, l_surge_profit = calc_standalone(l_price, l_salvage, l_surge_cost, l_surge_tlc, l_mean, dynamic_sigma_surge, l_surge_hold, l_scenario)
        
        q_b = l_base_q_only if base_cu >= base_co else int(nv_b['optimal_q'])
        
        df_l_sweep = dual_source_sweep(l_price, l_salvage, l_mean, dynamic_sigma_surge, l_base_cost, l_base_tlc, l_surge_cost, l_surge_tlc, q_b, l_scenario)
        l_best = df_l_sweep.loc[df_l_sweep["Exp. Profit (£)"].idxmax()]
        
        sc1, sc2, sc3 = st.columns(3)
        l_best_single = max(l_base_profit, l_surge_profit)
        l_value_add = l_best['Exp. Profit (£)'] - l_base_profit
        sc1.metric("Base Supplier Only", f"£{int(l_base_profit):,}")
        sc2.metric("Surge Supplier Only", f"£{int(l_surge_profit):,}")
        sc3.metric("Dual Sourcing", f"£{int(l_best['Exp. Profit (£)']):,}", delta=f"+£{int(l_value_add):,} value added")

        fig4 = make_subplots(specs=[[{"secondary_y": True}]])
        fig4.add_trace(go.Scatter(x=df_l_sweep["Service Level (%)"], y=df_l_sweep["Exp. Profit (£)"], name="Expected Profit (£)", line=dict(color="#2563eb", width=3)), secondary_y=False)
        fig4.add_vline(x=l_best["Service Level (%)"], line_dash="dash", line_color="red", annotation_text=f"Max Profit at {l_best['Service Level (%)']:.1f}%")
        fig4.update_layout(height=350, template="plotly_white")
        st.plotly_chart(fig4, use_container_width=True)
