import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Dual Sourcing Optimizer", layout="wide")

# ─────────────────────────────────────────────
# CORE NEWSVENDOR MATH (Shared by both modes)
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
    if (cu + co) <= 0: return None
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
    if (cu + co) <= 0: return None
    cr = cu / (cu + co)
    z  = stats.norm.ppf(cr)
    q  = max(0.0, mu + z * sigma)
    exp_sales, exp_leftover, exp_stockout = expected_metrics(q, mu, sigma)
    exp_profit = (price * exp_sales) + (salvage * exp_leftover) - (surge_cost * q) - (holding_per_period * exp_leftover)
    return dict(cu=cu, co=co, critical_ratio=cr, z_score=z, optimal_q=q,
                exp_sales=exp_sales, exp_leftover=exp_leftover, exp_stockout=exp_stockout, exp_profit=exp_profit)

def dual_source_sweep(price, salvage, mu, sigma, base_cost, surge_cost, base_moq, surge_moq, holding_per_period, q_base_fixed, scenario):
    results = []
    surge_holding_per_period = (surge_cost / base_cost) * holding_per_period
    for pct in np.arange(0.50, 0.999, 0.005):
        q_target = mu + stats.norm.ppf(pct) * sigma
        q_surge_raw = max(0.0, q_target - q_base_fixed)
        q_surge = max(float(surge_moq), q_surge_raw) if q_surge_raw >= surge_moq else 0.0
        q_total = q_base_fixed + q_surge

        exp_sales, exp_leftover, exp_stockout = expected_metrics(q_total, mu, sigma)
        base_frac = q_base_fixed / q_total if q_total > 0 else 1.0
        base_leftover = exp_leftover * base_frac
        surge_leftover = exp_leftover * (1.0 - base_frac)

        revenue = price * exp_sales
        if scenario == "End of Life (Sunset)": salvage_rev = salvage * exp_leftover
        elif scenario == "FMCG (Risk of Obsolescence)": salvage_rev = (salvage * 0.5 * exp_leftover) + (base_cost * 0.5 * exp_leftover)
        else: salvage_rev = base_cost * exp_leftover
        
        base_spend = base_cost * q_base_fixed
        surge_spend = surge_cost * q_surge
        hold_cost = (holding_per_period * base_leftover) + (surge_holding_per_period * surge_leftover)
        exp_profit = revenue + salvage_rev - base_spend - surge_spend - hold_cost

        results.append({
            "Service Level (%)": round(pct * 100, 1), "Q Base": int(q_base_fixed), "Q Surge": int(q_surge),
            "Q Total": int(q_total), "Surge %": round(q_surge / q_total * 100, 1) if q_total > 0 else 0,
            "Base Spend (£)": round(base_spend), "Surge Spend (£)": round(surge_spend),
            "Holding Cost (£)": round(hold_cost), "Exp. Leftover (units)": round(exp_leftover),
            "Exp. Stockout (units)": round(exp_stockout), "Exp. Profit (£)": round(exp_profit),
        })
    return pd.DataFrame(results)


# ─────────────────────────────────────────────
# APP MODE SELECTOR
# ─────────────────────────────────────────────
app_mode = st.sidebar.radio("Select App Mode", ["🎓 Learning Mode (Concepts)", "🚀 Pro Mode (Dashboard)"])
st.sidebar.markdown("---")

if app_mode == "🚀 Pro Mode (Dashboard)":
    # =========================================================================
    # PRO MODE (The Full Operational Dashboard)
    # =========================================================================
    st.title("⚖️ Dual Sourcing: Operational Optimizer")
    st.markdown("Configure your supplier economics in the sidebar to generate your optimal procurement split.")

    with st.sidebar:
        st.header("📦 Product Economics")
        selling_price = st.number_input("Selling Price (£)", value=60.0, step=1.0)
        salvage_value = st.number_input("Salvage / Markdown Value (£)", value=15.0, step=1.0)
        scenario = st.radio("Lifecycle Scenario", ["Shelf-Stable (Ongoing)", "FMCG (Risk of Obsolescence)", "End of Life (Sunset)"], index=1)
        
        st.markdown("---")
        st.header("📈 Demand Profile")
        mean_demand = st.number_input("Mean Seasonal/Period Demand", value=1000, step=50)
        volatility_pct = st.slider("Demand Volatility (CV %)", 5, 80, 25)
        sigma = mean_demand * (volatility_pct / 100.0)
        
        st.markdown("---")
        st.header("🏭 Base Supplier (Cheap, Slow)")
        base_cost = st.number_input("Unit Cost — Base (£)", value=20.0, step=0.5)
        base_lead_time = st.number_input("Lead Time (weeks)", value=12, step=1)
        base_moq = st.number_input("MOQ — Base (units)", value=500, step=100)
        
        st.markdown("---")
        st.header("⚡ Surge Supplier (Expensive, Fast)")
        surge_cost = st.number_input("Unit Cost — Surge (£)", value=22.0, step=0.5)
        surge_moq = st.number_input("MOQ — Surge (units)", value=100, step=50)
        
        st.markdown("---")
        st.header("💰 Holding Cost")
        holding_cost_pct = st.slider("Annual Holding Cost / WACC (%)", 10, 40, 20) / 100.0
        holding_per_period = ((base_cost * holding_cost_pct) / 52.0) * base_lead_time
        
        run = st.button("🚀 Run Optimizer", type="primary", use_container_width=True)

    if run:
        nv_base  = newsvendor_base(salvage_value, base_cost, surge_cost, mean_demand, sigma, holding_per_period, scenario)
        nv_surge = newsvendor_surge(selling_price, salvage_value, base_cost, surge_cost, mean_demand, sigma, holding_per_period, scenario)
        q_base_fixed = max(float(base_moq), nv_base["optimal_q"])
        df_sweep = dual_source_sweep(selling_price, salvage_value, mean_demand, sigma, base_cost, surge_cost, base_moq, surge_moq, holding_per_period, q_base_fixed, scenario)
        best = df_sweep.loc[df_sweep["Exp. Profit (£)"].idxmax()]

        st.subheader(f"🎯 Optimal Split: {scenario}")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Target Service Level", f"{best['Service Level (%)']:.1f}%")
        k2.metric("Base Order", f"{int(best['Q Base']):,}")
        k3.metric("Surge Order", f"{int(best['Q Surge']):,}")
        k4.metric("Expected Profit", f"£{int(best['Exp. Profit (£)']):,}")

        t1, t2, t3, t4 = st.tabs(["📈 Financials & Profit Curve", "⚙️ Newsvendor Mechanics", "📐 Demand Distribution", "📋 Raw Data Sweep"])
        
        with t1:
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
                
            st.markdown("#### Financial Value of the Hedge")
            c_f1, c_f2, c_f3 = st.columns(3)
            surge_premium_paid = int(best["Q Surge"]) * (surge_cost - base_cost)
            unhedged_exposure = int(nv_base["exp_stockout"]) * (surge_cost - base_cost)
            c_f1.metric("Surge Premium Paid", f"£{surge_premium_paid:,}", help="Incremental cost paid to Surge supplier over Base cost.")
            c_f2.metric("Unhedged Surge Exposure", f"£{unhedged_exposure:,}", help="What you'd pay in emergency surge premiums if you relied ONLY on the base order.")
            c_f3.metric("Net Benefit of Pre-Ordering Surge", f"£{unhedged_exposure - surge_premium_paid:,}")

        with t2:
            st.markdown("The underlying math calculates the optimal service level (Critical Ratio) by balancing the **Cost of Under-ordering ($C_u$)** against the **Cost of Over-ordering ($C_o$)**.")
            
            col1, col2 = st.columns(2)
            with col1:
                with st.container(border=True):
                    st.markdown("### 🏭 Base Supplier")
                    st.markdown(f"- **Cost of Under-ordering ($C_u$):** Surge Cost (£{surge_cost}) - Base Cost (£{base_cost}) = **£{nv_base['cu']:.2f}**/unit")
                    st.markdown(f"- **Cost of Over-ordering ($C_o$):** Scenario Penalty + Holding = **£{nv_base['co']:.2f}**/unit")
                    st.markdown("---")
                    st.markdown(f"**Critical Ratio ($CR_1$):** `Cu / (Cu + Co)` = **{nv_base['critical_ratio']:.3f}**")
                    st.markdown(f"**Target Z-Score:** **{nv_base['z_score']:.3f}** std deviations")
                    st.markdown(f"**Optimal Base $Q_1$:** **{int(nv_base['optimal_q']):,} units**")
            with col2:
                with st.container(border=True):
                    st.markdown("### ⚡ Surge Supplier")
                    st.markdown(f"- **Cost of Under-ordering ($C_u$):** Price (£{selling_price}) - Surge Cost (£{surge_cost}) = **£{nv_surge['cu']:.2f}**/unit")
                    st.markdown(f"- **Cost of Over-ordering ($C_o$):** Scenario Penalty + Holding = **£{nv_surge['co']:.2f}**/unit")
                    st.markdown("---")
                    st.markdown(f"**Critical Ratio ($CR_2$):** `Cu / (Cu + Co)` = **{nv_surge['critical_ratio']:.3f}**")
                    st.markdown(f"**Target Z-Score:** **{nv_surge['z_score']:.3f}** std deviations")
                    st.markdown(f"**Total Optimal $Q_2$:** **{int(nv_surge['optimal_q']):,} units**")

        with t3:
            x = np.linspace(mean_demand - 4*sigma, mean_demand + 4*sigma, 600)
            y_pdf = stats.norm.pdf(x, mean_demand, sigma)
            fig2 = go.Figure()
            m_base = x <= int(best["Q Base"])
            m_surge = (x > int(best["Q Base"])) & (x <= int(best["Q Total"]))
            m_stock = x > int(best["Q Total"])
            
            fig2.add_trace(go.Scatter(x=np.concatenate([x[m_base], [int(best["Q Base"]), x[m_base][0]]]), y=np.concatenate([y_pdf[m_base], [0, 0]]), fill='toself', fillcolor='rgba(16,185,129,0.2)', line=dict(color='rgba(0,0,0,0)'), name='Base Coverage'))
            if m_surge.any():
                fig2.add_trace(go.Scatter(x=np.concatenate([[int(best["Q Base"])], x[m_surge], [int(best["Q Total"]), int(best["Q Base"])]]), y=np.concatenate([[0], y_pdf[m_surge], [0, 0]]), fill='toself', fillcolor='rgba(245,158,11,0.25)', line=dict(color='rgba(0,0,0,0)'), name='Surge Coverage'))
            if m_stock.any():
                fig2.add_trace(go.Scatter(x=np.concatenate([[int(best["Q Total"])], x[m_stock], [x[m_stock][-1], int(best["Q Total"])]]), y=np.concatenate([[0], y_pdf[m_stock], [0, 0]]), fill='toself', fillcolor='rgba(239,68,68,0.25)', line=dict(color='rgba(0,0,0,0)'), name='Stockout Risk'))
            
            fig2.add_trace(go.Scatter(x=x, y=y_pdf, mode='lines', line=dict(color='#1e3a5f'), name='Demand Profile'))
            fig2.add_vline(x=int(best["Q Base"]), line_dash="dash", line_color="#10b981", annotation_text="Base Order")
            fig2.add_vline(x=int(best["Q Total"]), line_dash="dash", line_color="#f59e0b", annotation_text="Total Inventory")
            fig2.update_layout(height=450, template="plotly_white", xaxis_title="Demand Level", yaxis_title="Probability")
            st.plotly_chart(fig2, use_container_width=True)

        with t4:
            st.dataframe(df_sweep.style.format({"Service Level (%)": "{:.1f}%", "Exp. Profit (£)": "£{:,.0f}"}), use_container_width=True, hide_index=True)

else:
    # =========================================================================
    # LEARNING MODE (The Guided Masterclass)
    # =========================================================================
    st.sidebar.info("🎓 **Learning Mode Active**\n\nThe complex controls have been hidden. Follow the tabs on the right to learn the mechanics of Dual Sourcing step-by-step.")
    
    st.title("🎓 Dual Sourcing Masterclass")
    st.markdown("Why do companies use two suppliers? Because predicting the future is impossible. We use a **cheap, slow** supplier for what we *expect* to happen, and an **expensive, fast** supplier to protect us against *surprises*.")

    # Fixed Baseline Parameters for the lesson
    l_price, l_salvage, l_mean, l_base_cost = 60.0, 15.0, 1000, 20.0
    l_holding_pct = 0.20 # 20% WACC

    t1, t2, t3, t4, t5 = st.tabs(["1. The Dilemma", "2. The Hedge (Cu vs Co)", "3. The Lead Time Trap 🚨", "4. The Lifecycle", "5. The Bottom Line"])

    with t1:
        st.subheader("Step 1: The Problem with Forecasting")
        st.markdown("If you knew exactly how many units you would sell, you would order 100% of them from your cheapest factory in China. But demand is volatile. Adjust the slider below to see how uncertainty creates the need for a backup plan.")
        
        l_volatility = st.slider("Demand Volatility (CV %)", 5, 80, 25, key="l_vol")
        l_sigma = l_mean * (l_volatility / 100.0)
        
        x = np.linspace(200, 1800, 600)
        y_pdf = stats.norm.pdf(x, l_mean, l_sigma)
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=x, y=y_pdf, fill='tozeroy', fillcolor='rgba(37, 99, 235, 0.2)', line=dict(color='#2563eb', width=3)))
        fig1.add_vline(x=l_mean, line_dash="dash", line_color="black", annotation_text=f"Expected: {l_mean}")
        fig1.update_layout(height=350, template="plotly_white", xaxis_title="Possible Demand", yaxis_title="Probability")
        st.plotly_chart(fig1, use_container_width=True)
        st.info("💡 **Takeaway:** The wider the curve, the riskier a single large order becomes. We need a second supplier to cover the long right tail.")

    with t2:
        st.subheader("Step 2: The Cost of Being Wrong (Cu vs Co)")
        st.markdown("How much of that 'expected' demand should we commit to the Base supplier? We decide using the Newsvendor model, which weighs the **Cost of Under-ordering ($C_u$)** against the **Cost of Over-ordering ($C_o$)**.")
        
        l_surge_cost = st.slider("Surge Supplier Premium (Unit Cost)", 20.5, 40.0, 22.0, step=0.5, key="l_surge")
        st.markdown(f"*Base Supplier Cost is fixed at £{l_base_cost:.2f}. Selling Price is £{l_price:.2f}.*")

        c1, c2 = st.columns(2)
        base_cu = l_surge_cost - l_base_cost
        base_co = (l_base_cost - l_salvage) + 1.85
        base_cr = base_cu / (base_cu + base_co)
        with c1:
            st.markdown("### 🏭 Base Supplier")
            st.metric("Cost of Under-ordering (Cu)", f"£{base_cu:.2f}", help="If you order too little from Base, you don't lose the sale! You just pay the Surge premium.")
            st.metric("Cost of Over-ordering (Co)", f"£{base_co:.2f}", help="Markdown loss + Holding cost")
            st.markdown(f"**Target Probability:** `Cu / (Cu + Co)` = **{base_cr*100:.1f}%**")
        
        surge_cu = l_price - l_surge_cost
        surge_co = (l_surge_cost - l_salvage) + 1.85
        surge_cr = surge_cu / (surge_cu + surge_co)
        with c2:
            st.markdown("### ⚡ Surge Supplier")
            st.metric("Cost of Under-ordering (Cu)", f"£{surge_cu:.2f}", help="If you order too little from Surge, you stock out and lose the entire profit margin.")
            st.metric("Cost of Over-ordering (Co)", f"£{surge_co:.2f}")
            st.markdown(f"**Target Probability:** `Cu / (Cu + Co)` = **{surge_cr*100:.1f}%**")

        st.info("💡 **Takeaway:** The formula `Cu / (Cu + Co)` is called the Critical Ratio. It dictates your Service Level. Because the penalty for under-ordering from Base is *only the surge premium*, the ratio is low, keeping your initial base order intentionally conservative.")

    with t3:
        st.subheader("Step 3: The Lead Time Trap (Working Capital & Volatility)")
        st.markdown("Why do companies care about fast shipping? Because **Lead Time equals Uncertainty**. Predicting demand 2 weeks from now is easy. Predicting demand 16 weeks from now is basically guessing. Long lead times do two destructive things: they widen your forecast error, and they trap working capital in transit pipelines.")
        
        l_lead_time = st.slider("Base Supplier Lead Time (Weeks)", 2, 24, 12, key="l_lt_mba")
        
        # Financial Math based on Little's Law
        weekly_demand = l_mean / 12.0 # Assuming 1000 units over a 12-week season/quarter
        pipeline_units = weekly_demand * l_lead_time
        working_capital = pipeline_units * l_base_cost
        cost_of_capital = working_capital * (l_holding_pct * (l_lead_time/52.0))
        
        dynamic_cv = l_lead_time * 2.5  
        dynamic_sigma = l_mean * (dynamic_cv / 100.0)
        
        col_f1, col_f2, col_f3 = st.columns(3)
        col_f1.metric("Resulting Volatility (CV)", f"{dynamic_cv:.1f}%")
        col_f2.metric("Working Capital Tied Up", f"£{int(working_capital):,}")
        col_f3.metric("Opportunity Cost of Capital", f"£{int(cost_of_capital):,}")
        
        x5 = np.linspace(0, 2000, 600)
        y5 = stats.norm.pdf(x5, l_mean, dynamic_sigma)
        fig5 = go.Figure()
        fig5.add_trace(go.Scatter(x=x5, y=y5, fill='tozeroy', fillcolor='rgba(239, 68, 68, 0.2)', line=dict(color='#ef4444', width=3)))
        fig5.add_vline(x=l_mean, line_dash="dash", line_color="black")
        fig5.update_layout(height=250, template="plotly_white", xaxis_title="Demand", yaxis_title="Probability Density", yaxis_range=[0, 0.008])
        st.plotly_chart(fig5, use_container_width=True)
        
        st.error("💡 **Takeaway:** Move the slider to 20 weeks. Notice how the curve flattens out. Because extreme upside spikes become much more likely, your mathematical optimal reliance on the fast Surge supplier grows drastically.")

    with t4:
        st.subheader("Step 4: The Lifecycle Reality")
        st.markdown("In traditional academic models, unsold goods are liquidated at a loss. But what if your product is shelf-stable? Change the scenario below to see how the Overage Cost ($C_o$) and your optimal Service Level adapt.")
        
        l_scenario = st.radio("Product Lifecycle Scenario", ["End of Life (Sunset)", "FMCG (Risk of Obsolescence)", "Shelf-Stable (Ongoing)"], index=0, key="l_scen")
        
        # Calculate optimal Qs to shade the curve
        nv_b = newsvendor_base(l_salvage, l_base_cost, l_surge_cost, l_mean, l_sigma, 1.85, l_scenario)
        nv_s = newsvendor_surge(l_price, l_salvage, l_base_cost, l_surge_cost, l_mean, l_sigma, 1.85, l_scenario)
        q_b, q_t = int(nv_b['optimal_q']), int(nv_s['optimal_q'])

        c3, c4 = st.columns([1, 2])
        with c3:
            st.metric("Base Co (Overage Penalty)", f"£{nv_b['co']:.2f}")
            st.metric("Optimal Base Order", f"{q_b:,} units")
            st.metric("Optimal Total Order", f"{q_t:,} units")
        with c4:
            x2 = np.linspace(200, 1800, 600)
            y2 = stats.norm.pdf(x2, l_mean, l_sigma)
            fig3 = go.Figure()
            m_base = x2 <= q_b
            fig3.add_trace(go.Scatter(x=np.concatenate([x2[m_base], [q_b, x2[m_base][0]]]), y=np.concatenate([y2[m_base], [0, 0]]), fill='toself', fillcolor='rgba(16,185,129,0.3)', line=dict(color='rgba(0,0,0,0)'), name='Base Coverage'))
            m_surge = (x2 > q_b) & (x2 <= q_t)
            fig3.add_trace(go.Scatter(x=np.concatenate([[q_b], x2[m_surge], [q_t, q_b]]), y=np.concatenate([[0], y2[m_surge], [0, 0]]), fill='toself', fillcolor='rgba(245,158,11,0.3)', line=dict(color='rgba(0,0,0,0)'), name='Surge Coverage'))
            fig3.add_trace(go.Scatter(x=x2, y=y2, mode='lines', line=dict(color='#1e3a5f'), showlegend=False))
            fig3.update_layout(height=300, margin=dict(l=0,r=0,t=0,b=0), template="plotly_white")
            st.plotly_chart(fig3, use_container_width=True)

        st.info("💡 **Takeaway:** When inventory doesn't spoil (Shelf-Stable), the penalty for over-ordering drops to almost zero. The model shifts the green area to the right, telling you to buy almost everything from the cheap Base supplier.")

    with t5:
        st.subheader("Step 5: The Bottom Line (Target Service Level)")
        st.markdown("**Target Service Level** isn't an arbitrary goal set by management (e.g., \"We must hit 99% fulfillment\"). In supply chain finance, Service Level is a mathematical output. It is the exact probability of *not* stocking out that maximizes your expected profit.")
        st.markdown("By calculating the Expected Profit across every possible probability, we map out the financial frontier. The absolute peak of this curve is your optimal Service Level.")
        
        df_l_sweep = dual_source_sweep(l_price, l_salvage, l_mean, l_sigma, l_base_cost, l_surge_cost, 0, 0, 1.85, q_b, l_scenario)
        l_best = df_l_sweep.loc[df_l_sweep["Exp. Profit (£)"].idxmax()]
        
        fig4 = make_subplots(specs=[[{"secondary_y": True}]])
        fig4.add_trace(go.Scatter(x=df_l_sweep["Service Level (%)"], y=df_l_sweep["Exp. Profit (£)"], name="Expected Profit (£)", line=dict(color="#2563eb", width=3)), secondary_y=False)
        fig4.add_vline(x=l_best["Service Level (%)"], line_dash="dash", line_color="red", annotation_text=f"Max Profit at {l_best['Service Level (%)']:.1f}%")
        fig4.update_layout(height=400, template="plotly_white", xaxis_title="Target Service Level (%)", yaxis_title="Expected Profit (£)")
        st.plotly_chart(fig4, use_container_width=True)
        
        st.success("🎉 You've mastered the math! You can now switch back to **Pro Mode** in the sidebar to run custom numbers for your own business cases.")
