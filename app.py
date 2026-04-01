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
        holding_cost_pct = st.slider("Annual Holding Cost (%)", 10, 40, 20) / 100.0
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
        k1.metric("Total Service Level", f"{best['Service Level (%)']:.1f}%")
        k2.metric("Base Order", f"{int(best['Q Base']):,}")
        k3.metric("Surge Order", f"{int(best['Q Surge']):,}")
        k4.metric("Expected Profit", f"£{int(best['Exp. Profit (£)']):,}")

        t1, t2, t3 = st.tabs(["📊 Profit Curve", "📐 Distribution", "📋 Data Table"])
        with t1:
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Scatter(x=df_sweep["Service Level (%)"], y=df_sweep["Exp. Profit (£)"], name="Profit", line=dict(color="#2563eb", width=3)), secondary_y=False)
            fig.add_trace(go.Scatter(x=df_sweep["Service Level (%)"], y=df_sweep["Q Surge"], name="Surge Units", line=dict(color="#f59e0b", dash="dot")), secondary_y=True)
            fig.add_vline(x=best["Service Level (%)"], line_dash="dash", line_color="red", annotation_text="Optimum")
            fig.update_layout(height=400, template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
        with t2:
            x = np.linspace(mean_demand - 4*sigma, mean_demand + 4*sigma, 600)
            y_pdf = stats.norm.pdf(x, mean_demand, sigma)
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=x, y=y_pdf, mode='lines', line=dict(color='#1e3a5f'), name='Demand'))
            fig2.add_vline(x=best["Q Base"], line_dash="dash", line_color="#10b981", annotation_text="Base Q")
            fig2.add_vline(x=best["Q Total"], line_dash="dash", line_color="#f59e0b", annotation_text="Total Q")
            fig2.update_layout(height=400, template="plotly_white")
            st.plotly_chart(fig2, use_container_width=True)
        with t3:
            st.dataframe(df_sweep, use_container_width=True, hide_index=True)

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

    t1, t2, t3, t4, t5 = st.tabs(["1. The Dilemma", "2. The Hedge", "3. The Lifecycle", "4. The Bottom Line", "5. The Lead Time Trap 🚨"])

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
        st.markdown("How much of that 'expected' demand should we commit to the Base supplier? We decide by weighing the cost of under-ordering ($C_u$) against the cost of over-ordering ($C_o$).")
        
        l_surge_cost = st.slider("Surge Supplier Premium (Unit Cost)", 20.5, 40.0, 22.0, step=0.5, key="l_surge")
        st.markdown(f"*Base Supplier Cost is fixed at £{l_base_cost:.2f}. Selling Price is £{l_price:.2f}.*")

        c1, c2 = st.columns(2)
        base_cu = l_surge_cost - l_base_cost
        base_co = (l_base_cost - l_salvage) + 1.85
        with c1:
            st.markdown("### 🏭 Base Supplier")
            st.metric("Cost of Under-ordering (Cu)", f"£{base_cu:.2f}", help="If you order too little from Base, you don't lose the sale! You just pay the Surge premium.")
            st.metric("Cost of Over-ordering (Co)", f"£{base_co:.2f}", help="Markdown loss + Holding cost")
        
        surge_cu = l_price - l_surge_cost
        surge_co = (l_surge_cost - l_salvage) + 1.85
        with c2:
            st.markdown("### ⚡ Surge Supplier")
            st.metric("Cost of Under-ordering (Cu)", f"£{surge_cu:.2f}", help="If you order too little from Surge, you stock out and lose the entire profit margin.")
            st.metric("Cost of Over-ordering (Co)", f"£{surge_co:.2f}")

        st.info("💡 **Takeaway:** Because the penalty for under-ordering from your Base supplier is *only the surge premium* (not the lost sale), the math keeps your initial base order intentionally conservative.")

    with t3:
        st.subheader("Step 3: The Lifecycle Reality")
        st.markdown("In traditional Newsvendor models, unsold goods are liquidated at a loss. But what if your product is shelf-stable? Change the scenario below to see how the Overage Cost ($C_o$) and your optimal Service Level adapt.")
        
        l_scenario = st.radio("Product Lifecycle Scenario", ["End of Life (Sunset)", "FMCG (Risk of Obsolescence)", "Shelf-Stable (Ongoing)"], index=0, key="l_scen")
        
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

    with t4:
        st.subheader("Step 4: The Bottom Line")
        st.markdown("By calculating the Expected Profit across every possible service level, we can map out the financial frontier. The peak of this curve is your mathematical optimum.")
        
        df_l_sweep = dual_source_sweep(l_price, l_salvage, l_mean, l_sigma, l_base_cost, l_surge_cost, 0, 0, 1.85, q_b, l_scenario)
        l_best = df_l_sweep.loc[df_l_sweep["Exp. Profit (£)"].idxmax()]
        
        fig4 = make_subplots(specs=[[{"secondary_y": True}]])
        fig4.add_trace(go.Scatter(x=df_l_sweep["Service Level (%)"], y=df_l_sweep["Exp. Profit (£)"], name="Expected Profit (£)", line=dict(color="#2563eb", width=3)), secondary_y=False)
        fig4.add_vline(x=l_best["Service Level (%)"], line_dash="dash", line_color="red", annotation_text=f"Max Profit at {l_best['Service Level (%)']:.1f}%")
        fig4.update_layout(height=400, template="plotly_white", xaxis_title="Target Service Level (%)", yaxis_title="Expected Profit (£)")
        st.plotly_chart(fig4, use_container_width=True)

    with t5:
        st.subheader("Step 5: Working Capital & The Bullwhip Effect")
        st.markdown("For an MBA, lead time isn't just a supply chain metric—it's a massive financial lever. Longer lead times do two destructive things: they trap working capital (destroying ROCE), and they exponentially amplify the **Bullwhip Effect**.")
        
        l_lead_time = st.slider("Base Supplier Lead Time (Weeks)", 2, 24, 12, key="l_lt_mba")
        
        # Financial Math based on Little's Law
        weekly_demand = l_mean / 12.0 # Assuming 1000 units over a 12-week season/quarter
        pipeline_units = weekly_demand * l_lead_time
        working_capital = pipeline_units * l_base_cost
        cost_of_capital = working_capital * (l_holding_pct * (l_lead_time/52.0))
        
        st.markdown("#### 💰 Financial Impact: The Pipeline Cash Trap")
        st.markdown("*Assume an average demand rate of ~83 units/week and a 20% Annual Cost of Capital (WACC).*")
        col_f1, col_f2, col_f3 = st.columns(3)
        col_f1.metric("Units in Transit (Pipeline)", f"{int(pipeline_units):,} units")
        col_f2.metric("Working Capital Tied Up", f"£{int(working_capital):,}")
        col_f3.metric("Opportunity Cost of Capital", f"£{int(cost_of_capital):,}")

        st.markdown("---")
        st.markdown("#### 🌊 The Bullwhip Effect Simulator")
        st.markdown("What happens if customer demand suddenly spikes by **20% in Week 4**? With a 2-week lead time, you just order 20% more. But with a long lead time, you have to order enough to cover the spike *for the entire lead time horizon* to refill your pipeline. This creates a massive phantom order.")
        
        # Bullwhip Math
        weeks = np.arange(1, 15)
        actual_demand = [100]*3 + [120]*11 # 20% spike at week 4
        
        order_qty = []
        for i, d in enumerate(actual_demand):
            if i == 3: # The spike
                # Panic order: The new demand + adjusting the pipeline safety stock over the lead time
                panic_spike = d + ((d - 100) * l_lead_time)
                order_qty.append(panic_spike)
            else:
                order_qty.append(d)
                
        fig_bw = go.Figure()
        fig_bw.add_trace(go.Scatter(x=weeks, y=actual_demand, name="Actual Customer Demand", line=dict(color="#10b981", width=3)))
        fig_bw.add_trace(go.Scatter(x=weeks, y=order_qty, name="Your Order to Supplier", line=dict(color="#ef4444", width=3, dash="dash")))
        fig_bw.update_layout(height=350, template="plotly_white", yaxis_title="Units", xaxis_title="Week", yaxis_range=[0, max(order_qty)+50])
        st.plotly_chart(fig_bw, use_container_width=True)
        
        st.error("💡 **Takeaway:** Move the slider to 20 weeks. Look at the red spike. Long lead times force your supply chain to violently overreact to small changes in demand, leading to massive excess inventory arriving just as the demand spike fades.")
