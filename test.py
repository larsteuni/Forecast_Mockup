import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# --- Data Loading and Augmentation (UNCHANGED) ---
@st.cache_data
def load_and_augment_from_url():
    url = 'https://drive.google.com/uc?id=1ks8TEp5opVZURUlYkeajaKfuHfx8N2OC'
    try:
        df = pd.read_csv(url, encoding='latin1')
    except Exception as e:
        st.error(f"Failed to load data from the online source: {e}")
        return None
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df.dropna(subset=['StockCode', 'Description'], inplace=True)
    df = df[~df['Invoice'].astype(str).str.startswith('C')]; df = df[df['Quantity'] > 0]
    df['Description'] = df['Description'].str.strip().str.upper()
    end_date = df['InvoiceDate'].max(); start_date = end_date - pd.Timedelta(days=180)
    recent_sales_df = df[df['InvoiceDate'] >= start_date]
    sku_agg = recent_sales_df.groupby(['StockCode', 'Description']).agg(total_quantity_sold=('Quantity', 'sum'), Price=('Price', 'mean')).reset_index()
    sku_agg['Avg_Daily_Demand'] = sku_agg['total_quantity_sold'] / 180
    sku_agg = sku_agg[sku_agg['Avg_Daily_Demand'] > 0].copy()
    daily_sales = recent_sales_df.groupby(['StockCode', pd.Grouper(key='InvoiceDate', freq='D')])['Quantity'].sum().reset_index()
    volatility_df = daily_sales.groupby('StockCode')['Quantity'].std() / daily_sales.groupby('StockCode')['Quantity'].mean()
    volatility_df = volatility_df.reset_index(name='Demand_Volatility').fillna(0.1)
    sku_df = pd.merge(sku_agg, volatility_df, on='StockCode', how='left').fillna(0.1)
    sku_df.rename(columns={'StockCode': 'SKU'}, inplace=True)
    sku_df['Unit_Cost'] = sku_df['Price'] * np.random.uniform(0.5, 0.7, size=len(sku_df))
    sku_df['Current_Stock'] = (sku_df['Avg_Daily_Demand'] * np.random.uniform(0, 100, size=len(sku_df))).astype(int)
    sku_df['Days_Out_Of_Stock'] = 0
    zero_stock_indices = sku_df[sku_df['Current_Stock'] == 0].index
    sku_df.loc[zero_stock_indices, 'Days_Out_Of_Stock'] = np.random.randint(1, 91, size=len(zero_stock_indices))
    sku_df['Is_Active'] = True
    sku_df.loc[sku_df['Days_Out_Of_Stock'] > 30, 'Is_Active'] = False
    sku_df['Inbound_Stock'] = (sku_df['Avg_Daily_Demand'] * np.random.uniform(0, 40, size=len(sku_df))).astype(int)
    sku_df['Lead_Time'] = np.random.randint(7, 45, size=len(sku_df))
    def assign_category(desc):
        if any(word in desc for word in ['SADDLE', 'BIKE', 'FRAME']): return 'Cycling'
        if any(word in desc for word in ['SHOE', 'TIGHTS']): return 'Running'
        if any(word in desc for word in ['BACKPACK', 'TENT', 'FLASK']): return 'Outdoor'
        if any(word in desc for word in ['BOTTLE', 'LIGHT', 'ALARM', 'CLOCK']): return 'Accessories'
        return 'General Goods'
    sku_df['Category'] = sku_df['Description'].apply(assign_category)
    def assign_supplier_brand(category):
        if category == 'Cycling': return ('Shimano', 'Performance Pro')
        if category == 'Running': return ('ASICS', 'Runners\' World')
        if category == 'Outdoor': return ('The North Face', 'Summit Series')
        if category == 'Accessories': return ('Garmin', 'LifeWare')
        return ('Global Sourcing Inc.', 'Everyday Essentials')
    assignments = sku_df['Category'].apply(assign_supplier_brand)
    sku_df['Supplier'] = assignments.str[0]; sku_df['Brand'] = assignments.str[1]
    final_cols = ['SKU', 'Description', 'Avg_Daily_Demand', 'Demand_Volatility', 'Price', 'Unit_Cost', 'Current_Stock', 'Inbound_Stock', 'Lead_Time', 'Supplier', 'Brand', 'Is_Active', 'Days_Out_Of_Stock']
    return sku_df[final_cols].copy()

# --- Business Logic (Verbose Style - UNCHANGED) ---
@st.cache_data
def apply_business_logic(df_input):
    df = df_input.copy(); df['p50'] = df['Avg_Daily_Demand']; df['Demand_Volatility'] = df['Demand_Volatility'].clip(0.1, 1.5); df['p90'] = df['p50'] * (1 + df['Demand_Volatility']); df['p50_safe'] = df['p50'].replace(0, 0.001); df['Safety_Stock_Days'] = (df['p90'] - df['p50']) * df['Lead_Time'] / df['p50_safe']; df['Forecast_DOS'] = (df['Current_Stock'] + df['Inbound_Stock']) / df['p50_safe']; df['Target_DOS'] = df['Lead_Time'] + df['Safety_Stock_Days'] + 7; df['AI_Reorder?'] = df['Forecast_DOS'] < (df['Lead_Time'] + df['Safety_Stock_Days']); df['AI_Reorder_Qty'] = ((df['Target_DOS'] * df['p50']) - (df['Current_Stock'] + df['Inbound_Stock'])).clip(lower=0).astype(int); df['BO4_Min_Stock'] = df['Avg_Daily_Demand'] * 5; df['BO4_Max_Stock'] = df['Avg_Daily_Demand'] * 10; df['BO4_Advice?'] = df['Current_Stock'] < df['BO4_Min_Stock']; df['BO4_Reorder_Qty'] = (df['BO4_Max_Stock'] - df['Current_Stock']).clip(lower=0).astype(int); df['Delta'] = df['AI_Reorder_Qty'] - df['BO4_Reorder_Qty']; df['Risk'] = pd.cut(df['Forecast_DOS'] / (df['Lead_Time'] + df['Safety_Stock_Days']), bins=[0, 1, 1.5, np.inf], labels=['🔴 High', '🟠 Medium', '🟢 Low'], right=False).cat.add_categories(['⚪ N/A']).fillna('⚪ N/A'); df.loc[~df['Is_Active'], ['AI_Reorder?', 'AI_Reorder_Qty', 'BO4_Advice?', 'BO4_Reorder_Qty']] = [False, 0, False, 0]; return df

# --- Main App UI ---
st.set_page_config(layout="wide", page_title="AI Reorder Intelligence")
st.title("🤖 AI Reorder Intelligence")

if 'active_tab' not in st.session_state: st.session_state.active_tab = "📈 Dashboard"
if 'selected_sku' not in st.session_state: st.session_state.selected_sku = None
processed_df = None
with st.spinner('Loading and processing live data from online source...'):
    base_df = load_and_augment_from_url()
if base_df is not None:
    processed_df = apply_business_logic(base_df)
if processed_df is not None:
    st.sidebar.header("Workbench Filters"); selected_suppliers = st.sidebar.multiselect("Filter by Supplier:", options=sorted(processed_df['Supplier'].unique()), default=sorted(processed_df['Supplier'].unique()), key="supplier_filter"); selected_brands = st.sidebar.multiselect("Filter by Brand:", options=sorted(processed_df['Brand'].unique()), default=sorted(processed_df['Brand'].unique()), key="brand_filter"); selected_risk = st.sidebar.multiselect("Filter by Risk Level:", options=processed_df['Risk'].unique(), default=sorted(processed_df['Risk'].unique()), key="risk_filter"); show_all = st.sidebar.checkbox("Show all SKUs (not just reorders)", value=False, key="show_all_filter"); include_legacy = st.sidebar.checkbox("Include Legacy (Inactive) SKUs", value=False, key="legacy_filter")
    filtered_df = processed_df[(processed_df['Supplier'].isin(selected_suppliers)) & (processed_df['Brand'].isin(selected_brands)) & (processed_df['Risk'].isin(selected_risk))]
    if not show_all: filtered_df = filtered_df[filtered_df['AI_Reorder?'] | filtered_df['BO4_Advice?']]
    active_df = filtered_df[filtered_df['Is_Active']]; legacy_df = filtered_df[~filtered_df['Is_Active']]; workbench_df = filtered_df if include_legacy else active_df
    try: active_tab_index = ["📈 Dashboard", "🛠️ Reorder Workbench", "🔍 SKU Investigation"].index(st.session_state.active_tab)
    except ValueError: active_tab_index = 0
    st.session_state.active_tab = st.radio("Navigation", options=["📈 Dashboard", "🛠️ Reorder Workbench", "🔍 SKU Investigation"], index=active_tab_index, horizontal=True, label_visibility="collapsed")

    if st.session_state.active_tab == "📈 Dashboard":
        st.header("Overall Performance Dashboard"); st.markdown("Metrics below are for **active SKUs** only."); active_kpi_df = processed_df[processed_df['Is_Active']]; col1, col2, col3 = st.columns(3); reorders_count = active_kpi_df[active_kpi_df['AI_Reorder?']].shape[0]; value_at_risk = (active_kpi_df[active_kpi_df['Risk'] == '🔴 High']['AI_Reorder_Qty'] * active_kpi_df['Price']).sum(); overstock_value = (active_kpi_df[active_kpi_df['Current_Stock'] > (active_kpi_df['Target_DOS'] * active_kpi_df['p50'])]['Current_Stock'] * active_kpi_df['Price']).sum(); col1.metric("AI Reorders Required", f"{reorders_count} SKUs"); col2.metric("Retail Value at Risk (High Risk)", f"€{value_at_risk:,.0f}"); col3.metric("Potential Overstock Retail Value", f"€{overstock_value:,.0f}")
    
    elif st.session_state.active_tab == "🛠️ Reorder Workbench":
        st.header("Interactive Reorder Workbench"); st.info(f"Showing **{len(workbench_df)}** SKUs. **Click any row to drill down.**")
        def handle_row_selection():
            if st.session_state.workbench_selection["selection"]["rows"]:
                selected_row_index = st.session_state.workbench_selection["selection"]["rows"][0]; st.session_state.selected_sku = workbench_df.iloc[selected_row_index]['SKU']; st.session_state.active_tab = "🔍 SKU Investigation"
        
        # --- FIX #2: Replaced use_container_width=True with width='stretch' ---
        st.dataframe(
            workbench_df[['SKU', 'Description', 'Supplier', 'Price', 'Current_Stock', 'Days_Out_Of_Stock', 'AI_Reorder_Qty', 'Risk', 'Is_Active']], 
            on_select=handle_row_selection, 
            selection_mode="single-row", 
            hide_index=True, 
            key="workbench_selection"
        )
        
    elif st.session_state.active_tab == "🔍 SKU Investigation":
        st.header("SKU Drilldown & Investigation"); options = workbench_df['SKU'].unique()
        try: default_index = list(options).index(st.session_state.selected_sku)
        except (ValueError, TypeError): default_index = 0
        if not workbench_df.empty:
            sku_to_inspect = st.selectbox("Select an SKU to investigate:", options=options, index=default_index, format_func=lambda x: f"{x} - {processed_df.loc[processed_df['SKU'] == x, 'Description'].iloc[0]}", key="sku_selector"); st.session_state.selected_sku = sku_to_inspect; sample_sku_data = processed_df.loc[processed_df['SKU'] == sku_to_inspect].iloc[0]
            st.subheader("AI Logic & Financials"); st.markdown(f"**Description:** {sample_sku_data['Description']}"); st.markdown(f"- **Avg. Selling Price:** €{sample_sku_data['Price']:.2f}\n- **Simulated Unit Cost:** €{sample_sku_data['Unit_Cost']:.2f}"); st.success(f"➡ Recommended Reorder: {sample_sku_data['AI_Reorder_Qty']} units"); st.divider()
            graph_col1, graph_col2 = st.columns(2)
            days = np.arange(84); avg_demand = sample_sku_data['p50']; volatility = sample_sku_data['Demand_Volatility']; seasonality = 1 + np.sin(days / 7 * 2 * np.pi) * 0.15; random_noise = np.random.normal(0, avg_demand * volatility * 0.5, 84); p50_forecast = (avg_demand * seasonality) + random_noise; p50_forecast[p50_forecast < 0] = 0
            with graph_col1:
                st.subheader("1. Dynamic Demand Forecast"); p90_band = p50_forecast * (1 + volatility); p10_band = p50_forecast / (1 + volatility); fig, ax = plt.subplots(); ax.plot(days, p50_forecast, label='p50 (Forecasted Demand)', color='#1976D2', lw=2); ax.fill_between(days, p10_band, p90_band, color='#CFD8DC', alpha=0.6, label='p10-p90 Volatility Range'); ax.set_xlabel('Days from Today'); ax.set_ylabel('Forecasted Daily Demand (units)'); ax.set_ylim(bottom=0); ax.grid(True, linestyle='--', alpha=0.5); ax.legend(); st.pyplot(fig)
            with graph_col2:
                st.subheader("2. Projected Stock Level"); stock_over_time = np.zeros(84, dtype=float); stock_over_time[0] = sample_sku_data['Current_Stock']; lead_time_day = int(sample_sku_data['Lead_Time'])
                for day in range(1, 84):
                    stock_over_time[day] = stock_over_time[day-1] - p50_forecast[day-1]
                    if day == lead_time_day: stock_over_time[day] += sample_sku_data['Inbound_Stock']
                stock_over_time[stock_over_time < 0] = 0
                safety_stock_units = int(sample_sku_data['Safety_Stock_Days'] * sample_sku_data['p50']); fig, ax = plt.subplots(); ax.plot(days, stock_over_time, label='Projected Stock on Hand', color='#27ae60', lw=2)
                ax.axhspan(0, safety_stock_units, color='#c0392b', alpha=0.2, label='Safety Stock Zone')
                if sample_sku_data['Inbound_Stock'] > 0: ax.axvline(x=lead_time_day, color='#e67e22', linestyle='--', lw=2, label=f'Inbound Stock Arrival (+{int(sample_sku_data["Inbound_Stock"])})')
                ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: '{:.0f}'.format(x))); ax.set_xlabel('Days from Today'); ax.set_ylabel('Inventory Units (Whole Numbers)'); ax.set_ylim(bottom=0); ax.grid(True, linestyle='--', alpha=0.5); ax.legend(); st.pyplot(fig)
            
            st.divider()

            # --- FIX #1: Reverted to the working st.expander for the override workflow ---
            with st.expander(f"Override AI Recommendation for {sku_to_inspect}"):
                st.warning(f"AI Recommended Quantity: **{sample_sku_data['AI_Reorder_Qty']}**")
                buyer_qty = st.number_input("Your Order Quantity:", min_value=0, value=sample_sku_data['AI_Reorder_Qty'], key=f"override_qty_{sku_to_inspect}")
                reason = st.selectbox("Reason for Override:", options=['(No Override)', 'Promotional Stock Up', 'Supplier Deal', 'Discontinuation Risk', 'Manual Correction', 'Other...'], key=f"reason_{sku_to_inspect}")
                if st.button("Confirm Action & Log Override", key=f"confirm_{sku_to_inspect}"):
                    if buyer_qty != sample_sku_data['AI_Reorder_Qty'] and reason != '(No Override)':
                        st.success(f"Override for {sku_to_inspect} logged: Buyer Quantity {buyer_qty}, Reason: {reason}")
                    elif buyer_qty == sample_sku_data['AI_Reorder_Qty']:
                        st.info("Action logged. No override was made.")
                    else:
                        st.error("Please select a reason for the override.")
        else:
            st.warning("No SKUs match the current filters.")

