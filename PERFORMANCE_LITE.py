import sqlite3
import streamlit as st
import numpy as np
import pandas as pd
from datetime import timedelta
import os
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)

def SEC_USERS():
    users_folder = '/Users'
    usernames = [name for name in os.listdir(users_folder) if
                 os.path.isdir(os.path.join(users_folder, name)) and name != 'Shared']
    for user in usernames:
        user_id = user
    return user_id

user_id = SEC_USERS()

if user_id == 'VictorCianni' :
    database_path = "C:/Users/"+user_id+"/Library/CloudStorage/OneDrive-SharedLibraries-AlpianSA/Investments - General/IMM.db"
else :
    database_path = "/Users/"+user_id+"/Library/CloudStorage/OneDrive-SharedLibraries-AlpianSA/Investments - General/IMM.db"
try:
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()
    print("Connection successful!")
except Exception as e:
    print(f"Error connecting to database: {e}")

df_grille_essential = pd.read_sql("SELECT * FROM GRILLE_ESSENTIAL",conn)
plan_essential = ['BASIC_DISCRETIONARY_CAUTIOUS_SWISS', 'BASIC_DISCRETIONARY_MODERATE_SWISS', 'BASIC_DISCRETIONARY_BALANCED_SWISS', 'BASIC_DISCRETIONARY_AGGRESSIVE_SWISS', 'BASIC_DISCRETIONARY_VERY_AGGRESSIVE_SWISS', 'BASIC_DISCRETIONARY_CAUTIOUS_FOREIGN', 'BASIC_DISCRETIONARY_MODERATE_FOREIGN', 'BASIC_DISCRETIONARY_BALANCED_FOREIGN', 'BASIC_DISCRETIONARY_AGGRESSIVE_FOREIGN', 'BASIC_DISCRETIONARY_VERY_AGGRESSIVE_FOREIGN', 'BASIC_DISCRETIONARY_CAUTIOUS_ESG', 'BASIC_DISCRETIONARY_MODERATE_ESG', 'BASIC_DISCRETIONARY_BALANCED_ESG', 'BASIC_DISCRETIONARY_AGGRESSIVE_ESG', 'BASIC_DISCRETIONARY_VERY_AGGRESSIVE_ESG','BASIC_DISCRETIONARY_BALANCED_CRYPTO','BASIC_DISCRETIONARY_AGGRESSIVE_CRYPTO', 'BASIC_DISCRETIONARY_VERY_AGGRESSIVE_CRYPTO']

df_PW = pd.read_sql("SELECT * FROM PW_BENCH",conn).set_index('date')
df_PW.index = pd.to_datetime(df_PW.index)
df_PW = df_PW + 1
df_PW = df_PW.pct_change().fillna(0)


df = pd.read_sql("SELECT * FROM Daily_Return_Premium", conn).set_index('date')
list_premium = ['CAUTIOUS_PREMIUM', 'MODERATE_PREMIUM', 'BALANCED_PREMIUM', 'AGGRESSIVE_PREMIUM', 'VERY_AGGRESSIVE_PREMIUM']
list_premium_Essential = ['CAUTIOUS_PREMIUM', 'MODERATE_PREMIUM', 'BALANCED_PREMIUM', 'AGGRESSIVE_PREMIUM', 'VERY_AGGRESSIVE_PREMIUM', 'ESSENTIAL']
df = df.rename({'1.0' : 'CAUTIOUS_PREMIUM', '2.0':'MODERATE_PREMIUM', '3.0':'BALANCED_PREMIUM', '4.0':'AGGRESSIVE_PREMIUM', '5.0':'VERY_AGGRESSIVE_PREMIUM'}, axis= 1).drop(columns=['0.0'])
df.index = pd.to_datetime(df.index)
df = df.loc['2022-12-31':]

df_asset_return = pd.read_sql("SELECT * FROM asset_class_return", conn).set_index('date')
df_asset_setup = pd.read_sql("SELECT * FROM asset_class_setup", conn)
bench_imm = df_asset_setup['ImmId'].tolist()
bench_imm = [f'{num}'for num in bench_imm]
df_price = pd.read_sql("SELECT * FROM PRICE_INSTRU_DISCR_DEV", conn).set_index('DATE')
#df_price.index = pd.to_datetime(df_price.index)
df_cur = pd.read_sql("SELECT * FROM CUR_RATE", conn).set_index('DATE')
df_cur.index = pd.to_datetime(df_cur.index)
df_cur['CHF'] = 1
filtered_columns = df_price.columns.intersection(bench_imm)
df_price_filtered = df_price[filtered_columns]
mapping = dict(zip(df_asset_setup['ImmId'].astype(str), df_asset_setup['Asset class']))
df_price_filtered = df_price_filtered.rename(mapping, axis=1)
df_price_filtered = df_price_filtered.replace(0, np.nan)
df_price_filtered = df_price_filtered.fillna(method='ffill')
#df_bench_return = df_price_filtered.pct_change().fillna(0).iloc[:-1]
df_price_filtered = df_price_filtered.iloc[:-1]
df_price_filtered.index = pd.to_datetime(df_price_filtered.index)
df_price_filtered = df_price_filtered.sort_index()
df_price_filtered = df_price_filtered.pct_change().fillna(0)
df_price_filtered = df_price_filtered.replace([np.inf, -np.inf], 0)

df_bench = df_price_filtered.merge(df_PW, right_index=True, left_index=True, how ='left')
list_bench = df_bench.columns.tolist()

def PREMIUM(selected_plan, start_date, end_date, selected_bench, CHF):
    if CHF == 'True':
        df_devise = df_asset_setup[df_asset_setup['Asset class'].isin(selected_bench)]
        df_devise = df_devise[['Asset class', 'Currency']]
        df_cur_selected = df_cur.loc[start_date:end_date]
        currency_mapping = df_devise.set_index('Asset class')['Currency'].to_dict()

        df_bench_return = df_bench
        filtered_columns = df_bench_return.columns.intersection(selected_bench)
        df_bench_return_filtered = df_bench_return[filtered_columns]
        bench = df_bench_return_filtered.loc[start_date:end_date]

        df_result = bench.copy()
        for product in bench.columns:
            currency = currency_mapping.get(product)
            if currency:
                df_result[product] = df_bench_return[product] * df_cur_selected[currency]
        # df_result = df_result.pct_change().fillna(0)
        # df_result = df_result.replace([np.inf, -np.inf], 0)

    else:
        df_bench_return = df_bench
        filtered_columns = df_bench_return.columns.intersection(selected_bench)
        df_bench_return_filtered = df_bench_return[filtered_columns]
        bench = df_bench_return_filtered.loc[start_date:end_date]
        df_result = bench
        # df_result = df_result.replace([np.inf, -np.inf], 0)

    if selected_plan in list_premium :
        grille_selected = df[[selected_plan]]
        grille_selected = grille_selected.loc[start_date:end_date]
        if not df_result.empty:
            df_merge = grille_selected.merge(df_result, left_index=True, right_index=True, how='left')
            df_cumul_grille = (df_merge + 1).cumprod()
        else:
            df_cumul_grille = (grille_selected + 1).cumprod()
    else :
        df_price_essential = pd.read_sql("SELECT * FROM PRICE_INSTRU_DISCR_DEV", conn).set_index('DATE').iloc[:-1]
        df_price_essential.index = pd.to_datetime(df_price_essential.index)
        df_price_essential = df_price_essential.sort_index()
        df_price_essential = df_price_essential.loc[start_date:end_date]
        df_price_essential = df_price_essential.replace(0, np.nan)
        df_price_essential = df_price_essential.fillna(method='ffill')
        df_price_essential = df_price_essential.pct_change()
        df_price_essential = df_price_essential.fillna(0)
        df_price_essential = df_price_essential.replace([np.inf, -np.inf], 0)
        grille_selected = df_grille_essential[['immId', selected_plan]]
        grille_selected = grille_selected[grille_selected[selected_plan] != 0]
        weights_grille = grille_selected[selected_plan].tolist()
        selected_columns = [str(col) for col in grille_selected.immId.tolist() if str(col) in df_price.columns]
        df_price_grille = df_price_essential[selected_columns]
        total_weighted_return_grille = (df_price_grille * weights_grille).sum(axis=1)
        df_cumul_grille = total_weighted_return_grille.to_frame(name=selected_plan)
        if not df_result.empty:
            df_merge = df_cumul_grille.merge(df_result, left_index=True, right_index=True, how='left')
            df_cumul_grille = (df_merge + 1).cumprod()
        else:
            df_cumul_grille = (df_cumul_grille + 1).cumprod()
    df_gain_loss = []
    if len(selected_bench) == 1 :
        df_gain_loss = df_cumul_grille.copy()
        df_gain_loss['P&L'] = ((df_gain_loss[selected_plan] - df_gain_loss.iloc[:, -1]) * 100) + 100
    return df_cumul_grille, df_gain_loss

#print(PREMIUM('BASIC_DISCRETIONARY_BALANCED_CRYPTO','2024-01-01','2025-02-24', ['PW_very_dyn_bench'], 'True'))

st.write("**Sélectionnez une plage de dates :**")
date_range_option = st.selectbox(
    "Choisissez la plage de dates",
    options=["ALL","3Y", "1Y", "YTD", "6m","3m","1m", "1s", "Custom"],index=0)

#end_date = df_price_filtered.index.max()

if date_range_option == "Custom":
    col1, col2 = st.columns(2)
    start_date = col1.date_input("Start Date", value=df_price_filtered.index.min().date())
    end_date = col2.date_input("End Date", value=df_price_filtered.index.max().date())
    if start_date > end_date:
        st.error("Start Date cannot be after End Date. Please adjust the range.")
else:
    end_date = df_price_filtered.index.max()
    if date_range_option == "3Y":
        start_date = end_date - timedelta(days=3 * 365)
    elif date_range_option == "1Y":
        start_date = end_date - timedelta(days=365)
    elif date_range_option == "YTD":
        start_date = pd.Timestamp(f"{end_date.year}-01-01")
    elif date_range_option == "6m":
        start_date = end_date - timedelta(days=6 * 30)
    elif date_range_option == "1m":
        start_date = end_date - timedelta(days=30)
    elif date_range_option == "3m":
        start_date = end_date - timedelta(days=3 * 30)
    elif date_range_option == "1s":
        start_date = end_date - timedelta(days=7)
    elif date_range_option == "ALL":
        start_date = df.index.min()

st.write(f"Plage de dates : {start_date} à {end_date}")

if 'clicked_buttons' not in st.session_state:
    st.session_state.clicked_buttons = set()

cols = st.columns(4)

list_bench = sorted(list_bench)

for i, button_label in enumerate(list_bench):
    col = cols[i % len(cols)]
    is_selected = button_label in st.session_state.clicked_buttons
    button_style = "background-color: red; color: white;" if is_selected else ""

    if col.button(button_label, key=f"btn_{button_label}"):
        if is_selected:
            st.session_state.clicked_buttons.remove(button_label)
        else:
            st.session_state.clicked_buttons.add(button_label)

bench = list(st.session_state.clicked_buttons)


selected_plan = st.selectbox(
    "**Sélectionnez un plan :**",
    options=list_premium_Essential,
    index=0
)
if selected_plan == "ESSENTIAL":
    selected_plan = st.selectbox(
        "**Sélectionnez un plan :**",
        options=plan_essential,
        index=0)

CHF = st.checkbox("**Mettre le bench en CHF:**")

if CHF:
    st.write("Checkbox is TRUE: Bench is in CHF")
else:
    st.write("Checkbox is FALSE: Bench is not in CHF")

#st.write(f"Plage de dates : {start_date.date()} à {end_date.date()}")
df_cumul_selected, df_gain_loss = PREMIUM(selected_plan, start_date, end_date, bench, str(CHF))
min_value = df_cumul_selected.min().min()
max_value = df_cumul_selected.max().max()
buffer = 0.05
adjusted_min = min_value * (1 - buffer)
adjusted_max = max_value * (1 + buffer)
df_cumul_selected = df_cumul_selected / df_cumul_selected.iloc[0]
fig, ax = plt.subplots(figsize=(10, 6))
df_cumul_selected.plot(ax=ax)
for idx, line in enumerate(ax.lines):
    x = df_cumul_selected.index[-1]
    y = df_cumul_selected.iloc[-1, idx]
    if idx == 0:
        line.set_linewidth(2.5)
    else:
        line.set_linestyle('--')
    ax.annotate(
        f"{y-1:.4f}",
        xy=(x, y),
        xytext=(5, 0),
        textcoords="offset points",
        fontsize=10,
        color=line.get_color(),
        ha="left",
        va="center"
    )
ax.set_title("Rendements Cumulés des Produits", fontsize=16)
ax.set_xlabel("Date", fontsize=12)
ax.set_ylabel("Rendement Cumulé", fontsize=12)
ax.set_ylim(adjusted_min, adjusted_max)
ax.set_xlim(df_cumul_selected.index.min(), df_cumul_selected.index.max())
plt.xticks(rotation=45)
plt.grid()
st.pyplot(fig)


if len(bench) == 1:
    df_gain_loss_plot = df_gain_loss
    df_gain_loss_plot['color'] = df_gain_loss_plot['P&L'].apply(lambda x: 'green' if x > 100 else 'red')
    df_gain_loss_plot['adjusted_P&L'] = df_gain_loss_plot['P&L'] - 100
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(df_gain_loss_plot.index, df_gain_loss_plot['adjusted_P&L'], color=df_gain_loss_plot['color'])
    ax.axhline(y=0, color='black', linewidth=1.5, linestyle='--')
    ax.set_xlabel('Date')
    ax.set_ylabel('P&L Deviation from 100')
    ax.set_title('P&L Over Time (Flipped Axis for <100)')
    plt.xticks(rotation=45)
    st.pyplot(fig)
