import pandas as pd
import numpy as np
import streamlit as st
from utiils_pairs import *

st.title('Pairs Trading App')

df_price = pd.read_parquet('streamlit/df_price.parquet')
df_backtest_result = pd.read_parquet('streamlit/df_backtest_result.parquet')

#dropdown box to show how pairs are ranked
st.text('Select how pairs are ranked')
rank_by = st.selectbox('Rank by', ['Best AUM', 'Best win rate', 'Worst AUM', 'Worst win rate'])
df_show = df_backtest_result.sort_values(by='capital', ascending=False) #default
if rank_by == 'Best AUM':
    df_show = df_backtest_result.sort_values(by='capital', ascending=False)
elif rank_by == 'Best win rate':
    st.text('Only show pairs with more than 5 trades')
    df_show = df_backtest_result[df_backtest_result['long_enter_count'] + df_backtest_result['short_enter_count'] > 5].sort_values(by='win_rate', ascending=False)
elif rank_by == 'Worst AUM':
    df_show = df_backtest_result.sort_values(by='capital', ascending=True)
elif rank_by == 'Worst win rate':
    st.text('Only show pairs with more than 5 trades')
    df_show = df_backtest_result[df_backtest_result['long_enter_count'] + df_backtest_result['short_enter_count'] > 5].sort_values(by='win_rate', ascending=True)
st.table(df_show.head(10))

pairs = []
for i in range(len(df_backtest_result)):
    pairs.append((df_backtest_result.iloc[i,0], df_backtest_result.iloc[i,1]))

#selectbox inline
stock1, stock2 = st.selectbox('Select Stock 1', df_price.columns[1:]), st.selectbox('Select Stock 2', df_price.columns[1:], index=1)
#selecbox inline
# stock1 = st.selectbox('Select Stock 1', df_price.columns[1:])
# stock2 = st.selectbox('Select Stock 2', df_price.columns[1:])
if stock1 == stock2:
    st.error('You selected the same stock! Please select another stock.')
    st.stop()
st.pyplot(plot_coint(df_price, stock1, stock2, TRADING_PERIOD=200, SD_MULTIPLIER=1))
st.text('Example of signal generation based on 1SD entry')

#return setting to default
if st.button('Reset'):
    st.session_state['entry'] = 1.0
    st.session_state['exit'] = 0.0
    st.session_state['cutloss'] = 2.0

ENTRY_SD = st.slider('Entry SD', 0.1, 3.0, 1.0, key='entry')
EXIT_SD = st.slider('Exit SD', 0.1, 1.0, 0.0, key='exit')
CUTLOSS_SD = st.slider('Cutloss SD', 0.1, 5.0, 2.0, key='cutloss')


# stock1, stock2 = 'AWC', 'CRC'   # Best AUM growth
# stock1, stock2 = 'BH', 'PTT'   # Worst AUM growth, hold position too long while regime already change. Consider cutloss if holding period too long.
# stock1, stock2 = 'AOT', 'BH'
TRADING_PERIOD = 200
INITIAL_CAPITAL = 100
pairs = Backtest(df_price, stock1, stock2, ENTRY_SD, EXIT_SD, CUTLOSS_SD, TRADING_PERIOD, INITIAL_CAPITAL)
pairs.gen_signal()
capital = pairs.df['capital'].values[-1]
long_enter_count = len(pairs.df_txn[pairs.df_txn['position_type'] == 'long'])
long_win_count = len(pairs.df[pairs.df['is_just_exit_long'] == 1])
short_enter_count = len(pairs.df_txn[pairs.df_txn['position_type'] == 'short'])
short_win_count = len(pairs.df[pairs.df['is_just_exit_short'] == 1])
win_rate = (long_win_count + short_win_count) / (long_enter_count + short_enter_count)
st.text(f'Capital: {capital:.2f}')
st.text('Long enter count: ' + str(long_enter_count))
st.text('Long win count: ' + str(long_win_count))
st.text('Short enter count: ' + str(short_enter_count))
st.text('Short win count: ' + str(short_win_count))
st.text(f'Win rate: {win_rate:.2%}')
st.pyplot(plot_backtest(pairs.df))
st.table(pairs.df_txn)