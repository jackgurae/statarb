import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import coint
from statsmodels.regression.linear_model import OLS

def create_price_pair(df, stock1, stock2, TRADING_PERIOD=None):
    df_price_pair = df[['date', stock1, stock2]].copy()
    #fill missing value
    df_price_pair = df_price_pair.fillna(method='ffill')
    #drop row with missing value
    df_price_pair = df_price_pair.dropna()
    if TRADING_PERIOD:
        df_train = df_price_pair.iloc[:-TRADING_PERIOD].copy()
        df_test = df_price_pair.iloc[-TRADING_PERIOD:].copy()
    else:
        df_train = df_price_pair.copy()
        df_test = df_price_pair.iloc[:int(len(df_price_pair)*0.1)].copy()   #10% of data for testing
   
    #calculate spread
    beta = OLS(df_train[stock1], df_train[stock2]).fit().params[0] #first stock is y, second stock is x
    df_train['spread'] = df_train[stock1] - beta * df_train[stock2]
    df_train['zscore'] = (df_train['spread'] - df_train['spread'].mean()) / df_train['spread'].std()  
     
    df_test['spread'] = df_test[stock1] - beta * df_test[stock2]
    df_test['zscore'] = (df_test['spread'] - df_train['spread'].mean()) / df_train['spread'].std()
    return df_train, df_test, beta

def plot_coint(df, stock1, stock2, TRADING_PERIOD=None, SD_MULTIPLIER=1):
    df_train, df_test, _ = create_price_pair(df, stock1, stock2, TRADING_PERIOD)
    df_price_pair = pd.concat([df_train, df_test]).sort_values(by='date')
    df_price_pair.dropna(inplace=True)
    #plot price and spread in 2 subplots
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    #plot prices of 2 stocks using separate y-axis
    ax_twin = ax.twinx()
    ax.plot(df_price_pair['date'], df_price_pair[stock1], label=stock1)
    ax_twin.plot(df_price_pair['date'], df_price_pair[stock2], label=stock2, color='orange')
    ax.set_ylabel('Price')
    #color region for test period
    test_start = df_test['date'].iloc[0]
    test_end = df_test['date'].iloc[-1]
    ax.axvspan(test_start, test_end, alpha=0.5, color='lightgrey')
    ax.set_ylabel('Price')
    ax.legend()
    ax_twin.legend()
    
    #plot spread
    ax2.plot(df_price_pair['date'], df_price_pair['spread'], color='red', label='spread')
    ax2.set_ylabel('Spread')
    ax2.legend()
    #plot cointegration region
    ax2.axhline(y=np.mean(df_train['spread']), color='black', linestyle='--')
    ax2.axhline(y=np.mean(df_train['spread'])+np.std(df_train['spread']), color='black', linestyle='--')
    ax2.axhline(y=np.mean(df_train['spread'])-np.std(df_train['spread']), color='black', linestyle='--')
    ax2.axvspan(test_start, test_end, alpha=0.5, color='lightgrey')
    #mark buy/sell signal using zscore
    #plot with scatter
    buy = df_price_pair[df_price_pair['zscore'] < -SD_MULTIPLIER]
    sell = df_price_pair[df_price_pair['zscore'] > SD_MULTIPLIER]
    ax2.scatter(buy['date'], buy['spread'], color='green', marker='^', label='buy')
    ax2.scatter(sell['date'], sell['spread'], color='red', marker='v', label='sell')
    return fig

def plot_backtest(df_plot):
    #plot backtest result
    #set index to date
    df_plot = df_plot.set_index('date')
    fig, ax = plt.subplots(2, 1, figsize=(15, 10))
    ax[0].plot(df_plot['spread'], label='spread')
    ax[0].scatter(df_plot[df_plot['is_just_long'] == 1].index, df_plot[df_plot['is_just_long'] == 1]['spread'], label='long', marker='^', color='g')
    ax[0].scatter(df_plot[df_plot['is_just_short'] == 1].index, df_plot[df_plot['is_just_short'] == 1]['spread'], label='short', marker='v', color='r')
    ax[0].scatter(df_plot[df_plot['is_just_exit_long'] == 1].index, df_plot[df_plot['is_just_exit_long'] == 1]['spread'], label='exit long', marker='x', color='g')
    ax[0].scatter(df_plot[df_plot['is_just_exit_short'] == 1].index, df_plot[df_plot['is_just_exit_short'] == 1]['spread'], label='exit short', marker='x', color='r')
    ax[0].legend()
    ax[0].set_title('Spread')
    ax[0].grid()

    #ax[1] for accumulated profit
    ax[1].plot(df_plot['capital'], label='capital')
    ax[1].scatter(df_plot[df_plot['is_just_long'] == 1].index, df_plot[df_plot['is_just_long'] == 1]['capital'], label='long', marker='^', color='g')
    ax[1].scatter(df_plot[df_plot['is_just_short'] == 1].index, df_plot[df_plot['is_just_short'] == 1]['capital'], label='short', marker='v', color='r')
    ax[1].scatter(df_plot[df_plot['is_just_exit_long'] == 1].index, df_plot[df_plot['is_just_exit_long'] == 1]['capital'], label='exit long', marker='x', color='g')
    ax[1].scatter(df_plot[df_plot['is_just_exit_short'] == 1].index, df_plot[df_plot['is_just_exit_short'] == 1]['capital'], label='exit short', marker='x', color='r')
    ax[1].legend()
    ax[1].grid()
    ax[1].set_title('Capital')
    return fig

# create backtest class
# largely inspired from https://bsic.it/pairs-trading-building-a-backtesting-environment-with-python/
class Backtest:
    def __init__(self, df, stock1, stock2, ENTRY_SD, EXIT_SD, CUTLOSS_SD, TRADING_PERIOD, INITIAL_CAPITAL):
        self.df = df
        self.stock1 = stock1
        self.stock2 = stock2
        self.ENTRY_SD = ENTRY_SD
        self.EXIT_SD = EXIT_SD
        self.CUTLOSS_SD = CUTLOSS_SD
        self.TRADING_PERIOD = TRADING_PERIOD
        self.INITIAL_CAPITAL = INITIAL_CAPITAL

    def gen_signal(self):
        df_train, df_test, beta = create_price_pair(self.df, self.stock1, self.stock2, self.TRADING_PERIOD)
        self.df = pd.concat([df_train, df_test]).sort_values(by='date')
        #reset index
        self.df.reset_index(drop=True, inplace=True)
        self.df['signal'] = 0
        self.df['signal'] = np.where(self.df['zscore'] < -self.ENTRY_SD, 1, self.df['signal'])
        self.df['signal'] = np.where(self.df['zscore'] > self.ENTRY_SD, -1, self.df['signal'])
        # checking signal if signal is 1 then close position when spread is above mean
        # if signal is -1 then close position when spread is below mean
        #enter_long_spread, when zscore cross down below -ENTRY_SD
        self.df['enter_long_spread'] = self._cross_over(self.df['zscore'], -self.ENTRY_SD, cross_up=False)
        self.df['enter_short_spread'] = self._cross_over(self.df['zscore'], self.ENTRY_SD, cross_up=True)
        self.df['close_long_spread'] = self._cross_over(self.df['zscore'], self.EXIT_SD, cross_up=True)
        self.df['close_short_spread'] = self._cross_over(self.df['zscore'], -self.EXIT_SD, cross_up=False)
        self.df['cutloss_long_spread'] = self._cross_over(self.df['zscore'], self.CUTLOSS_SD, cross_up=True)
        self.df['cutloss_short_spread'] = self._cross_over(self.df['zscore'], -self.CUTLOSS_SD, cross_up=False)

        # initialize position
        self.df['position'] = 0
        self.df[self.stock1+'_position'] = 0
        self.df[self.stock2+'_position'] = 0
        self.df['entry_spread'] = np.NaN
        self.df['exit_spread'] = np.NaN
        self.df['profit_stock1'] = 0
        self.df['profit_stock2'] = 0
        self.df['ugl'] = 0
        self.df['rgl'] = 0
        self.df['spread_profit'] = 0
        self.df['capital'] = np.NaN # or market value
        # loop through all rows
        open_position = 0
        self.df.loc[0, 'capital'] = self.INITIAL_CAPITAL
        df_txn = pd.DataFrame()
        for i in range(1, self.df.shape[0]):
            # if position is 0 then check signal
            stock1_price = self.df[self.stock1][i]
            stock2_price = self.df[self.stock2][i]
            
            if open_position == 0: 
                capital = self.df.loc[i-1, 'capital']
                if self.df['enter_long_spread'][i]:
                    open_position = 1
                    close_position_flag = False
                    self.df.loc[i, 'position'] = open_position
                    #record entry price
                    self.df.loc[i, 'entry_spread'] = self.df['spread'][i]
                    entry_spread = self.df['spread'][i]
                    stock1_cost = stock1_price
                    stock2_cost = stock2_price
                    #calculate number of shares, use all capital to buy stock1 and stock2 using beta, assuming that short selling is allowed and have to put up margin = 100%
                    # stock1_price = beta * stock2_price >> OLS
                    # for each 1 share of stock1, buy beta shares of stock2
                    # capital = n * (stock1_price + stock1_price * beta)
                    n = capital / (stock1_price + stock2_price*beta)
                    m = n * beta
                    self.df.loc[i, self.stock1+'_position'] = n
                    self.df.loc[i, self.stock2+'_position'] = -m
                    self.df.loc[i, 'capital'] = n*stock1_price + m*stock2_price
                    txn = {'date': self.df['date'][i], 'position_type': 'long', 'entry_spread': entry_spread, 
                            self.stock1: self.df[self.stock1][i], self.stock2: self.df[self.stock2][i],
                            self.stock1+'_shares': n, self.stock2+'_shares': -m, 'capital': n*stock1_price + m*stock2_price}
                    new_txn = pd.DataFrame(txn, index=[0])
                    df_txn = pd.concat([df_txn, new_txn], ignore_index=True)

                elif self.df['enter_short_spread'][i]: 
                    open_position = -1
                    close_position_flag = False
                    self.df.loc[i, 'position'] = open_position
                    self.df.loc[i, 'entry_spread'] = -self.df['spread'][i]  
                    entry_spread = self.df['spread'][i]
                    stock1_cost = stock1_price
                    stock2_cost = stock2_price
                    n = capital / (stock1_price + stock2_price*beta)
                    m = n * beta
                    self.df.loc[i, self.stock1+'_position'] = -n
                    self.df.loc[i, self.stock2+'_position'] = m
                    self.df.loc[i, 'capital'] = n*stock1_price + m*stock2_price
                    txn = {'date': self.df['date'][i], 'position_type': 'short', 'entry_spread': entry_spread, 
                            self.stock1: self.df[self.stock1][i], self.stock2: self.df[self.stock2][i],
                            self.stock1+'_shares': -n, self.stock2+'_shares': m, 'capital': n*stock1_price + m*stock2_price}
                    new_txn = pd.DataFrame(txn, index=[0])
                    df_txn = pd.concat([df_txn, new_txn], ignore_index=True)
                else:
                    self.df.loc[i, 'capital'] = capital

            # if already have position then check exit signal
            else:
                # if long spread position and exit signal is true then close position
                if (open_position == 1) & (self.df['close_long_spread'][i]):
                    self.df.loc[i, 'exit_spread'] = -self.df['spread'][i]
                    self.df.loc[i, 'spread_profit'] = self.df['spread'][i] - entry_spread
                    close_position_flag = True                    
                    txn = {'date': self.df['date'][i], 'position_type': 'exit long', 'exit_spread': self.df['spread'][i], self.stock1: self.df[self.stock1][i], self.stock2: self.df[self.stock2][i]}

                elif (open_position == -1) & (self.df['close_short_spread'][i]):
                    self.df.loc[i, 'exit_spread'] = self.df['spread'][i]
                    self.df.loc[i, 'spread_profit'] = entry_spread - self.df['spread'][i]
                    close_position_flag = True     
                    txn = {'date': self.df['date'][i], 'position_type': 'exit short', 'exit_spread': self.df['spread'][i], self.stock1: self.df[self.stock1][i], self.stock2: self.df[self.stock2][i]}               
                    
                elif (open_position == 1) & (self.df['cutloss_long_spread'][i]):
                    self.df.loc[i, 'exit_spread'] = -self.df['spread'][i]
                    self.df.loc[i, 'spread_profit'] = self.df['spread'][i] - entry_spread
                    close_position_flag = True     
                    txn = {'date': self.df['date'][i], 'position_type': 'cutloss long', 'exit_spread': self.df['spread'][i], self.stock1: self.df[self.stock1][i], self.stock2: self.df[self.stock2][i]}
                    
                elif (open_position == -1) & (self.df['cutloss_short_spread'][i]):
                    self.df.loc[i, 'exit_spread'] = self.df['spread'][i]
                    self.df.loc[i, 'spread_profit'] = entry_spread - self.df['spread'][i]
                    close_position_flag = True    
                    txn = {'date': self.df['date'][i], 'position_type': 'cutloss short', 'exit_spread': self.df['spread'][i], self.stock1: self.df[self.stock1][i], self.stock2: self.df[self.stock2][i]}
                    
                if close_position_flag == True:
                    self.df.loc[i, 'position'] = 0
                    entry_spread = np.NaN
                    self.df.loc[i, 'rgl'] = (stock1_price - stock1_cost) * open_position * n - (stock2_price - stock2_cost) * open_position * m
                    self.df.loc[i, 'capital'] = self.df['capital'][i-1] +  self.df['rgl'][i] - self.df['ugl'][i-1]
                    txn['capital'] = self.df.loc[i, 'capital']
                    txn['rgl'] = self.df.loc[i, 'rgl']
                    new_txn = pd.DataFrame(txn, index=[0])
                    df_txn = pd.concat([df_txn, new_txn], ignore_index=True)
                    open_position = 0
                else:
                    # if no exit signal then keep position
                    self.df.loc[i, 'position'] = open_position
                    self.df.loc[i, self.stock1+'_position'] = self.df[self.stock1+'_position'][i-1]
                    self.df.loc[i, self.stock2+'_position'] = self.df[self.stock2+'_position'][i-1]
                    #mark to market
                    self.df.loc[i, 'ugl'] = (stock1_price - stock1_cost) * open_position * n - (stock2_price - stock2_cost) * open_position * m
                    self.df.loc[i, 'capital'] = self.df['capital'][i-1] +  self.df['ugl'].diff()[i]


            self.df_txn = df_txn
            self.df['is_just_long'] = (self.df['position'] == 1) & (self.df['position'].shift(1) == 0)
            self.df['is_just_short'] = (self.df['position'] == -1) & (self.df['position'].shift(1) == 0)
            self.df['is_just_exit_long'] = (self.df['position'] == 0) & (self.df['position'].shift(1) == 1)
            self.df['is_just_exit_short'] = (self.df['position'] == 0) & (self.df['position'].shift(1) == -1)
    # cross over function, accept series of zscore and SD
    def _cross_over(self, zscore: pd.Series, SD: float, cross_up: bool = True) -> pd.Series:
        if cross_up:
            # return (zscore > SD) & (zscore.shift(1) < SD)
            return (zscore > SD)
        else:
            # return (zscore < SD) & (zscore.shift(1) > SD)
            return (zscore < SD)