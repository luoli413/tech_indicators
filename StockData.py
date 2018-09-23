import pandas as pd
import os
import warnings
import numpy as np
import datetime
warnings.resetwarnings()  # Maybe somebody else is messing with the warnings system.
warnings.filterwarnings('ignore')


class StockData():
    def __init__(self, path):
        self.path = path
        self.all = os.listdir(path)
        self.default_start = '19000101'


    def __format_date(self, symbol):
        '''

        :param symbol: a stockcode
        :return: df
        '''
        def field_name_transform(x):
            loc = x.find('S_DQ_')
            temp = x[loc + len('S_DQ_'):].lower()
            if temp == 'avgprice':
                temp = 'vwap'
            if temp == 'amount':
                temp = 'turnover'
            return temp

        vt = np.vectorize(field_name_transform)
        path = os.path.join(self.path + str(symbol))
        df = pd.read_csv(path, header=0)
        df = df.iloc[:, 3:-2]
        columns_names = list(vt(np.array(df.columns.values[2:])))
        columns = df.columns.values
        columns[2:] = columns_names
        columns[0] = 'date'
        df.columns = columns
        df['date'] = df['date'].apply(lambda x:datetime.datetime.\
            strptime(str(x), '%Y%m%d'))
        df.sort_values('date', inplace=True)
        return df


    def read(self, symbols):
        '''

        :param symbols: a list, don't contain .sz or.sh
        :return: a dict of all the dataframes needed
        '''

        self.data_dict = dict()
        if len(symbols) != 0:
            for symbol in symbols:
                symbol_list = [symbol + '.SZ.csv', symbol + '.SH.csv']
                flag = np.isin(symbol_list,self.all)
                symbol_new = np.array(symbol_list)[flag]
                if len(symbol_new) == 0:
                    print(symbol+' is not in database！')
                else:
                    df = self.__format_date(str(symbol_new[0]))
                    # df.set_index('TRADE_DT', drop = True, inplace = True)
                    # df.index.rename('date', inplace = True)
                    self.data_dict[symbol] = df
                    print(symbol + ' loaded!')
        self.universe = list(self.data_dict.keys())


    def get_data_by_symbol(self, symbol, start_date, end_date = -1,
                           fields = ['open', 'high', 'low', 'close']):
        '''

        :param symbol: s str of stockcode
        :param start_date: a str
        :param end_date: a str
        :param fields: default as above
        :return: a dataframe of all data indexed by date, close-close interval
        '''
        if symbol not in self.universe:
            print(symbol + ' need be loaded firstly!')
            return
        df = self.data_dict[symbol]
        start_date = datetime.datetime.strptime(start_date, '%Y%m%d')
        if end_date != -1:
            end_date = datetime.datetime.strptime(end_date, '%Y%m%d')
        start_min = df['date'].min()
        end_max = df['date'].max()
        if start_date < start_min:
            start_date = start_min
        if (end_date == -1) or (end_date > end_max):
            end_date = end_max
        if start_date > end_date:
            print('wrong dates!')
            return
        cols = ['date'] + fields
        df = df.loc[(df['date'] >= start_date) & \
                      (df['date'] <= end_date),cols]
        df.set_index('date', inplace = True)
        return df


    def get_adate_symbol(self, adate, symbols,
                         fields = ['open', 'high', 'low', 'close']):
        '''

        :param adate: a str of date, accurate searching
        :param symbols: a list of stockcode
        :param fields: default is open,high,low,close
        :return: a dataframe of all data indexed by symbols
        '''
        mask = np.isin(symbols,self.universe)
        if len(list(np.array(symbols)[~mask])) > 0:
            print(list(np.array(symbols)[~mask]),' need be loaded firstly!')

        symbol_ready = list(np.array(symbols)[mask])
        adate = datetime.datetime.strptime(adate,'%Y%m%d')
        s = 0
        if len(symbol_ready) <= 0:
            return
        else:
            for symbol in symbol_ready:
                temp = self.data_dict[symbol]
                temp = temp.loc[temp['date'] == adate, fields]
                temp['symbols'] = symbol
                if s == 0:
                    df = temp
                else:
                    df = pd.concat([df,temp], axis = 0)
                s += 1
        cols = ['symbols'] + fields
        df = df.loc[:, cols]
        df.set_index('symbols', inplace = True)
        return df


    def get_data_by_field(self, field, symbols):
        '''

        :param field: a field
        :param symbols:a lsit of stockcode
        :return:a dataframe of all field indexed by date,
        which is the longest range among symbols
        '''
        mask = np.isin(symbols, self.universe)
        if len(list(np.array(symbols)[~mask])) > 0:
            print(list(np.array(symbols)[~mask]),' need be loaded firstly!')
        symbol_ready = list(np.array(symbols)[mask])
        s = 0
        df = pd.DataFrame()
        if len(symbol_ready) <= 0:
            return
        else:
            for symbol in symbol_ready:
                temp = self.data_dict[symbol]
                if field in temp.columns.values:
                    temp_df = temp.loc[:,[field] + ['date']]
                    temp_df.rename(columns = {field : symbol}, inplace =True)
                    if s == 0:
                        df = temp_df
                    else:
                        if isinstance(df,pd.Series):
                            df = df.to_frame()
                        df = df.merge(temp_df, how = 'outer', on = 'date')
                    s += 1
            if not df.empty:
                df.set_index('date',inplace = True)
        return df


    def plot(self, symbol, field, kind = 'line'):
        '''

        :param symbol:
        :param field:
        :return: no return, just a graph
        '''
        line = ('open','high','close','low','vwap')
        bar = ('turnover','volume')
        if symbol not in self.universe:
            print(symbol + ' need be loaded firstly!')
            return
        temp = list(self.data_dict[symbol].columns)
        if field not in temp:
            print(field + ' is not in storage!')
            return
        if isinstance(symbol, str):
            symbol = [symbol]
        df = self.get_data_by_field(field, symbol)
        if (df.empty) | (df is None):
            print('No ' + field + ' data of '+ symbol[0])
            return
        import matplotlib.pyplot as plt
        if field in line:
            df.plot(title = field.upper() +' of '+ symbol[0], legend = False)
        elif field in bar:
            df.plot(title = field.upper()+' of '+ symbol[0],\
                    legend = False, kind = 'bar')
        else:
            df.plot(title = field.upper() + ' of ' + symbol[0], legend = False, kind = kind)

        plt.show()


    def adjust_data(self, symbol, store = False, method = 'backwards'):
        '''

        :param symbol: a stockcode
        :param store: whether to restore the adjprice data into self
        :param method: default is backwards
        :return:
        '''
        fields = ['open', 'high', 'low', 'close']
        if symbol not in self.universe:
            print(symbol + ' need be loaded firstly!')
            return
        prices = self.get_data_by_symbol(symbol, \
                                         start_date = self.default_start, fields = fields)
        af = self.get_data_by_field('adjfactor', [symbol])
        if method == 'forwards':
            constant = af.iat[-1, 0]
            af = af / constant
        adj_price = np.multiply(np.array(prices), np.array(af).reshape((af.shape[0],1)))
        if store:
            self.data_dict[symbol].loc[:, fields] = adj_price


    def resample(self,symbol, freq):
        '''

        :param symbol: stockcode
        :param freq: resample step
        :return: dataframe
        '''

        fields = ['open', 'high', 'low', 'close', 'turnover', 'volume',]
        if symbol in list(self.universe):

            price_df = self.get_data_by_symbol(symbol,start_date = self.default_start,fields = fields)
            rolling_df = price_df.rolling(freq, min_periods = freq,)
            price_df['price_early'] = rolling_df['open'].apply(lambda x: x[0], raw = True)
            price_df['price_last'] = rolling_df['close'].apply(lambda x: x[-1], raw =True)
            price_df['price_highest'] = rolling_df['high'].apply(lambda x: np.nanmax(x), raw = True)
            price_df['price_lowest'] = rolling_df['low'].apply(lambda x: np.nanmin(x), raw = True)
            price_df[['sum_t','sum_v']] = rolling_df[['turnover', 'volume',]].sum()
            price_df['vwap_ave'] = price_df['sum_v'] / price_df['sum_t']
            sampling = list(np.arange(0, price_df.shape[0], step = freq))
            df_sample = price_df[['price_early', 'price_last', 'price_highest', 'price_lowest', \
                                  'sum_t', 'sum_v', 'vwap_ave']].iloc[sampling, :]
            return df_sample


    def __to_series(self, df):

        if df.empty:
            # Empty dataframe, so convert to empty Series.
            result = pd.Series()
        elif df.shape == (1, 1):
            # DataFrame with one value, so convert to series with appropriate index.
            result = pd.Series(df.iat[0, 0], index = df.columns)
        elif df.shape[1] == 1:
            result = df.T.squeeze()
        else:
            result = df
        return result


    def moving_average(self, symbol, field, store = False, **kwargs):
        '''

        :param symbol:
        :param field:
        :param kwargs: parameter needed in computation: window, alpha, etc.
        :return: a series
        '''

        df = self.get_data_by_symbol(symbol,self.default_start, fields = [field])
        window = kwargs['window']
        df = df.rolling(window, min_periods = window).mean()
        if isinstance(df, pd.DataFrame):
            df = self.__to_series(df,)
        if store:
            self.data_dict[symbol]['ma'] = np.array(df)
        return df


    def ema(self, symbol, field,store =True, **kwargs):
        def func(x, alpha):
            ewa = np.zeros(x.shape)
            ewa[0] = x[0]
            for i in range(1, len(ewa)):
                ewa[i] = alpha * x[i] + (1 - alpha) * ewa[i - 1]
            return np.asscalar(ewa[-1])

        df = self.get_data_by_symbol(symbol, start_date = '19000101', fields = [field])
        if (df is None) or (df.empty):
            return
        alpha = kwargs['alpha']
        window = kwargs['window']
        df = df.rolling(window = window, min_periods = window).apply(func,args = (alpha,), raw = True)
        if isinstance(df, pd.DataFrame):
            df = self.__to_series(df)
        if store:
            self.data_dict[symbol]['ema'] = np.array(df)
        return df


    def atr(self, symbol,store = True, **kwargs):

        fields = ['preclose', 'high', 'low']
        window = kwargs['window']
        df = self.get_data_by_symbol(symbol,self.default_start, fields = fields)
        if (df is None) or (df.empty):
            return
        df['abs1'] = np.abs(df['high'] - df['preclose'])
        df['abs2'] = np.abs(df['preclose'] - df['low'])
        df['abs3'] = np.abs(df['high'] - df['low'])
        df_max = df.iloc[:,-3:].max(axis = 1)
        df = df_max.rolling(window, min_periods = window).mean()
        if isinstance(df,pd.DataFrame):
            df = self.__to_series(df)
        if store:
            self.data_dict[symbol]['atr'] = np.array(df)
        return df


    def dif(self, symbol, field, store = False, **kwargs):
        '''

        :param symbol: a stockcode
        :param field:
        :param store: whether to restore data in self
        :param kwargs: alpha, long_alpha, short_alpha, long, short,
        :return: a series
        '''
        long = kwargs['long']
        short = kwargs['short']
        mask = np.isin(['alpha', 'long_alpha', 'short_alpha'],list(kwargs.keys()))
        if mask[1]:
            long_alpha = kwargs['long_alpha']
        elif mask[0]:
            long_alpha = kwargs['alpha']
        if mask[2]:
            short_alpha = kwargs['short_alpha']
        elif mask[0]:
            short_alpha = kwargs['alpha']
        long_ema = self.ema(symbol, field, alpha = long_alpha, window = long)
        short_ema = self.ema(symbol, field, alpha = short_alpha, window = short)
        if (long_ema is None) or (long_ema.empty):
            return
        df = pd.concat([long_ema, short_ema], axis=1)
        df = df.iloc[:,1] - df.iloc[:,0] # short - long
        if isinstance(df, pd.DataFrame):
            df = self.__to_series(df)
        if store:
            self.data_dict[symbol]['dif'] = np.array(df)
        return df


    def macd(self, symbol, field, store = False, **kwargs):
        '''

        :param symbol: a stock code
        :param kwargs: alpha, long_alpha, short_alpha, long, short, macd_window, macd_alpha
        :return: a series
        '''

        mask = np.isin(['alpha', 'macd_alpha'], list(kwargs.keys()))
        if mask[1]:
            macd_alpha = kwargs['macd_alpha']
        elif mask[0]:
            macd_alpha = kwargs['alpha']
        macd_window = kwargs['macd_window']
        dif = self.dif(symbol, field, store = True, **kwargs)
        df = self.ema(symbol, 'dif', alpha = macd_alpha, window = macd_window)
        if store:
            self.data_dict[symbol]['macd'] = np.array(df)
        return df


    def rsi(self, symbol,field, store = False, **kwargs):
        def func(x):
            strong = np.nanmean(x[x > 0])
            weak = - np.nanmean(x[x < 0])
            return 100 - 100 / (1 + strong / weak)
        self.adjust_data(symbol, 'forwards')
        df = self.get_data_by_symbol(symbol, self.default_start, fields = [field])
        df = df - df.shift(1)
        window = kwargs['window'] + 1
        df = df.rolling(window, min_periods = window).apply(func, raw = True)
        if isinstance(df, pd.DataFrame):
            df = self.__to_series(df)
        if store:
            self.data_dict[symbol]['rsi'] = np.array(df)
        return df



