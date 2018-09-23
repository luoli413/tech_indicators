from StockData import StockData
import os
path = os.getcwd()
data_path = os.path.join(path+'\\data\\')

if __name__=='__main__':
    st = StockData(data_path)
    stock_pool = ['000021','000022','000003','000002','300393','600626']
    st.read(stock_pool)
    df = st.get_data_by_symbol('000021','20100101','20100901')
    print(df.head())
    df = st.get_adate_symbol('20100104',['000021','000003','000002'])
    print(df.head())
    df = st.get_data_by_field('open',['000021','000003','000002','300393'])
    print(df.head())
    # It's an adjustment price function and you only need call it once
    # before computing all indicators of certain stock you need.
    # Because if store = True, then it would be stored in self.
    st.adjust_data('000021', True, 'forwards')
    st.plot('000021', 'close')
    df = st.resample('000021',5)
    df = st.resample('300393',20)
    print(df.head())
    for window in [5, 20, 60]:
        st.moving_average('000021', 'close',True, window = window,)
        st.atr('000003', window = window)
        st.ema('000021', 'close', alpha = 0.5, window = window)
        st.dif('000021', 'close', alpha = 0.5, long = window + 13, short = window)
        st.adjust_data('300393', True, 'forwards')
        st.macd('300393', 'close',store = True, alpha = 0.5, long = 26, short = 12,\
                macd_alpha = 0.65, macd_window = window)
        st.rsi('000021','close', True, window = window)
        st.plot('000021', 'ma')
        st.plot('300393', 'macd', 'bar')
        st.plot('000021','rsi')
