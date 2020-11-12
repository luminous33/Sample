from django.shortcuts import render
import io
import urllib, base64
import datetime
from mpl_finance import candlestick_ohlc
import matplotlib.dates as mdates
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import numpy as np
import matplotlib.pyplot as plt
import Analyzer
from .models import CompanyInfo
from .models import DailyPrice
from .models import Merge
from bs4 import BeautifulSoup
from urllib.request import urlopen
# import DualMomentum
import json
import pandas as pd
import pymysql
from datetime import datetime
from datetime import timedelta
from dateutil.relativedelta import relativedelta
import re

def introbol(request):
    return render(request, 'introbol.html')

def introtriple(request):
    return render(request, 'introtriple.html')

def introdual(request):
    return render(request, 'introdual.html')

def introdeep(request):
    return render(request, 'introdeep.html')

class DualMomentum:
    def __init__(self):
        """생성자: KRX 종목코드(codes)를 구하기 위한 MarkgetDB 객체 생성"""
        self.mk = Analyzer.MarketDB()

    def get_rltv_momentum(self, start_date, end_date, stock_count):
        """특정 기간 동안 수익률이 제일 높았던 stock_count 개의 종목들 (상대 모멘텀)
            - start_date  : 상대 모멘텀을 구할 시작일자 ('2020-01-01')
            - end_date    : 상대 모멘텀을 구할 종료일자 ('2020-12-31')
            - stock_count : 상대 모멘텀을 구할 종목수
        """
        connection = pymysql.connect(host='localhost', port=3306,
                                     db='investar', user='root', passwd='1111', autocommit=True)
        cursor = connection.cursor()

        # 사용자가 입력한 시작일자를 DB에서 조회되는 일자로 보정
        sql = f"select max(date) from daily_price where date <= '{start_date}'"
        cursor.execute(sql)
        result = cursor.fetchone()
        if (result[0] is None):
            print("start_date : {} -> returned None".format(sql))
            return
        start_date = result[0].strftime('%Y-%m-%d')

        # 사용자가 입력한 종료일자를 DB에서 조회되는 일자로 보정
        sql = f"select max(date) from daily_price where date <= '{end_date}'"
        cursor.execute(sql)
        result = cursor.fetchone()
        if (result[0] is None):
            print("end_date : {} -> returned None".format(sql))
            return
        end_date = result[0].strftime('%Y-%m-%d')

        # KRX 종목별 수익률을 구해서 2차원 리스트 형태로 추가
        rows = []
        columns = ['code', 'company', 'old_price', 'new_price', 'returns']
        for _, code in enumerate(self.mk.codes):
            sql = f"select close from daily_price " \
                  f"where code='{code}' and date='{start_date}'"
            cursor.execute(sql)
            result = cursor.fetchone()
            if (result is None):
                continue
            old_price = int(result[0])
            sql = f"select close from daily_price " \
                  f"where code='{code}' and date='{end_date}'"
            cursor.execute(sql)
            result = cursor.fetchone()
            if (result is None):
                continue
            new_price = int(result[0])
            returns = (new_price / old_price - 1) * 100
            rows.append([code, self.mk.codes[code], old_price, new_price,
                         returns])

        # 상대 모멘텀 데이터프레임을 생성한 후 수익률순으로 출력
        df = pd.DataFrame(rows, columns=columns)
        df = df[['code', 'company', 'old_price', 'new_price', 'returns']]
        df = df.sort_values(by='returns', ascending=False)
        df = df.head(stock_count)
        df.index = pd.Index(range(stock_count))
        connection.close()
        print(df)
        print(f"\nRelative momentum ({start_date} ~ {end_date}) : " \
              f"{df['returns'].mean():.2f}% \n")
        return df

    def get_abs_momentum(self, rltv_momentum, start_date, end_date):
        """특정 기간 동안 상대 모멘텀에 투자했을 때의 평균 수익률 (절대 모멘텀)
            - rltv_momentum : get_rltv_momentum() 함수의 리턴값 (상대 모멘텀)
            - start_date    : 절대 모멘텀을 구할 매수일 ('2020-01-01')
            - end_date      : 절대 모멘텀을 구할 매도일 ('2020-12-31')
        """
        stockList = list(rltv_momentum['code'])
        connection = pymysql.connect(host='localhost', port=3306,
                                     db='investar', user='root', passwd='1111', autocommit=True)
        cursor = connection.cursor()

        # 사용자가 입력한 매수일을 DB에서 조회되는 일자로 변경
        sql = f"select max(date) from daily_price " \
              f"where date <= '{start_date}'"
        cursor.execute(sql)
        result = cursor.fetchone()
        if (result[0] is None):
            print("{} -> returned None".format(sql))
            return
        start_date = result[0].strftime('%Y-%m-%d')

        # 사용자가 입력한 매도일을 DB에서 조회되는 일자로 변경
        sql = f"select max(date) from daily_price " \
              f"where date <= '{end_date}'"
        cursor.execute(sql)
        result = cursor.fetchone()
        if (result[0] is None):
            print("{} -> returned None".format(sql))
            return
        end_date = result[0].strftime('%Y-%m-%d')

        # 상대 모멘텀의 종목별 수익률을 구해서 2차원 리스트 형태로 추가
        rows = []
        columns = ['code', 'company', 'old_price', 'new_price', 'returns']
        for _, code in enumerate(stockList):
            sql = f"select close from daily_price " \
                  f"where code='{code}' and date='{start_date}'"
            cursor.execute(sql)
            result = cursor.fetchone()
            if (result is None):
                continue
            old_price = int(result[0])
            sql = f"select close from daily_price " \
                  f"where code='{code}' and date='{end_date}'"
            cursor.execute(sql)
            result = cursor.fetchone()
            if (result is None):
                continue
            new_price = int(result[0])
            returns = (new_price / old_price - 1) * 100
            rows.append([code, self.mk.codes[code], old_price, new_price,
                         returns])

        # 절대 모멘텀 데이터프레임을 생성한 후 수익률순으로 출력
        df = pd.DataFrame(rows, columns=columns)
        df = df[['code', 'company', 'old_price', 'new_price', 'returns']]
        df = df.sort_values(by='returns', ascending=False)
        connection.close()
        print(df)
        print(f"\nAbasolute momentum ({start_date} ~ {end_date}) : " \
              f"{df['returns'].mean():.2f}%")
        return df


def homepage(request):
    url = 'https://finance.naver.com/sise/'
    with urlopen(url) as doc:
        html = BeautifulSoup(doc, 'lxml')

        kospi = html.find('span', id='KOSPI_now').text
        kospiChange = html.find('span', id='KOSPI_change').text
        kospiChange = kospiChange[:-3]  # 하락인데 상승으로 받아와서 string 뒷부분 자름..

        kosdak = html.find('span', id='KOSDAQ_now').text
        kosdakChange = html.find('span', id='KOSDAQ_change').text
        kosdakChange = kosdakChange[:-3]  # 하락인데 상승으로 받아와서 string 뒷부분 자름..

        kospi200 = html.find('span', id='KPI200_now').text
        kospi200Change = html.find('span', id='KPI200_change').text
        kospi200Change = kospi200Change[:-3]  # 하락인데 상승으로 받아와서 string 뒷부분 자름..

    if kospiChange.find('+') > 0:
        color1 = "red"
    else:
        color1 = "blue"

    if kosdakChange.find('+') > 0:
        color2 = "red"
    else:
        color2 = "blue"

    if kospi200Change.find('+') > 0:
        color3 = "red"
    else:
        color3 = "blue"

    # 최종 context 부분
    context = {
        'color1': color1,
        'color2': color2,
        'color3': color3,
        'kospi': kospi,
        'kospiChange': kospiChange,
        'kosdak': kosdak,
        'kosdakChange': kosdakChange,
        'kospi200': kospi200,
        'kospi200Change': kospi200Change,
        # 'd': datarm,
        # 'd2': dataam

    }

    try:

        dualcount = int(request.GET['dualcount'])
        dualstart = str(request.GET['dualstart'])
        dualend = str(request.GET['dualend'])

        dm = DualMomentum()
        rm = dm.get_rltv_momentum(dualstart, dualend, dualcount)
        json_records = rm.reset_index().to_json(orient='records')
        datarm = []
        datarm = json.loads(json_records)

        # 절대모멘텀 시작일자
        dualstart2 = dualend

        dualend2 = datetime.strptime(dualstart2, '%Y-%m-%d')
        dualend2 = dualend2 + relativedelta(months=3)
        dualend2 = str(dualend2)

        am = dm.get_abs_momentum(rm, dualstart2, dualend2)
        json_records = am.reset_index().to_json(orient='records')
        dataam = []
        dataam = json.loads(json_records)

        context['d'] = datarm
        context['d2'] = dataam
        context['dualstart'] = str(dualstart)
        context['dualend'] = str(dualend)
        context['dualstart2'] = str(dualstart2)
        context['dualend2'] = str(dualend2[:-9])

        return render(request, 'homepage.html', context)

    except:

        # default 값
        dualcount = 10
        dualstart = '2020-04-01'
        dualend = '2020-06-30'

        dm = DualMomentum()
        rm = dm.get_rltv_momentum(dualstart, dualend, dualcount)
        json_records = rm.reset_index().to_json(orient='records')
        datarm = []
        datarm = json.loads(json_records)

        # 절대모멘텀 시작일자
        dualstart2 = dualend

        dualend2 = datetime.strptime(dualstart2, '%Y-%m-%d')
        dualend2 = dualend2 + relativedelta(months=3)
        dualend2 = str(dualend2)

        am = dm.get_abs_momentum(rm, dualstart2, dualend2)
        json_records = am.reset_index().to_json(orient='records')
        dataam = []
        dataam = json.loads(json_records)

        context['d'] = datarm
        context['d2'] = dataam
        context['dualstart'] = str(dualstart)
        context['dualend'] = str(dualend)
        context['dualstart2'] = str(dualstart2)
        context['dualend2'] = str(dualend2[:-9])

        return render(request, 'homepage.html', context)

    return render(request, 'homepage.html', context)


def introduce(request):
    return render(request, 'introduce.html')


def deep(request):
    name = request.GET['name']

    cur_price = Merge.objects.filter(date="2020-10-21", company=name).values("close")
    # cur_price = cur_price[0]
    cur_price = str(cur_price)
    cur_price = int(re.findall("\d+", cur_price)[0])

    mk = Analyzer.MarketDB()
    raw_df = mk.get_daily_price(name, '2019-03-01', '2020-09-01')

    window_size = 10
    data_size = 5

    def MinMaxScaler(data):
        """최솟값과 최댓값을 이용하여 0 ~ 1 값으로 변환"""
        numerator = data - np.min(data, 0)
        denominator = np.max(data, 0) - np.min(data, 0)
        # 0으로 나누기 에러가 발생하지 않도록 매우 작은 값(1e-7)을 더해서 나눔
        return numerator / (denominator + 1e-7)

    dfx = raw_df[['open', 'high', 'low', 'volume', 'close']]
    dfx = MinMaxScaler(dfx)  # 삼성전자 OHLVC정보를  MinMaxScaler()를 이용하여 0~1사이 값으로 변환
    dfy = dfx[['close']]  # dfx는 OHLVC가격정보.  dfy는 종가정보.

    x = dfx.values.tolist()
    y = dfy.values.tolist()

    # 데이터셋 준비하기
    data_x = []
    data_y = []
    for i in range(len(y) - window_size):
        _x = x[i: i + window_size]  # 다음 날 종가(i+windows_size)는 포함되지 않음
        _y = y[i + window_size]  # 다음 날 종가
        data_x.append(_x)
        data_y.append(_y)
    print(_x, "->", _y)

    # 훈련용데이터셋
    train_size = int(len(data_y) * 0.7)
    train_x = np.array(data_x[0: train_size])
    train_y = np.array(data_y[0: train_size])

    # 테스트용 데이터셋
    test_size = len(data_y) - train_size
    test_x = np.array(data_x[train_size: len(data_x)])
    test_y = np.array(data_y[train_size: len(data_y)])

    # 모델 생성
    model = Sequential()
    model.add(LSTM(units=10, activation='relu', return_sequences=True, input_shape=(window_size, data_size)))
    model.add(Dropout(0.1))
    model.add(LSTM(units=10, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(units=1))
    model.summary()

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(train_x, train_y, epochs=60, batch_size=30)
    pred_y = model.predict(test_x)

    # Visualising the results
    plt.figure()
    plt.plot(test_y, color='red', label='real stock price')
    plt.plot(pred_y, color='blue', label='predicted  stock price')
    plt.title('stock price prediction')
    plt.xlabel('time')
    plt.ylabel('stock price')
    plt.legend()
    fig = plt.gcf()
    # convert graph into dtring buffer and then we convert 64 bit code into image
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    string = base64.b64encode(buf.read())
    uri = urllib.parse.quote(string)
    price = int(raw_df.close[-1] * pred_y[-1] / dfy.close[-1])

    # context = {'data': uri, 'name': name, 'price': price, 'cur_price': cur_price}
    return render(request, 'deep.html', {'data': uri, 'name': name, 'price': price, 'cur_price': cur_price})


def bol2(request):
    try:

        name = request.GET['name']
        startdate = request.GET['startdate']
        mk = Analyzer.MarketDB()
        df = mk.get_daily_price(name, startdate)

        df['MA20'] = df['close'].rolling(window=20).mean()
        df['stddev'] = df['close'].rolling(window=20).std()
        df['upper'] = df['MA20'] + (df['stddev'] * 2)
        df['lower'] = df['MA20'] - (df['stddev'] * 2)
        df['PB'] = (df['close'] - df['lower']) / (df['upper'] - df['lower'])

        df['II'] = (2 * df['close'] - df['high'] - df['low']) / (df['high'] - df['low']) * df['volume']  # ①
        df['IIP21'] = df['II'].rolling(window=21).sum() / df['volume'].rolling(window=21).sum() * 100  # ②
        df = df.dropna()

        plt.figure(figsize=(9, 9))
        plt.subplot(3, 1, 1)
        plt.title('Bollinger Band(20 day, 2 std) - Reversals')
        plt.plot(df.index, df['close'], 'b', label='Close')
        plt.plot(df.index, df['upper'], 'r--', label='Upper band')
        plt.plot(df.index, df['MA20'], 'k--', label='Moving average 20')
        plt.plot(df.index, df['lower'], 'c--', label='Lower band')
        plt.fill_between(df.index, df['upper'], df['lower'], color='0.9')
        for i in range(0, len(df.close)):
            if df.PB.values[i] < 0.05 and df.IIP21.values[i] > 0:
                plt.plot(df.index.values[i], df.close.values[i], 'r^')
            elif df.PB.values[i] > 0.95 and df.IIP21.values[i] < 0:
                plt.plot(df.index.values[i], df.close.values[i], 'bv')

        plt.legend(loc='best')
        plt.subplot(3, 1, 2)
        plt.plot(df.index, df['PB'], 'b', label='%b')
        plt.grid(True)
        plt.legend(loc='best')

        plt.subplot(3, 1, 3)  # ③
        plt.bar(df.index, df['IIP21'], color='g', label='II% 21day')  # ④
        plt.grid(True)
        plt.legend(loc='best')

        fig = plt.gcf()
        # convert graph into dtring buffer and then we convert 64 bit code into image
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        string = base64.b64encode(buf.read())
        uri = urllib.parse.quote(string)

        df['TP'] = (df['high'] + df['low'] + df['close']) / 3
        df['PMF'] = 0
        df['NMF'] = 0
        for i in range(len(df.close) - 1):
            if df.TP.values[i] < df.TP.values[i + 1]:
                df.PMF.values[i + 1] = df.TP.values[i + 1] * df.volume.values[i + 1]
                df.NMF.values[i + 1] = 0
            else:
                df.NMF.values[i + 1] = df.TP.values[i + 1] * df.volume.values[i + 1]
                df.PMF.values[i + 1] = 0
        df['MFR'] = (df.PMF.rolling(window=10).sum() /
                     df.NMF.rolling(window=10).sum())
        df['MFI10'] = 100 - 100 / (1 + df['MFR'])
        df = df[19:]

        plt.figure(figsize=(9, 8))
        plt.subplot(2, 1, 1)
        plt.title('Bollinger Band(20 day, 2 std) - Trend Following')
        plt.plot(df.index, df['close'], color='#0000ff', label='Close')
        plt.plot(df.index, df['upper'], 'r--', label='Upper band')
        plt.plot(df.index, df['MA20'], 'k--', label='Moving average 20')
        plt.plot(df.index, df['lower'], 'c--', label='Lower band')
        plt.fill_between(df.index, df['upper'], df['lower'], color='0.9')
        for i in range(len(df.close)):
            if df.PB.values[i] > 0.8 and df.MFI10.values[i] > 80:
                plt.plot(df.index.values[i], df.close.values[i], 'r^')
            elif df.PB.values[i] < 0.2 and df.MFI10.values[i] < 20:
                plt.plot(df.index.values[i], df.close.values[i], 'bv')
        plt.legend(loc='best')

        plt.subplot(2, 1, 2)
        plt.plot(df.index, df['PB'] * 100, 'b', label='%B x 100')
        plt.plot(df.index, df['MFI10'], 'g--', label='MFI(10 day)')
        plt.yticks([-20, 0, 20, 40, 60, 80, 100, 120])
        for i in range(len(df.close)):
            if df.PB.values[i] > 0.8 and df.MFI10.values[i] > 80:
                plt.plot(df.index.values[i], 0, 'r^')
            elif df.PB.values[i] < 0.2 and df.MFI10.values[i] < 20:
                plt.plot(df.index.values[i], 0, 'bv')
        plt.grid(True)
        plt.legend(loc='best')

        fig2 = plt.gcf()
        # convert graph into dtring buffer and then we convert 64 bit code into image
        buf2 = io.BytesIO()
        fig2.savefig(buf2, format='png')
        buf2.seek(0)
        string2 = base64.b64encode(buf2.read())
        uri2 = urllib.parse.quote(string2)

        return render(request, 'bol2.html', {'data': uri, 'data2': uri2, 'name': name})

    except:

        name = "삼성전자"
        startdate = "2019-01-01"

        mk = Analyzer.MarketDB()
        df = mk.get_daily_price(name, startdate)

        df['MA20'] = df['close'].rolling(window=20).mean()
        df['stddev'] = df['close'].rolling(window=20).std()
        df['upper'] = df['MA20'] + (df['stddev'] * 2)
        df['lower'] = df['MA20'] - (df['stddev'] * 2)
        df['PB'] = (df['close'] - df['lower']) / (df['upper'] - df['lower'])

        df['II'] = (2 * df['close'] - df['high'] - df['low']) / (df['high'] - df['low']) * df['volume']  # ①
        df['IIP21'] = df['II'].rolling(window=21).sum() / df['volume'].rolling(window=21).sum() * 100  # ②
        df = df.dropna()

        plt.figure(figsize=(9, 9))
        plt.subplot(3, 1, 1)
        plt.title('Bollinger Band(20 day, 2 std) - Reversals')
        plt.plot(df.index, df['close'], 'b', label='Close')
        plt.plot(df.index, df['upper'], 'r--', label='Upper band')
        plt.plot(df.index, df['MA20'], 'k--', label='Moving average 20')
        plt.plot(df.index, df['lower'], 'c--', label='Lower band')
        plt.fill_between(df.index, df['upper'], df['lower'], color='0.9')
        for i in range(0, len(df.close)):
            if df.PB.values[i] < 0.05 and df.IIP21.values[i] > 0:
                plt.plot(df.index.values[i], df.close.values[i], 'r^')
            elif df.PB.values[i] > 0.95 and df.IIP21.values[i] < 0:
                plt.plot(df.index.values[i], df.close.values[i], 'bv')


        plt.legend(loc='best')
        plt.subplot(3, 1, 2)
        plt.plot(df.index, df['PB'], 'b', label='%b')
        plt.grid(True)
        plt.legend(loc='best')

        plt.subplot(3, 1, 3)  # ③
        plt.bar(df.index, df['IIP21'], color='g', label='II% 21day')  # ④
        plt.grid(True)
        plt.legend(loc='best')

        fig = plt.gcf()
        # convert graph into dtring buffer and then we convert 64 bit code into image
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        string = base64.b64encode(buf.read())
        uri = urllib.parse.quote(string)

        df['TP'] = (df['high'] + df['low'] + df['close']) / 3
        df['PMF'] = 0
        df['NMF'] = 0
        for i in range(len(df.close) - 1):
            if df.TP.values[i] < df.TP.values[i + 1]:
                df.PMF.values[i + 1] = df.TP.values[i + 1] * df.volume.values[i + 1]
                df.NMF.values[i + 1] = 0
            else:
                df.NMF.values[i + 1] = df.TP.values[i + 1] * df.volume.values[i + 1]
                df.PMF.values[i + 1] = 0
        df['MFR'] = (df.PMF.rolling(window=10).sum() /
                     df.NMF.rolling(window=10).sum())
        df['MFI10'] = 100 - 100 / (1 + df['MFR'])
        df = df[19:]

        plt.figure(figsize=(9, 9))
        plt.subplot(2, 1, 1)
        plt.title('Bollinger Band(20 day, 2 std) - Trend Following')
        plt.plot(df.index, df['close'], color='#0000ff', label='Close')
        plt.plot(df.index, df['upper'], 'r--', label='Upper band')
        plt.plot(df.index, df['MA20'], 'k--', label='Moving average 20')
        plt.plot(df.index, df['lower'], 'c--', label='Lower band')
        plt.fill_between(df.index, df['upper'], df['lower'], color='0.9')
        for i in range(len(df.close)):
            if df.PB.values[i] > 0.8 and df.MFI10.values[i] > 80:
                plt.plot(df.index.values[i], df.close.values[i], 'r^')
            elif df.PB.values[i] < 0.2 and df.MFI10.values[i] < 20:
                plt.plot(df.index.values[i], df.close.values[i], 'bv')
        plt.legend(loc='best')

        plt.subplot(2, 1, 2)
        plt.plot(df.index, df['PB'] * 100, 'b', label='%B x 100')
        plt.plot(df.index, df['MFI10'], 'g--', label='MFI(10 day)')
        plt.yticks([-20, 0, 20, 40, 60, 80, 100, 120])
        for i in range(len(df.close)):
            if df.PB.values[i] > 0.8 and df.MFI10.values[i] > 80:
                plt.plot(df.index.values[i], 0, 'r^')
            elif df.PB.values[i] < 0.2 and df.MFI10.values[i] < 20:
                plt.plot(df.index.values[i], 0, 'bv')
        plt.grid(True)
        plt.legend(loc='best')

        fig2 = plt.gcf()
        # convert graph into dtring buffer and then we convert 64 bit code into image
        buf2 = io.BytesIO()
        fig2.savefig(buf2, format='png')
        buf2.seek(0)
        string2 = base64.b64encode(buf2.read())
        uri2 = urllib.parse.quote(string2)

        return render(request, 'bol2.html', {'data': uri, 'data2': uri2, 'name': name})


def triple(request):
    try:
        name = request.GET['name']
        startdate = request.GET['startdate']
        mk = Analyzer.MarketDB()
        df = mk.get_daily_price(name, startdate)

        ema60 = df.close.ewm(span=60).mean()
        ema130 = df.close.ewm(span=130).mean()
        macd = ema60 - ema130
        signal = macd.ewm(span=45).mean()
        macdhist = macd - signal
        df = df.assign(ema130=ema130, ema60=ema60, macd=macd, signal=signal, macdhist=macdhist).dropna()

        df['number'] = df.index.map(mdates.date2num)
        ohlc = df[['number', 'open', 'high', 'low', 'close']]

        ndays_high = df.high.rolling(window=14, min_periods=1).max()
        ndays_low = df.low.rolling(window=14, min_periods=1).min()

        fast_k = (df.close - ndays_low) / (ndays_high - ndays_low) * 100
        slow_d = fast_k.rolling(window=3).mean()
        df = df.assign(fast_k=fast_k, slow_d=slow_d).dropna()

        plt.figure(figsize=(9, 9))
        p1 = plt.subplot(3, 1, 1)
        plt.title('Triple Screen Trading')
        plt.grid(True)
        candlestick_ohlc(p1, ohlc.values, width=.6, colorup='red', colordown='blue')
        p1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.plot(df.number, df['ema130'], color='c', label='EMA130')
        for i in range(1, len(df.close)):
            if df.ema130.values[i - 1] < df.ema130.values[i] and df.slow_d.values[i - 1] >= 20 and df.slow_d.values[
                i] < 20:
                plt.plot(df.number.values[i], 250000, 'r^')
            elif df.ema130.values[i - 1] > df.ema130.values[i] and df.slow_d.values[i - 1] <= 80 and df.slow_d.values[
                i] > 80:
                plt.plot(df.number.values[i], 250000, 'bv')
        plt.legend(loc='best')

        p2 = plt.subplot(3, 1, 2)
        plt.grid(True)
        p2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.bar(df.number, df['macdhist'], color='m', label='MACD-Hist')
        plt.plot(df.number, df['macd'], color='b', label='MACD')
        plt.plot(df.number, df['signal'], 'g--', label='MACD-Signal')
        plt.legend(loc='best')

        p3 = plt.subplot(3, 1, 3)
        plt.grid(True)
        p3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.plot(df.number, df['fast_k'], color='c', label='%K')
        plt.plot(df.number, df['slow_d'], color='k', label='%D')
        plt.yticks([0, 20, 80, 100])
        plt.legend(loc='best')
        fig = plt.gcf()
        # convert graph into dtring buffer and then we convert 64 bit code into image
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        string = base64.b64encode(buf.read())
        uri = urllib.parse.quote(string)
        return render(request, 'triple.html', {'data': uri, 'name': name})

    except:

        name = "삼성전자"
        startdate = "2019-01-01"

        mk = Analyzer.MarketDB()
        df = mk.get_daily_price(name, startdate)

        ema60 = df.close.ewm(span=60).mean()
        ema130 = df.close.ewm(span=130).mean()
        macd = ema60 - ema130
        signal = macd.ewm(span=45).mean()
        macdhist = macd - signal
        df = df.assign(ema130=ema130, ema60=ema60, macd=macd, signal=signal, macdhist=macdhist).dropna()

        df['number'] = df.index.map(mdates.date2num)
        ohlc = df[['number', 'open', 'high', 'low', 'close']]

        ndays_high = df.high.rolling(window=14, min_periods=1).max()
        ndays_low = df.low.rolling(window=14, min_periods=1).min()

        fast_k = (df.close - ndays_low) / (ndays_high - ndays_low) * 100
        slow_d = fast_k.rolling(window=3).mean()
        df = df.assign(fast_k=fast_k, slow_d=slow_d).dropna()

        plt.figure(figsize=(9, 9))
        p1 = plt.subplot(3, 1, 1)
        plt.title('Triple Screen Trading')
        plt.grid(True)
        candlestick_ohlc(p1, ohlc.values, width=.6, colorup='red', colordown='blue')
        p1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.plot(df.number, df['ema130'], color='c', label='EMA130')
        for i in range(1, len(df.close)):
            if df.ema130.values[i - 1] < df.ema130.values[i] and df.slow_d.values[i - 1] >= 20 and df.slow_d.values[
                i] < 20:
                plt.plot(df.number.values[i], 250000, 'r^')
            elif df.ema130.values[i - 1] > df.ema130.values[i] and df.slow_d.values[i - 1] <= 80 and df.slow_d.values[
                i] > 80:
                plt.plot(df.number.values[i], 250000, 'bv')
        plt.legend(loc='best')

        p2 = plt.subplot(3, 1, 2)
        plt.grid(True)
        p2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.bar(df.number, df['macdhist'], color='m', label='MACD-Hist')
        plt.plot(df.number, df['macd'], color='b', label='MACD')
        plt.plot(df.number, df['signal'], 'g--', label='MACD-Signal')
        plt.legend(loc='best')

        p3 = plt.subplot(3, 1, 3)
        plt.grid(True)
        p3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.plot(df.number, df['fast_k'], color='c', label='%K')
        plt.plot(df.number, df['slow_d'], color='k', label='%D')
        plt.yticks([0, 20, 80, 100])
        plt.legend(loc='best')
        fig = plt.gcf()
        # convert graph into dtring buffer and then we convert 64 bit code into image
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        string = base64.b64encode(buf.read())
        uri = urllib.parse.quote(string)
        return render(request, 'triple.html', {'data': uri, 'name': name})


# 주식검색
def search(request):
    merges = Merge.objects.filter(date="2020-10-21").values('company', 'code', 'date', 'open', 'high', 'low', 'close',
                                                            'diff', 'volume')
    context = {'merges': merges}
    return render(request, 'search.html', context)

