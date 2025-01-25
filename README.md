# Data_Processing
코드 구조 설명


---

FinanaceDataReader에서 모든 국장(2800개)의 시가, 고가, 저가, 종가 불러오기

yfinance에서 모든 국장의 재무제표(roe, roa, dy) 불러오기

5일, 20일, 60일의 Moving Average, Volatillity, RSI 등의 기술적 지표 불러오기

코드 구조 설명

FinanaceDataReader에서 모든 국장(2800개)의 시가, 고가, 저가, 종가 불러오기

yfinance에서 모든 국장의 재무제표(roe, roa, dy) 불러오기

5일, 20일, 60일의 Moving Average, Volatillity, RSI 등의 기술적 지표 불러오기

해당 값들을 모두 결합해 XGBoost, RandomForest, SVM를 bagging한 앙상블 모델에 학습 및 최적화

학습 후 10일 후의 종목 종가 예측값, 각 예측의 평가지표인 RMSE, MAPE print

10일 후 예측값이 현재 종가보다 2%이상 높으면 상승, 2%~-2%면 중립, -2% 이하면 하락으로 함.

상기 항목은 json 파일로 저장함.

json 파일에 저장된 항목들 중, MACD, Bollinger, RSI 중 최소 하나가 매수이고 매도는 하나도 없으며, 10일 후 예측결과가 하락이 아니고, ROE > 0.1, ROA > 0.05, DY > 0.02 중 하나라도 만족하는 종목만 따로 df로 제작.

해당 df에서 예측력이 65% 이상이면서 예상 종가 상승량이 가장 높은 10개의 종목의 정보에 기본적 분석 결과, 최근 정보(전일 종가, 현재 시가 등)를 추가해 다시 json으로 저장, 기술적 분석 그래프는 png로 저장

더블클릭 또는 Enter 키를 눌러 수정


[ ]
import FinanceDataReader as fdr
import pandas as pd
import mplfinance as mpf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import json
import os

# JSON file path
file_path = '/content/drive/MyDrive/filtered_stock_predictions.json'
save_dir = '/content/drive/MyDrive/stock_charts'

# Create save directory if it does not exist
os.makedirs(save_dir, exist_ok=True)

# Load JSON file
with open(file_path, 'r', encoding='utf-8') as f:
    stock_data = json.load(f)

# Set date range (1 year from today)
end_date = datetime.now()
start_date = end_date - timedelta(days=365)

# Function to plot stock chart
def plot_stock_chart(symbol, company_name, graph_num):
    # Fetch stock data
    df = fdr.DataReader(symbol, start_date, end_date)

    # Calculate moving averages
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA60'] = df['Close'].rolling(window=60).mean()
    df['MA120'] = df['Close'].rolling(window=120).mean()

    # Calculate Bollinger Bands (20-day, 3 standard deviations)
    df['BB_middle'] = df['Close'].rolling(window=20).mean()
    df['BB_std'] = df['Close'].rolling(window=20).std()
    df['BB_upper'] = df['BB_middle'] + 3 * df['BB_std']
    df['BB_lower'] = df['BB_middle'] - 3 * df['BB_std']

    # Select last month of data
    last_month = df.last('30D')

    # Set candlestick chart style
    mc = mpf.make_marketcolors(up='#ff3333', down='#3333ff', inherit=True)
    s = mpf.make_mpf_style(marketcolors=mc)

    # Define colors for moving averages and Bollinger Bands
    line_colors = {
        'MA5': 'purple', 'MA20': 'orange', 'MA60': 'green', 'MA120': 'brown',
        'BB_upper': 'yellow', 'BB_lower': 'yellow'
    }

    # Set up additional plots for moving averages and Bollinger Bands
    add_plots = [
        mpf.make_addplot(last_month[ma], color=color, width=0.5)
        for ma, color in line_colors.items() if ma.startswith('MA')
    ] + [
        mpf.make_addplot(last_month['BB_upper'], color='yellow', width=0.5),
        mpf.make_addplot(last_month['BB_lower'], color='yellow', width=0.5)
    ]

    # Set fill area between Bollinger Bands
    fill_between = dict(y1=last_month['BB_lower'].values, y2=last_month['BB_upper'].values, color='yellow', alpha=0.1)

    # Plot candlestick chart
    fig, axes = mpf.plot(last_month, type='candle', style=s,
                         addplot=add_plots, volume=True, figsize=(12, 8),
                         returnfig=True, panel_ratios=(3,1), fill_between=fill_between)

    # Manually create legend
    legend_elements = [
        Line2D([0], [0], color=color, lw=0.5, label=ma.replace('MA', 'Moving Average '))
        for ma, color in line_colors.items() if ma.startswith('MA')
    ] + [Line2D([0], [0], color='yellow', lw=0.5, label='Bollinger Bands (3σ)')]

    # Add legend
    axes[0].legend(handles=legend_elements, loc='upper left')

    # Save the chart to file
    save_path = os.path.join(save_dir, f'graph{graph_num}.png')
    plt.savefig(save_path, format='png', dpi=300)

    # Display the chart
    plt.show()

# Plot and save charts for the top 10 stocks
for i, stock in enumerate(stock_data[:10]):
    symbol = stock["symbol"]
    company_name = stock["company_name"]
    plot_stock_chart(symbol, company_name, i + 1)



[ ]
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# 모델 로드
model = joblib.load('/content/drive/MyDrive/stock_models/model_10d.joblib')

# 특성 중요도 추출
feature_importance = model.named_estimators_['xgb'].feature_importances_

# 특성 이름 가져오기 (모델 학습 시 사용한 특성 순서와 동일해야 함)
feature_names = ['Open', 'High', 'Low', 'Volume', 'MA5', 'MA20', 'MA60', 'Volatility', 'RSI',
                 'PER', 'ROA', 'ROE', 'DY', 'macd_signal', 'bollinger_signal', 'rsi_signal']

# 특성 중요도를 데이터프레임으로 변환
importance_df = pd.DataFrame({'feature': feature_names, 'importance': feature_importance})
importance_df = importance_df.sort_values('importance', ascending=False)

# 시각화
plt.figure(figsize=(10, 6))
plt.bar(importance_df['feature'], importance_df['importance'])
plt.xticks(rotation=90)
plt.title('Feature Importance')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.tight_layout()
plt.show()

# 상위 10개 특성 출력
print("Top 10 Most Important Features:")
print(importance_df.head(10))

그래프 코드 설명

최종적으로 추출한 상위 10개 종목의 그래프.

30일간의 캔들차트이며, 20일 이동평균과 3시그마를 사용한 볼린저 밴드, 5, 20, 60, 120일 이동평균을 추가함.

생성된 그래프는 '/content/drive/MyDrive/stock_charts'에 graph(n).png로 저장.

Colab 유료 제품 - 여기에서 계약 취소
