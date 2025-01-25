# Data_Processing

코드 구조 설명

1. FinanaceDataReader에서 불러온 모든 국장(2800개)의 시가, 고가, 저가, 종가

2. yfinance에서 불러온 모든 국장의 재무제표(roe, roa, dy)

3. 5일, 20일, 60일의 Moving Average, Volatillity, RSI 등의 기술적 지표

상기 변수 모두 결합해 XGBoost, RandomForest, SVM를 bagging한 앙상블 모델에 학습 및 최적화

학습 후 10일 후의 종목 종가 예측값, 각 예측의 평가지표인 RMSE, MAPE print

10일 후 예측값이 현재 종가보다 2%이상 높으면 상승, 2%~-2%면 중립, -2% 이하면 하락으로 함.

상기 항목은 json 파일로 저장함.

json 파일에 저장된 항목들 중 다음 조건을 하나라도 만족하는 종목만 추출해 df화
1. MACD, Bollinger, RSI 중 최소 하나가 매수이고 매도는 하나도 없음
2. 10일 후 예측결과가 하락이 아님
3. ROE > 0.1, ROA > 0.05, DY > 0.02

df에서 예측력이 65% 이상이면서 예상 종가 상승량이 가장 높은 10개의 종목의 정보에 기본적 분석 결과, 최근 정보(전일 종가, 현재 시가 등)를 추가해 다시 json으로 저장, 기술적 분석 그래프는 png로 저장
