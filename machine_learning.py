import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet

# World Disaster Risk Dataset
# 종속변수
# WRI: World Risk Score of the Region
# 독립변수
# Exposure: Risk/exposure to natural hazards such as earthquakes, hurricanes, floods, droughts, and sea level rise.
# Vulnerability: Vulnerability depending on infrastructure, nutrition, housing situation, and economic framework conditions.
# Susceptibility: Susceptibility depending on infrastructure, nutrition, housing situation, and economic framework conditions.
# Lack of Coping Capabilities: Coping capacities in dependence of governance, preparedness and early warning, medical care, and social and material security.
# Lack of Adaptive Capacities: Adaptive capacities related to coming natural events, climate change, and other challenges.

# wolrd_risk_index 파일을 불러와 확인한다.
data = pd.read_csv("./data/world_risk_index.csv")
print(data.head())
# 독립변수와 종속변수로 사용하지 않는 데이터는 삭제한다.
data_1 = data.drop(['Region', 'Year', 'WRI Category', 'Exposure Category', 'Vulnerability Category', 'Susceptibility Category'], axis=1)
print(data_1.head())
# 데이터 안에 있는 결측치를 확인 후 제거한다.
print(data_1.isna().sum())
world_risk = data_1.dropna(axis=0)
print(world_risk.isna().sum())

# 각각의 밀도 그래프를 그려 데이터를 시각화한다.
world_risk.plot(kind='density', figsize=(15, 12), subplots=True, layout = (2,3), sharex=False)
plt.savefig('./data/world_risk.png')

# 첫 번째 열만 종속변수, 나머지는 독립변수로 설정
X = world_risk.iloc[2:, 2:].values # 독립변수
Y = world_risk.iloc[2:, 1].values  # 종속변수

# 선형회귀 학습
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=41)
model = ElasticNet(alpha=1.0, l1_ratio=0.5)
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)
print(Y_pred)

# 평균제곱오차 계산
mse = mean_squared_error(Y_test, Y_pred)
print(f'Mean Squared Error: {mse}')
# K-fold Cross Validation으로 학습 모델 성능 평가
kfold = KFold(n_splits=10, random_state=5, shuffle=True)
results = cross_val_score(model, X, Y, scoring='neg_mean_squared_error', cv=kfold)
print(results.mean())
