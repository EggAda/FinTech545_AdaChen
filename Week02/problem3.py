import pandas as pd
from statsmodels.tsa.arima.model import ARIMA


data = pd.read_csv('problem3.csv')
# Function to fit ARIMA model and print out the AIC
def fit_arima(data, order):
    model = ARIMA(data, order=order)
    model_fit = model.fit()
    print(f'ARIMA{order} AIC: {model_fit.aic}')
    return model_fit.aic

# Fit AR(1) through AR(3)
ar_aics = [fit_arima(data, (p,0,0)) for p in range(1, 4)]

# Fit MA(1) through MA(3)
ma_aics = [fit_arima(data, (0,0,q)) for q in range(1, 4)]

# Identify the model with the lowest AIC
min_aic = min(ar_aics + ma_aics)
best_fit = 'AR' if min_aic in ar_aics else 'MA'
best_order = (ar_aics + ma_aics).index(min_aic) % 3 + 1

print(f'The best model is {best_fit}({best_order}) with AIC: {min_aic}')
