import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.optimize import minimize
from scipy.stats import t
import matplotlib.pyplot as plt

data = pd.read_csv('problem2.csv')

# a
# Fit the data using OLS
X = sm.add_constant(data['x']) 
y = data['y']
ols_model = sm.OLS(y, X).fit()

# Fit the data using MLE assuming normality of errors
def normal_log_likelihood(params, X, y):
    beta = params[:-1]  
    sigma = params[-1]  
    y_hat = X @ beta 
    residuals = y - y_hat
    s2 = sigma**2  
    ll = -0.5 * np.sum(np.log(2 * np.pi * s2) + (residuals**2) / s2)
    return -ll

# Minimize the negative log likelihood
initial_guess = np.array([0, 0, 1])
res = minimize(normal_log_likelihood, initial_guess, args=(X, y))

# Compare the OLS and MLE estimates
beta_ols = ols_model.params
sigma_ols  = np.std(ols_model.resid)
beta_mle = res.x[:-1]
sigma_mle = res.x[-1]

print(f'OLS beta: {beta_ols}, sigma: {sigma_ols}')
print(f'MLE beta: {beta_mle}, sigma: {sigma_mle}')


# b
# Log-likelihood function for T-distribution
def t_log_likelihood(params, X, y):
    beta = params[:-2]
    df = params[-1]  
    y_hat = X @ beta
    residuals = y - y_hat
    sigma = params[-2]
    ll = np.sum(t.logpdf(x=residuals / sigma, df=df, loc=0, scale=sigma))
    return -ll

# Minimize the negative log likelihood with an initial guess
initial_guess = np.array([0, 0, 1, 10])  
res_t = minimize(t_log_likelihood, initial_guess, args=(X, y))

# Extract the MLE estimates for the T-distribution assumption
beta_mle_t = res_t.x[:-2]
sigma_mle_t = res_t.x[-2]
df_mle = res_t.x[-1]  

# Now compare results
y_hat = X @ beta_mle
R_squared_mle = 1 - (np.sum((y - y_hat)**2) / np.sum((y - np.mean(y))**2))
y_hat_t = X @ beta_mle_t
R_squared_mle_t = 1 - (np.sum((y - y_hat_t)**2) / np.sum((y - np.mean(y))**2))
print(f'MLE under normality assumption beta: {beta_mle}, sigma: {sigma_mle}, R^2: {R_squared_mle}')
print(f'MLE under T-distribution assumption beta: {beta_mle_t}, sigma: {sigma_mle_t}, degrees of freedom: {df_mle}, R^2: {R_squared_mle_t}')


# c

data_x1 = pd.read_csv('problem2_x1.csv')
data_x = pd.read_csv('problem2_x.csv')

x1 = data_x1['x1'].values
x2 = data_x[['x1', 'x2']].values
mean_x1 = np.mean(x1)
var_x1 = np.var(x1)

# Fit a multivariate normal distribution to X
mean_x = np.mean(x2, axis=0)
cov_x = np.cov(x2, rowvar=False)

# Extract the elements 
mean_x1_x2 = mean_x[0]
mean_x2 = mean_x[1]
var_x1_x2 = cov_x[0, 0]
cov_x1_x2 = cov_x[0, 1]
var_x2 = cov_x[1, 1]

# Calculate the parameters for the conditional distribution of X2 given X1
cond_mean_x2_given_x1 = lambda x1: mean_x2 + cov_x1_x2 / var_x1_x2 * (x1 - mean_x1_x2)
cond_var_x2_given_x1 = var_x2 - cov_x1_x2**2 / var_x1_x2

# Calculate the expected value and 95% confidence interval for each observed value of x1
expected_x2 = cond_mean_x2_given_x1(x1)
z_score_95 = 1.96 
margin_error = z_score_95 * np.sqrt(cond_var_x2_given_x1)

# Plotting
plt.figure(figsize=(10, 6))
plt.errorbar(x1, expected_x2, yerr=margin_error, fmt='o', label='Expected X2 with 95% CI')
plt.xlabel('X1')
plt.ylabel('Expected X2')
plt.title('Conditional Expectation of X2 given X1 with 95% CI')
plt.legend()
plt.grid(True)
plt.show()