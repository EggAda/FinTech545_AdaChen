import numpy as np
import pandas as pd
import csv
from scipy.stats import skew, kurtosis, ttest_1samp, norm, t


def read_csv(filename):
    with open(filename, newline='') as csvfile:
        data_reader = csv.reader(csvfile, delimiter=',')
        next(data_reader)
        x_values = []
        for row in data_reader:
            x_values.append(float(row[0]))
    return x_values
        
def first4Moments_1(sample):
    n = len(sample)

    # Mean
    μ_hat = sum(sample) / n

    # Variance
    variance_hat = sum([(x - μ_hat)**2 for x in sample]) / (n - 1)

    # Skewness
    skew_hat = sum([(x - μ_hat)**3 for x in sample]) / (n * variance_hat**(3/2))

    # kurtosis
    kurt_hat = sum([(x - μ_hat)**4 for x in sample]) / (n * variance_hat**2)

    excessKurt_hat = kurt_hat - 3
    return μ_hat, variance_hat, skew_hat, excessKurt_hat

def first4Moments_2(sample):
    # Mean
    μ_hat = np.mean(sample)
  
    # Variance
    variance_hat = np.var(sample, ddof=1) 

    # Skewness
    skew_hat = skew(sample)
    
    # Excess kurtosis
    excessKurt_hat = kurtosis(sample) 
    
    return μ_hat, variance_hat, skew_hat, excessKurt_hat

# Print the results
x_values = read_csv('problem1.csv')
x_m_1, x_s2_1, x_sk_1, x_k_1 = first4Moments_1(x_values)
x_m_2, x_s2_2, x_sk_2, x_k_2 = first4Moments_2(x_values)
print("calculate x using normalized formula:", x_m_1, x_s2_1, x_sk_1, x_k_1)
print("calculate x using statistical package:", x_m_2, x_s2_2, x_sk_2, x_k_2)


# c
# Initialize parameters
num_samples = 100
num_iterations = 1000
alpha = 0.05  # Significance level
mean_values = []
variance_values = []
skewness_values = []
kurtosis_values = []

for _ in range(num_iterations):
    # Generate a sample of 100 standardized random normal values
    sample = np.random.randn(num_samples)
    # Calculate 4 moments values for the sample
    sample_mean = np.mean(sample)
    sample_variance = np.var(sample) 
    sample_skewness = skew(sample)
    sample_kurtosis = kurtosis(sample)  

    mean_values.append(sample_mean)
    variance_values.append(sample_variance)
    skewness_values.append(sample_skewness)
    kurtosis_values.append(sample_kurtosis)

results_df = pd.DataFrame({'Mean':mean_values, 'Variance': variance_values, 'Skewness': skewness_values, 'Kurtosis': kurtosis_values})

# Calculate mean and standard deviation
mean_mean = results_df['Mean'].mean()
std_mean = results_df['Mean'].std()
mean_variance = results_df['Variance'].mean()
std_variance = results_df['Variance'].std()
mean_skewness = results_df['Skewness'].mean()
std_skewness = results_df['Skewness'].std()
mean_kurtosis = results_df['Kurtosis'].mean()
std_kurtosis = results_df['Kurtosis'].std()

# Calculate T-statistics
# For mean
t_statistic_mean = mean_mean / (std_mean / np.sqrt(num_iterations))
# For variance
t_statistic_variance = mean_variance / (std_variance / np.sqrt(num_iterations))
# For skewness
t_statistic_skewness = mean_skewness / (std_skewness / np.sqrt(num_iterations))
# For kurtosis
t_statistic_kurtosis = mean_kurtosis / (std_kurtosis / np.sqrt(num_iterations))

# Calculate p-values for two-sided test
p_value_mean = 2 * (1 - t.cdf(np.abs(t_statistic_mean), df=num_iterations-1))
p_value_variance = 2 * (1 - t.cdf(np.abs(t_statistic_variance), df=num_iterations-1))
p_value_skewness = 2 * (1 - t.cdf(np.abs(t_statistic_skewness), df=num_iterations-1))
p_value_kurtosis = 2 * (1 - t.cdf(np.abs(t_statistic_kurtosis), df=num_iterations-1))

# Print results
print(f"Mean T-statistic: {t_statistic_mean}, p-value: {p_value_mean}")
print(f"Variance T-statistic: {t_statistic_variance}, p-value: {p_value_variance}")
print(f"Skewness T-statistic: {t_statistic_skewness}, p-value: {p_value_skewness}")
print(f"Kurtosis T-statistic: {t_statistic_kurtosis}, p-value: {p_value_kurtosis}")

# Interpretation
"""
The null hypothesis for the first four moments values:
H0: functions for mean, variance, skewness and kurtosis are unbiased.
"""

if p_value_mean < alpha:
    print("Reject the null hypothesis for mean - function might be biased.")
else:
    print("Fail to reject the null hypothesis for mean - function might be unbiased.")

if p_value_variance < alpha:
    print("Reject the null hypothesis for variance - function might be biased.")
else:
    print("Fail to reject the null hypothesis for variance - function might be unbiased.")

if p_value_skewness < alpha:
    print("Reject the null hypothesis for skewness - function might be biased.")
else:
    print("Fail to reject the null hypothesis for skewness - function might be unbiased.")

if p_value_kurtosis < alpha:
    print("Reject the null hypothesis for kurtosis - function might be biased.")
else:
    print("Fail to reject the null hypothesis for kurtosis - function might be unbiased.")