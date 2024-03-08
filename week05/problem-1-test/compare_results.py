from missing_data import *
from EWMA import *
from nearpsd_higham import *
from chol import *
from simulation import *
from returns_fit_var_es import *

def compare_results(output_file, expected_output_file):
    output_df = pd.read_csv(output_file)
    expected_output_df = pd.read_csv(expected_output_file)

    # Ensure data is numeric
    output_df = output_df.apply(pd.to_numeric, errors='coerce').fillna(0)
    expected_output_df = expected_output_df.apply(pd.to_numeric, errors='coerce').fillna(0)

    # Check if the shapes are identical
    if output_df.shape != expected_output_df.shape:
        raise ValueError(f"Shape mismatch: {output_df.shape} vs {expected_output_df.shape}")

    # Proceed with comparison
    output_df_rounded = np.round(output_df.values, decimals=3)
    expected_output_df_rounded = np.round(expected_output_df.values, decimals=3)
    comparison = np.isclose(output_df_rounded, expected_output_df_rounded, atol=1e-3).all()
    
    return comparison

def compare_results_2(output_file, expected_output_file):
    output_df = pd.read_csv(output_file)
    expected_output_df = pd.read_csv(expected_output_file)

    # Ensure data is numeric
    output_df = output_df.apply(pd.to_numeric, errors='coerce').fillna(0)
    expected_output_df = expected_output_df.apply(pd.to_numeric, errors='coerce').fillna(0)

    # Check if the shapes are identical
    if output_df.shape != expected_output_df.shape:
        raise ValueError(f"Shape mismatch: {output_df.shape} vs {expected_output_df.shape}")

    # Proceed with comparison
    output_df_rounded = np.round(output_df.values, decimals=2)
    expected_output_df_rounded = np.round(expected_output_df.values, decimals=2)
    comparison = np.isclose(output_df_rounded, expected_output_df_rounded, atol=1e-2).all()
    
    return comparison

# test 1
md_handler = MissingData('test1.csv')
cov_matrix_skip = md_handler.SkipMissingRows_cov()
cov_matrix_skip.to_csv('myoutput_1.1.csv', index=False)
corr_matrix_skip = md_handler.SkipMissingRows_corr()
corr_matrix_skip.to_csv('myoutput_1.2.csv', index=False)
cov_pairwise = md_handler.Pairwise_cov()
cov_pairwise.to_csv('myoutput_1.3.csv', index=False)
corr_pairwise = md_handler.Pairwise_cov()
corr_pairwise.to_csv('myoutput_1.4.csv', index=False)

# compare the generated output with the expected output
is_identical_1_1 = compare_results('myoutput_1.1.csv', 'testout_1.1.csv')
print(f"1.1: The results are {'identical' if is_identical_1_1 else 'different'}.")

is_identical_1_2 = compare_results('myoutput_1.2.csv', 'testout_1.2.csv')
print(f"1.2: The results are {'identical' if is_identical_1_2 else 'different'}.")

is_identical_1_3 = compare_results('myoutput_1.3.csv', 'testout_1.3.csv')
print(f"1.3: The results are {'identical' if is_identical_1_3 else 'different'}.")

is_identical_1_4 = compare_results('myoutput_1.4.csv', 'testout_1.4.csv')
print(f"1.4: The results are {'identical' if is_identical_1_3 else 'different'}.")

# test2
ew = EWMAStatistics('test2.csv')
ew_covariance = ew.calculate_ewma_covariance(0.97)
ew_covariance_df = pd.DataFrame(ew_covariance) 
ew_covariance_df.to_csv('myoutput_2.1.csv', index=False)

ew_correlation = ew.calculate_ewma_correlation(0.94)
ew_correlation_df = pd.DataFrame(ew_correlation) 
ew_correlation_df.to_csv('myoutput_2.2.csv', index=False)

covar_with_var_adjustment = ew.calculate_ewma_covariance(0.97)
sd1 = np.sqrt(np.diag(covar_with_var_adjustment))
covar_main = ew.calculate_ewma_covariance(0.94)
sd = 1 / np.sqrt(np.diag(covar_main))
adjusted_covar = np.diag(sd1) @ np.diag(sd) @ covar_main @ np.diag(sd) @ np.diag(sd1)
adjusted_covar_df = pd.DataFrame(adjusted_covar)
adjusted_covar_df.to_csv("notmyoutput_2.3.csv")
df_x = pd.read_csv('notmyoutput_2.3.csv', index_col=0)
df_x.columns = ['x1', 'x2', 'x3', 'x4', 'x5']
df_x = df_x.round(8)
df_x.to_csv('myoutput_2.3.csv', index=False)

is_identical_2_1 = compare_results('myoutput_2.1.csv', 'testout_2.1.csv')
print(f"2.1: The results are {'identical' if is_identical_2_1 else 'different'}.")

is_identical_2_2 = compare_results('myoutput_2.2.csv', 'testout_2.2.csv')
print(f"2.2: The results are {'identical' if is_identical_2_2 else 'different'}.")

is_identical_2_3 = compare_results('myoutput_2.3.csv', 'testout_2.3.csv')
print(f"2.3: The results are {'identical' if is_identical_2_3 else 'different'}.")

# test3
data_array = pd.read_csv('testout_1.3.csv').to_numpy()
near_psd_cov = near_psd_cov(data_array)
near_psd_cov_df = pd.DataFrame(near_psd_cov) 
near_psd_cov_df.to_csv('myoutput_3.1.csv', index=False)

A = pd.read_csv('testout_1.3.csv').values 
higham_cov_matrix = higham_cov(A)
higham_cov_df = pd.DataFrame(higham_cov_matrix) 
higham_cov_df.to_csv('myoutput_3.3.csv', index=False)

data_array_2 = pd.read_csv('testout_1.4.csv').to_numpy()
near_psd_corr = near_psd_corr(data_array_2)
near_psd_corr_df = pd.DataFrame(near_psd_corr) 
near_psd_corr_df.to_csv('myoutput_3.2.csv', index=False)
higham_corr = highams_corr(data_array_2)
higham_corr_df = pd.DataFrame(near_psd_corr) 
higham_corr_df.to_csv('myoutput_3.4.csv', index=False)

is_identical_3_1 = compare_results('myoutput_3.1.csv', 'testout_3.1.csv')
print(f"3.1: The results are {'identical' if is_identical_3_1 else 'different'}.")

is_identical_3_2 = compare_results('myoutput_3.2.csv', 'testout_3.2.csv')
print(f"3.2: The results are {'identical' if is_identical_3_2 else 'different'}.")

is_identical_3_3 = compare_results('myoutput_3.3.csv', 'testout_3.3.csv')
print(f"3.3: The results are {'identical' if is_identical_3_3 else 'different'}.")

is_identical_3_4 = compare_results('myoutput_3.4.csv', 'testout_3.4.csv')
print(f"3.4: The results are {'identical' if is_identical_3_4 else 'different'}.")

# test4
data_array = pd.read_csv('testout_3.1.csv').to_numpy()
chol_matrix = chol_pd(data_array)
chol_df = pd.DataFrame(chol_matrix) 
chol_df.to_csv('myoutput_4.1.csv', index=False)

is_identical_4_1 = compare_results('myoutput_4.1.csv', 'testout_4.1.csv')
print(f"4.1: The results are {'identical' if is_identical_4_1 else 'different'}.")

#test5
# 5.1
df_51 = pd.read_csv('test5_1.csv')
expected_res_51 = pd.read_csv('testout_5.1.csv')
sim_51 = simulate_normal(100000, df_51)
result = np.cov(sim_51)
pd.DataFrame(result).to_csv('myoutput_5.1.csv', index=False) #Convert the numpy array to a pandas DataFrame before saving it to a CSV file
comparison_result = np.isclose(expected_res_51, result, atol=1e-3)
print(f"5.1: The results are {'identical' if np.all(comparison_result) else 'different'}.")

# 5.2
df_52 = pd.read_csv('test5_2.csv').values
expected_res_52 = pd.read_csv('testout_5.2.csv')
sim_52 = simulate_pca(100000, df_52)
result_52 = np.cov(sim_52)
pd.DataFrame(result_52).to_csv('myoutput_5.2.csv', index=False) #Convert the numpy array to a pandas DataFrame before saving it to a CSV file
comparison_result_52 = np.isclose(expected_res_52, result_52, atol=1e-3)
print(f"5.2: The results are {'identical' if np.all(comparison_result_52) else 'different'}.")

# 5.3
df_53 = pd.read_csv('test5_3.csv').values
sim_53 = simulate_normal(100000, df_53, mean=np.zeros(df_53.shape[0]), fix_method='near_psd')
result_53 = np.cov(sim_53)
pd.DataFrame(result_53).to_csv('myoutput_5.3.csv', index=False)
expected_res = pd.read_csv('testout_5.3.csv').values
comparison_result_53_near_psd = np.isclose(expected_res, result_53, atol=1e-3)
print(f"5.3: The results are {'identical' if np.all(comparison_result_53_near_psd) else 'different'}.")

# 5.4 Simulate with higham fix instead of near_psd
sim_54 = simulate_normal_higham(100000, df_53, fix_method='higham')
result_54 = np.cov(sim_54)  # Ensure correct orientation for np.cov
pd.DataFrame(result_54).to_csv('myoutput_5.4.csv', index=False)
expected_res_54 = pd.read_csv('testout_5.4.csv').values
comparison_result_54_higham = np.isclose(expected_res_54, result_54, atol=1e-3)
print(f"5.4: The results are {'identical' if np.all(comparison_result_54_higham) else 'different'}.")

# 5.5 Simulate using PCA with 99% explained variance
sim_55 = simulate_pca(100000, df_52, pctExp=0.99, mean=np.zeros(df_52.shape[0]))
result_55 = np.cov(sim_55)  # Ensure correct orientation for np.cov
pd.DataFrame(result_55).to_csv('myoutput_5.5.csv', index=False)
expected_res_55 = pd.read_csv('testout_5.5.csv').values
comparison_result_55_pca = np.isclose(expected_res_55, result_55, atol=1e-3)
print(f"5.5: The results are {'identical' if np.all(comparison_result_55_pca) else 'different'}.")


# test6
df6_1 = pd.read_csv('test6.csv')
returns_61 = return_calculate(df6_1, "DISCRETE", "Date")
returns_61.to_csv('myoutput_6.1.csv', index=False)
expected_res_61 = pd.read_csv('test6_1.csv')
comparison_result_61 = np.isclose(expected_res_61.iloc[:, 1:].to_numpy(), returns_61.iloc[:, 1:].to_numpy(), atol=1e-3) # to exclude the first column from both DataFrames
print(f"6.1: The results are {'identical' if np.all(comparison_result_61) else 'different'}.")

df6_2 = pd.read_csv('test6.csv')
returns_62 = return_calculate(df6_2, "LOG", "Date")
returns_62.to_csv('myoutput_6.2.csv', index=False)
expected_res_62 = pd.read_csv('test6_2.csv')
comparison_result_62 = np.isclose(expected_res_62.iloc[:, 1:].to_numpy(), returns_62.iloc[:, 1:].to_numpy(), atol=1e-3) # to exclude the first column from both DataFrames
print(f"6.2: The results are {'identical' if np.all(2) else 'different'}.")

# test7
# 7.1
df7_1 = pd.read_csv('test7_1.csv')

mu, std = fit_normal_distribution(df7_1)
result_71 = pd.DataFrame({
    'mu': [mu],
    'std': [std],
})
result_71.to_csv('myoutput_7.1.csv', index=False)
is_identical_7_1 = compare_results('myoutput_7.1.csv', 'testout7_1.csv')
print(f"7.1: The results are {'identical' if is_identical_7_1 else 'different'}.")

# 7.2
df7_2 = pd.read_csv('test7_2.csv')
df, loc, scale = fit_t_distribution(df7_2)
result_72 = pd.DataFrame({
    'mu': [loc],
    'std': [scale],
    'nu': [df]
})
result_72.to_csv('myoutput_7.2.csv', index=False)
is_identical_7_2 = compare_results('myoutput_7.2.csv', 'testout7_2.csv')
print(f"7.2: The results are {'identical' if is_identical_7_2 else 'different'}.") # bbb = aaa.iloc[0]  提取第一行 数据只有一行

# 7.3
df7_3 = pd.read_csv('test7_3.csv')
# X_71 = df7_3[['x1', 'x2', 'x3']]
# y_71 = df7_3['y']
loc, scale, df, beta0, beta1, beta2, beta3 = t_regression(df7_3)
result_73 = pd.DataFrame({
    'mu': [loc],
    'std': [scale],
    'nu': [df],
    'Alpha': [beta0],
    'B1': [beta1],
    'B2': [beta2],
    'B3': [beta3]
})

result_73.to_csv('myoutput_7.3.csv', index=False)
is_identical_7_3 = compare_results('myoutput_7.3.csv', 'testout7_3.csv')
print(f"7.3: The results are {'identical' if is_identical_7_3 else 'different'}.")

# 8
# 8.1
df8_1 = pd.read_csv('test7_1.csv')
var_norm = calculate_normal_VaR(df8_1, 0.95)
# 计算VaR与均值的差距
mean_of_data = df8_1.mean() 
var_diff_from_mean = var_norm + mean_of_data
result_df = pd.DataFrame({
    'VaR Absolute': [var_norm],
    'VaR Diff from Mean': [var_diff_from_mean.item()]
})
result_df.to_csv('myoutput_8.1.csv', index=False)
is_identical_8_1 = compare_results('myoutput_8.1.csv', 'testout8_1.csv')
print(f"8.1: The results are {'identical' if is_identical_8_1 else 'different'}.")

# 8.2
df8_2 = pd.read_csv('test7_2.csv')
var_norm_82 = calculate_t_VaR(df8_2)
mean_of_data_82 = df8_2.mean() 
var_diff_from_mean_82 = var_norm_82 + mean_of_data_82
result_df_82 = pd.DataFrame({
    'VaR Absolute': [var_norm_82],
    'VaR Diff from Mean': [var_diff_from_mean_82.item()]
})
result_df_82.to_csv('myoutput_8.2.csv', index=False)
is_identical_8_2 = compare_results('myoutput_8.2.csv', 'testout8_2.csv')
print(f"8.2: The results are {'identical' if is_identical_8_2 else 'different'}.")

# 8.3
var_norm_83 = calculate_simulation_VaR(df8_2['x1'])
mean_of_data_83 = df8_2.mean() 
var_diff_from_mean_83 = var_norm_83 + mean_of_data_83
result_df_83 = pd.DataFrame({
    'VaR Absolute': [var_norm_83],
    'VaR Diff from Mean': [var_diff_from_mean_83.item()]
})
result_df_83.to_csv('myoutput_8.3.csv', index=False)
is_identical_8_3 = compare_results_2('myoutput_8.3.csv', 'testout8_3.csv')
print(f"8.3: The results are {'identical' if is_identical_8_3 else 'different'}.")

# 8.4 ?
es_norm_84 = calculate_normal_ES(df8_1)
mean_of_data_84 = df8_1.mean() 
var_diff_from_mean_84 = es_norm_84 + mean_of_data_84
result_df_84 = pd.DataFrame({
    'ES Absolute': [es_norm_84],
    'ES Diff from Mean': [var_diff_from_mean_84.item()]
})
result_df_84.to_csv('myoutput_8.4.csv', index=False)
is_identical_8_4 = compare_results_2('myoutput_8.4.csv', 'testout8_4.csv')
print(f"8.4: The results are {'identical' if is_identical_8_4 else 'different'}.")

# 8.5
es_norm_85 = calculate_t_ES(df8_2)
mean_of_data_85 = df8_2.mean() 
var_diff_from_mean_85 = es_norm_85 + mean_of_data_85
result_df_85 = pd.DataFrame({
    'ES Absolute': [es_norm_85],
    'ES Diff from Mean': [var_diff_from_mean_85.item()]
})
result_df_85.to_csv('myoutput_8.5.csv', index=False)
is_identical_8_5 = compare_results_2('myoutput_8.5.csv', 'testout8_5.csv')
print(f"8.5: The results are {'identical' if is_identical_8_5 else 'different'}.")

# 8.6
es_norm_86 = calculate_simulation_ES(df8_2)
mean_of_data_86 = df8_2.mean() 
var_diff_from_mean_86 = es_norm_86 + mean_of_data_86
result_df_86 = pd.DataFrame({
    'ES Absolute': [es_norm_86],
    'ES Diff from Mean': [var_diff_from_mean_86.item()]
})
result_df_86.to_csv('myoutput_8.6.csv', index=False)
is_identical_8_6 = compare_results_2('myoutput_8.6.csv', 'testout8_6.csv')
print(f"8.6: The results are {'identical' if is_identical_8_6 else 'different'}.")

# 9.1
df1_91 = pd.read_csv('test9_1_portfolio.csv')
df2_91 = pd.read_csv('test9_1_returns.csv')
# each_stock_results, total_portfolio_value = calculate_each_stock(df1_91, df2_91)
results_df_91 = simulateCopula(df1_91, df2_91)
results_df_91.to_csv('myoutput_9.1.csv', index=False)

is_identical_9_1 = compare_results_2('myoutput_9.1.csv', 'testout9_1.csv')
print(f"9.1: The results are {'identical' if is_identical_9_1 else 'different'}.")



