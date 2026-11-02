# Impact of Temperature and Rainfall on Crop Yield
#
# Topic: Climate change impact on agriculture
#
# Problem statement: Analyse how average temperature and total precipitation affect crop yield (MT per hectare) across 10 countries and 10 crop types from 1990 to 2024, and identify the temperature range beyond which yield declines.
#
# The analysis is divided into three parts:
#   Part 1: Data Cleaning & Manipulation         (NumPy / Pandas)
#   Part 2: Data Visualization                   (Matplotlib / Seaborn)
#   Part 3: Statistical Analysis & Forecasting   (SciPy / Statsmodels)
#
# Install the libraries once, in the VS Code terminal, before running:
#   python -m pip install numpy pandas matplotlib seaborn scipy statsmodels
#
# Keep the CSV file in the same folder as this file before running it.
# Each chart opens in its own window. Close the window to move to the next step.


# =====================================================================
# PART 1: DATA CLEANING & MANIPULATION (NumPy / Pandas)
# =====================================================================

# Step 1: Import the libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.formula.api as smf

# Step 2: Load the data
# Functions used: read_csv(), head()

df = pd.read_csv('revised climate_change_impact_on_agriculture_2024.csv')
print(df.head())

# Step 3: Understand the data
# Functions used: shape, columns, info()

print('Rows and columns:', df.shape)
print(df.columns)
df.info()

# Step 4:To check is there a null(empty or undefined data)
#Function used: df.isnull which uses boolean to check

print('Missing values in each column:')
print(df.isnull().sum())

duplicate_rows = len(df) - len(df.drop_duplicates())
print('Duplicate rows:', duplicate_rows)

# Step 5: Clean the data
# Remove duplicate rows, and fill any missing number with the average of its column.
# This file has no duplicates and no missing values, so nothing changes here,
# but the step makes sure the analysis still works if the data is updated later.
# Functions used: drop_duplicates(), fillna(), mean()

df = df.drop_duplicates()

number_columns = ['Average_Temperature_C', 'Total_Precipitation_mm', 'Crop_Yield_MT_per_HA']
for col in number_columns:
    df[col] = df[col].fillna(df[col].mean())

print('Rows after cleaning:', len(df))

# Step 6:The columns listed below are not necessary as per the problem statement
#Functions used: df.drop to drop the listed columns

data = df.drop(columns=['Region', 'CO2_Emissions_MT', 'Extreme_Weather_Events',
                        'Irrigation_Access_%', 'Pesticide_Use_KG_per_HA',
                        'Fertilizer_Use_KG_per_HA', 'Soil_Health_Index',
                        'Adaptation_Strategies', 'Economic_Impact_Million_USD'])
print(data.head())

# Step 7: Check for outliers with NumPy (IQR method)
# A value is an outlier if it lies more than 1.5 x IQR below Q1 or above Q3.
# Functions used: to_numpy(), np.percentile(), np.sum()

for col in number_columns:
    values = data[col].to_numpy()
    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)
    iqr = q3 - q1
    lower_limit = q1 - 1.5 * iqr
    upper_limit = q3 + 1.5 * iqr
    outliers = np.sum((values < lower_limit) | (values > upper_limit))
    print(col, '-> outliers:', outliers)

# Step 8: Summary statistics
# Functions used: np.mean(), np.median(), np.std(), np.min(), np.max(), describe()

yield_values = data['Crop_Yield_MT_per_HA'].to_numpy()
print('Mean yield   :', np.round(np.mean(yield_values), 2))
print('Median yield :', np.round(np.median(yield_values), 2))
print('Std deviation:', np.round(np.std(yield_values), 2))
print('Lowest yield :', np.min(yield_values))
print('Highest yield:', np.max(yield_values))

print(data[number_columns].describe().round(2))

# Step 9: Create new columns with NumPy
# np.select() checks the conditions from top to bottom and gives each row
# the label of the first condition that is True.
# Functions used: np.select(), np.square()

temp = data['Average_Temperature_C']
rain = data['Total_Precipitation_mm']

# 5-degree temperature bands
band_order = ['Below 0', '0-5', '5-10', '10-15', '15-20', '20-25', '25-30', '30-35']
temp_conditions = [temp < 0, temp < 5, temp < 10, temp < 15, temp < 20, temp < 25, temp < 30]
temp_labels = ['Below 0', '0-5', '5-10', '10-15', '15-20', '20-25', '25-30']
data['Temp_Band'] = np.select(temp_conditions, temp_labels, default='30-35')

# Three wider temperature zones: Cold (below 10 C), Optimal (10 to 20 C), Hot (20 C and above)
zone_order = ['Cold', 'Optimal', 'Hot']
data['Temp_Zone'] = np.select([temp < 10, temp < 20], ['Cold', 'Optimal'], default='Hot')

# Rainfall bands
rain_order = ['200-900', '900-1600', '1600-2300', '2300-3000']
rain_conditions = [rain < 900, rain < 1600, rain < 2300]
data['Rain_Band'] = np.select(rain_conditions, ['200-900', '900-1600', '1600-2300'], default='2300-3000')

# Temperature squared, used for the curved model in Part 3
data['Temp_Squared'] = np.square(temp)

print(data.head())

# Step 10: Summary tables
# Functions used: groupby(), mean(), agg(), sort_values(), pivot_table(), reset_index()

temp_yield = data.groupby('Temp_Band')['Crop_Yield_MT_per_HA'].mean().round(2)
temp_yield = temp_yield.loc[band_order]
print(temp_yield)

zone_yield = data.groupby('Temp_Zone')['Crop_Yield_MT_per_HA'].agg(['mean', 'count']).round(2)
zone_yield = zone_yield.loc[zone_order]
print(zone_yield)

rain_yield = data.groupby('Rain_Band')['Crop_Yield_MT_per_HA'].mean().round(2)
rain_yield = rain_yield.loc[rain_order]
print(rain_yield)

country_yield = data.groupby('Country')['Crop_Yield_MT_per_HA'].mean().round(2)
country_yield = country_yield.sort_values(ascending=False)
print(country_yield)

# Rows are crops, columns are temperature bands, values are the average yield
crop_table = data.pivot_table(index='Crop_Type', columns='Temp_Band',
                              values='Crop_Yield_MT_per_HA', aggfunc='mean').round(2)
crop_table = crop_table[band_order]
print(crop_table)

# One row per year, used for the trend chart and the forecast
yearly = data.groupby('Year')[['Average_Temperature_C', 'Crop_Yield_MT_per_HA']].mean().reset_index()
print(yearly.tail())


# =====================================================================
# PART 2: DATA VISUALIZATION (Matplotlib / Seaborn)
# =====================================================================

sns.set_theme(style='whitegrid')

# Colours for the three temperature zones, used in every chart that shows them
zone_colors = {'Cold': '#2a78d6', 'Optimal': '#1baf7a', 'Hot': '#eb6834'}

# Step 11: Yearly trend of temperature and yield (Matplotlib)
# Functions used: plt.subplots(), plot(), tight_layout()

fig, axes = plt.subplots(2, 1)

axes[0].plot(yearly['Year'], yearly['Average_Temperature_C'], color='red', marker='o')
axes[0].set_title('Average Temperature by Year')
axes[0].set_ylabel('Temperature (C)')

axes[1].plot(yearly['Year'], yearly['Crop_Yield_MT_per_HA'], color='green', marker='o')
axes[1].set_title('Average Crop Yield by Year')
axes[1].set_xlabel('Year')
axes[1].set_ylabel('Yield (MT per HA)')

plt.tight_layout()
plt.show()

# Step 12: Distribution of crop yield (Seaborn)
# Shows which yield values are common and which are rare.
# Functions used: sns.histplot()

fig, ax = plt.subplots()
sns.histplot(data=data, x='Crop_Yield_MT_per_HA', bins=30, kde=True, color='#2a78d6', ax=ax)
ax.set_title('Distribution of Crop Yield')
ax.set_xlabel('Crop Yield (MT per HA)')
ax.set_ylabel('Number of Records')
plt.show()

# Step 13: Temperature vs crop yield (Seaborn)
# Each dot is one row of the data, coloured by its temperature zone.
# Functions used: sns.scatterplot()

fig, ax = plt.subplots()
sns.scatterplot(data=data, x='Average_Temperature_C', y='Crop_Yield_MT_per_HA',
                hue='Temp_Zone', hue_order=zone_order, palette=zone_colors,
                s=8, alpha=0.4, ax=ax)
ax.set_title('Temperature vs Crop Yield')
ax.set_xlabel('Average Temperature (C)')
ax.set_ylabel('Crop Yield (MT per HA)')
ax.legend(title='Temperature Zone')
plt.show()

# Step 14: Average yield in each temperature band (Matplotlib)
# This shows where yield is highest and where it falls.
# Functions used: bar(), bar_label()

fig, ax = plt.subplots()
bars = ax.bar(temp_yield.index, temp_yield.values, color='orange')
ax.bar_label(bars, padding=3)
ax.set_ylim(0, 3.5)
ax.set_title('Average Crop Yield by Temperature Band')
ax.set_xlabel('Temperature Band (C)')
ax.set_ylabel('Average Yield (MT per HA)')
plt.show()

# Step 15: Spread of yield in each temperature band (Seaborn)
# A wider part of a violin means more records have that yield. The small box inside shows the median.
# Functions used: sns.violinplot()

fig, ax = plt.subplots(figsize=(9, 5))
sns.violinplot(data=data, x='Temp_Band', y='Crop_Yield_MT_per_HA', order=band_order,
               color='#86b6ef', ax=ax)
ax.set_title('Spread of Crop Yield by Temperature Band')
ax.set_xlabel('Temperature Band (C)')
ax.set_ylabel('Crop Yield (MT per HA)')
plt.show()

# Step 16: Average yield in each rainfall band (Seaborn)
# The bar is the average yield. The small black line on top is the 95% confidence interval.
# Functions used: sns.barplot()

fig, ax = plt.subplots()
sns.barplot(data=data, x='Rain_Band', y='Crop_Yield_MT_per_HA', order=rain_order,
            color='skyblue', ax=ax)
ax.set_title('Average Crop Yield by Rainfall Band')
ax.set_xlabel('Total Precipitation (mm)')
ax.set_ylabel('Average Yield (MT per HA)')
plt.show()

# Step 17: Compare crop types (Seaborn)
# Each cell is the average yield of one crop in one temperature band. Darker means higher yield.
# Functions used: sns.heatmap()

fig, ax = plt.subplots(figsize=(9, 5))
sns.heatmap(crop_table, annot=True, fmt='.2f', cmap='Greens', ax=ax)
ax.set_title('Average Yield of Each Crop by Temperature Band')
ax.set_xlabel('Temperature Band (C)')
ax.set_ylabel('Crop Type')
plt.tight_layout()
plt.show()

# Step 18: Compare countries (Seaborn)
# Functions used: sns.barplot()

fig, ax = plt.subplots()
sns.barplot(data=data, x='Crop_Yield_MT_per_HA', y='Country', order=country_yield.index,
            color='seagreen', ax=ax)
ax.set_title('Average Crop Yield by Country')
ax.set_xlabel('Average Yield (MT per HA)')
ax.set_ylabel('Country')
plt.tight_layout()
plt.show()

# Step 19: Correlation heatmap (Seaborn)
# A value close to 1 or -1 means a strong relationship. A value close to 0 means a weak one.
# Functions used: corr(), sns.heatmap()

corr = data[['Year', 'Average_Temperature_C', 'Total_Precipitation_mm',
             'Crop_Yield_MT_per_HA']].corr().round(2)
print(corr['Crop_Yield_MT_per_HA'])

fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1, ax=ax)
ax.set_title('Correlation between Year, Temperature, Rainfall and Yield')
plt.tight_layout()
plt.show()


# =====================================================================
# PART 3: STATISTICAL ANALYSIS & FORECASTING (SciPy / Statsmodels)
# =====================================================================
# How to read a p-value: if it is below 0.05, the result is statistically
# significant, meaning it is unlikely to be caused by chance alone.
# A p-value printed as 0.0 means it is smaller than 0.0001.

# Step 20: Pearson correlation test (SciPy)
# Tests whether temperature and rainfall have a straight-line relationship with yield.
# Functions used: stats.pearsonr()

r_temp, p_temp = stats.pearsonr(data['Average_Temperature_C'], data['Crop_Yield_MT_per_HA'])
r_rain, p_rain = stats.pearsonr(data['Total_Precipitation_mm'], data['Crop_Yield_MT_per_HA'])

print('Temperature and yield: r =', round(r_temp, 3), ', p-value =', round(p_temp, 4))
print('Rainfall and yield   : r =', round(r_rain, 3), ', p-value =', round(p_rain, 4))

# Step 21: One-way ANOVA across the three temperature zones (SciPy)
# Tests whether the average yield of the Cold, Optimal and Hot zones is really different.
# Functions used: stats.f_oneway()

cold_yield = data[data['Temp_Zone'] == 'Cold']['Crop_Yield_MT_per_HA']
optimal_yield = data[data['Temp_Zone'] == 'Optimal']['Crop_Yield_MT_per_HA']
hot_yield = data[data['Temp_Zone'] == 'Hot']['Crop_Yield_MT_per_HA']

f_value, p_value = stats.f_oneway(cold_yield, optimal_yield, hot_yield)
print('ANOVA for temperature zones: F =', round(f_value, 1), ', p-value =', round(p_value, 4))

# Step 22: t-test between the Optimal and Hot zones (SciPy)
# This directly tests the problem statement: does yield decline beyond 20 C?
# Functions used: stats.ttest_ind()

t_value, p_value = stats.ttest_ind(optimal_yield, hot_yield, equal_var=False)
print('Average yield in Optimal zone:', round(optimal_yield.mean(), 2))
print('Average yield in Hot zone    :', round(hot_yield.mean(), 2))
print('t-test: t =', round(t_value, 1), ', p-value =', round(p_value, 4))

if p_value < 0.05:
    print('Result: the fall in yield beyond 20 C is statistically significant.')
else:
    print('Result: the fall in yield beyond 20 C is not statistically significant.')

# Step 23: One-way ANOVA across the rainfall bands (SciPy)
# Functions used: stats.f_oneway()

rain_groups = []
for band in rain_order:
    rain_groups.append(data[data['Rain_Band'] == band]['Crop_Yield_MT_per_HA'])

f_value, p_value = stats.f_oneway(rain_groups[0], rain_groups[1], rain_groups[2], rain_groups[3])
print('ANOVA for rainfall bands: F =', round(f_value, 1), ', p-value =', round(p_value, 4))

# Step 24: Linear regression (Statsmodels)
# Fits a straight-line model: yield = a + b x temperature + c x rainfall.
# R-squared shows how much of the change in yield the model explains (0 = nothing, 1 = everything).
# Functions used: smf.ols(), fit(), summary()

linear_model = smf.ols('Crop_Yield_MT_per_HA ~ Average_Temperature_C + Total_Precipitation_mm',
                       data=data).fit()
print(linear_model.summary())
print('R-squared of straight-line model:', round(linear_model.rsquared, 3))

# Step 25: Curved regression (Statsmodels)
# Yield rises and then falls with temperature, so a curve fits better than a straight line.
# Adding Temp_Squared lets the model bend. The top of the curve is the best temperature.
# Functions used: smf.ols(), fit(), params

curved_model = smf.ols('Crop_Yield_MT_per_HA ~ Average_Temperature_C + Temp_Squared + Total_Precipitation_mm',
                       data=data).fit()
print(curved_model.summary())
print('R-squared of curved model:', round(curved_model.rsquared, 3))

b_temp = curved_model.params['Average_Temperature_C']
b_squared = curved_model.params['Temp_Squared']
best_temperature = -b_temp / (2 * b_squared)
print('Temperature at which the model predicts the highest yield:', round(best_temperature, 1), 'C')

# Step 26: Forecast of average yield for 2025 to 2029 (Statsmodels)
# Fits a trend line through the 35 yearly averages and extends it five years ahead.
# Functions used: smf.ols(), fit(), get_prediction(), summary_frame(), fill_between()

trend_model = smf.ols('Crop_Yield_MT_per_HA ~ Year', data=yearly).fit()
print('Change in yield per year:', round(trend_model.params['Year'], 4))
print('p-value of the trend    :', round(trend_model.pvalues['Year'], 4))

future = pd.DataFrame({'Year': np.arange(2025, 2030)})
prediction = trend_model.get_prediction(future).summary_frame(alpha=0.05)

future['Forecast'] = prediction['mean'].to_numpy()
future['Lower'] = prediction['obs_ci_lower'].to_numpy()
future['Upper'] = prediction['obs_ci_upper'].to_numpy()
print(future.round(2))

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(yearly['Year'], yearly['Crop_Yield_MT_per_HA'], color='green', marker='o', label='Actual')
ax.plot(yearly['Year'], trend_model.fittedvalues, color='gray', linestyle='--', label='Trend line')
ax.plot(future['Year'], future['Forecast'], color='#eb6834', marker='o', label='Forecast')
ax.fill_between(future['Year'], future['Lower'], future['Upper'], color='#eb6834', alpha=0.2,
                label='95% prediction range')
ax.set_title('Average Crop Yield: Actual and Forecast (2025-2029)')
ax.set_xlabel('Year')
ax.set_ylabel('Yield (MT per HA)')
ax.legend()
plt.show()

# Step 27: Conclusion
#
# Part 1 - Data cleaning and manipulation
# - The data has 10,000 rows with no missing values, no duplicate rows and no outliers (IQR method), so no rows were removed.
# - New columns were created with NumPy: temperature bands, temperature zones (Cold, Optimal, Hot), rainfall bands and temperature squared.
#
# Part 2 - Data visualization
# - Yearly trend: the yearly average temperature moves between about 14.1 and 16.4 C and the yearly average yield between 2.13 and 2.39 MT per HA, with no clear upward or downward trend over 1990-2024.
# - Temperature is the main factor: yield is highest, about 3.0 MT per HA, when the average temperature is between 10 and 20 C.
# - Yield declines beyond 20 C: above 20 C the average yield falls to about 2.3 MT per HA, roughly a quarter lower than the best range.
# - Cold conditions are worst: below 10 C the average yield is only about 1.6 MT per HA, almost half of the best range.
# - Rainfall has very little effect: average yield is 2.18 MT per HA in the driest band and about 2.26 in the others.
# - Crops and countries: the heatmap shows all 10 crops follow the same temperature pattern, and country averages are nearly equal (2.20 to 2.28 MT per HA).
#
# Part 3 - Statistical analysis and forecasting
# - Pearson correlation: temperature and yield r = 0.264 (significant). Rainfall and yield r = 0.03, which is significant only because the dataset is large; the relationship is very weak.
# - ANOVA: the average yield of the Cold, Optimal and Hot zones is significantly different (F = 2100.3, p-value below 0.0001).
# - t-test: yield in the Optimal zone (3.03) is significantly higher than in the Hot zone (2.29), t = 31.0, p-value below 0.0001. This confirms that yield declines beyond 20 C.
# - Rainfall ANOVA: F = 4.4, p-value = 0.004. The difference is statistically significant but small (about 0.08 MT per HA), compared with 0.74 MT per HA between the Optimal and Hot zones.
# - Regression: the straight-line model explains only 7% of the change in yield (R-squared = 0.07). The curved model explains 15% (R-squared = 0.154) and puts the best temperature at 19.9 C, which agrees with the charts.
# - Forecast: the yearly trend is +0.001 MT per HA per year and is not significant (p-value = 0.39). The forecast for 2025-2029 is therefore almost flat at about 2.26 MT per HA, with a 95% range of about 2.11 to 2.41.
#
# Note: this dataset appears to be synthetic (computer generated), so the results show the pattern in this data and should not be quoted as real-world evidence.
