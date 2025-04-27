import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.stats.api as sms
import statsmodels.api as sm

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")

df = pd.read_csv('ab_data.csv')

print("\nDataset Preview:")
print(df.head())

print("\nBasic Dataset Information:")
print(df.info())
print("\nData Summary:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())

df['timestamp'] = pd.to_datetime(df['timestamp'])

print("\nA/B Test Analysis:")
print(f"Total samples: {len(df)}")
print(f"Control group (A): {sum(df['group'] == 'control')} samples")
print(f"Treatment group (B): {sum(df['group'] == 'treatment')} samples")

conversion_rate_A = df[df['group'] == 'control']['converted'].mean()
conversion_rate_B = df[df['group'] == 'treatment']['converted'].mean()

print(f"\nConversion rate for Control (A): {conversion_rate_A:.4f} ({conversion_rate_A*100:.2f}%)")
print(f"Conversion rate for Treatment (B): {conversion_rate_B:.4f} ({conversion_rate_B*100:.2f}%)")
print(f"Absolute difference: {abs(conversion_rate_B - conversion_rate_A):.4f} ({abs(conversion_rate_B - conversion_rate_A)*100:.2f}%)")
print(f"Relative difference: {(conversion_rate_B - conversion_rate_A) / conversion_rate_A * 100:.2f}%")

plt.figure(figsize=(10, 6))
sns.barplot(x='group', y='converted', data=df, ci=95)
plt.title('Conversion Rates by Group with 95% Confidence Intervals', fontsize=15)
plt.xlabel('Group', fontsize=12)
plt.ylabel('Conversion Rate', fontsize=12)
plt.savefig('conversion_rates.png', dpi=300, bbox_inches='tight')
plt.show()

successes_A = df[df['group'] == 'control']['converted'].sum()
trials_A = sum(df['group'] == 'control')
successes_B = df[df['group'] == 'treatment']['converted'].sum()
trials_B = sum(df['group'] == 'treatment')

chi2, p_chi2, _, _ = stats.chi2_contingency([
    [successes_A, trials_A - successes_A],
    [successes_B, trials_B - successes_B]
])

count = np.array([successes_A, successes_B])
nobs = np.array([trials_A, trials_B])
z_stat, p_z = sm.stats.proportions_ztest(count, nobs)

ci_A = sms.proportion_confint(successes_A, trials_A)
ci_B = sms.proportion_confint(successes_B, trials_B)

print("\nStatistical Tests Results:")
print(f"Chi-squared test: χ² = {chi2:.4f}, p-value = {p_chi2:.4f}")
print(f"Z-test for proportions: z = {z_stat:.4f}, p-value = {p_z:.4f}")

alpha = 0.05
if p_z < alpha:
    print(f"The difference is statistically significant at {alpha} significance level.")
else:
    print(f"The difference is not statistically significant at {alpha} significance level.")

print("\nConfidence Intervals (95%):")
print(f"Control: ({ci_A[0]:.4f}, {ci_A[1]:.4f})")
print(f"Treatment: ({ci_B[0]:.4f}, {ci_B[1]:.4f})")

df['date'] = df['timestamp'].dt.date
daily_conversion = df.groupby(['date', 'group'])['converted'].mean().reset_index()

plt.figure(figsize=(12, 6))
sns.lineplot(x='date', y='converted', hue='group', data=daily_conversion, marker='o')
plt.title('Daily Conversion Rates by Group', fontsize=15)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Conversion Rate', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('daily_conversion.png', dpi=300, bbox_inches='tight')
plt.show()

plt.figure(figsize=(10, 6))
conversion_counts = df.groupby('group')['converted'].value_counts(normalize=True).unstack().reset_index()
conversion_counts = pd.melt(conversion_counts, id_vars=['group'], value_vars=[0, 1], var_name='converted', value_name='percentage')

sns.barplot(x='group', y='percentage', hue='converted', data=conversion_counts)
plt.title('Distribution of Conversions by Group', fontsize=15)
plt.xlabel('Group', fontsize=12)
plt.ylabel('Percentage', fontsize=12)
plt.legend(title='Converted', labels=['No', 'Yes'])
plt.savefig('conversion_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

user_visits = df.groupby(['date', 'group'])['user_id'].count().reset_index()
user_visits.columns = ['date', 'group', 'visits']

plt.figure(figsize=(12, 6))
sns.lineplot(x='date', y='visits', hue='group', data=user_visits)
plt.title('Daily User Visits by Group', fontsize=15)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Number of Visits', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('daily_visits.png', dpi=300, bbox_inches='tight')
plt.show()

effect_size = 0.01
power = 0.9
baseline_conversion = conversion_rate_A

required_sample = sms.proportion_effectsize(baseline_conversion, baseline_conversion + effect_size)
sample_size = sms.NormalIndPower().solve_power(effect_size=required_sample, power=power, alpha=0.05, ratio=1)

print("\nSample Size Calculation for Future Tests:")
print(f"To detect a {effect_size*100}% difference with {power*100}% power and 5% significance level:")
print(f"Required sample size: {int(np.ceil(sample_size))} users per group ({int(np.ceil(sample_size*2))} total)")

df['day_of_week'] = df['timestamp'].dt.day_name()
day_conversion = df.groupby(['day_of_week', 'group'])['converted'].mean().reset_index()

plt.figure(figsize=(14, 7))
sns.barplot(x='day_of_week', y='converted', hue='group', data=day_conversion)
plt.title('Conversion Rates by Day of Week', fontsize=15)
plt.xlabel('Day of Week', fontsize=12)
plt.ylabel('Conversion Rate', fontsize=12)
plt.savefig('day_of_week_conversion.png', dpi=300, bbox_inches='tight')
plt.show()

df.to_csv('processed_ab_data.csv', index=False)