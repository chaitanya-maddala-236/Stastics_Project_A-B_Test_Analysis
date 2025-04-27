# A/B Testing Analysis Project

## Overview
This project analyzes a Kaggle A/B testing dataset to determine if a new webpage design (treatment) performs better than the original design (control) in terms of user conversion rates. The analysis includes data visualization, statistical significance testing, and additional segmentation analysis.

## Dataset
The project uses the ["AB Test Data" dataset from Kaggle](https://www.kaggle.com/datasets/zhangluyuan/ab-testing), which contains:
- **user_id**: Unique identifier for each user
- **timestamp**: When the user visited the page
- **group**: Whether the user was in the "control" (original) or "treatment" (new) group
- **converted**: Binary indicator (0 or 1) showing whether the user converted

"Conversion" in this context refers to whether users completed a target action, such as making a purchase, signing up for a service, or clicking a specific button.

## Features

### Data Processing
- Loading and exploration of the A/B test dataset
- Handling datetime conversion for time-based analysis
- Missing value detection

### Statistical Analysis
- Calculation of conversion rates for control and treatment groups
- Chi-squared test for independence
- Z-test for proportions
- Confidence interval calculation
- Statistical significance evaluation at 95% confidence level
- Sample size calculation for future experiments

### Visualizations
1. **Conversion Rates Comparison**: Bar chart showing conversion rates with confidence intervals
2. **Daily Conversion Rates**: Line chart tracking conversion rates over time
3. **Conversion Distribution**: Visualization of the proportion of conversions vs. non-conversions
4. **User Visits Analysis**: Tracking of daily visitor count by group
5. **Day of Week Analysis**: Examining conversion rates by day of the week

## Requirements
- Python 3.6+
- pandas
- numpy
- matplotlib
- seaborn
- scipy
- statsmodels

## Installation
```bash
pip install pandas numpy matplotlib seaborn scipy statsmodels
```

## Usage
1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/zhangluyuan/ab-testing)
2. Place the `ab_data.csv` file in your working directory
3. Run the script:
```bash
python ab_testing_analysis.py
```

## Output
The script generates:
- Conversion rate statistics for both groups
- Results of statistical significance tests
- Five visualizations saved as PNG files:
  - `conversion_rates.png`
  - `daily_conversion.png`
  - `conversion_distribution.png`
  - `daily_visits.png`
  - `day_of_week_conversion.png`
- A processed data file (`processed_ab_data.csv`)
- Sample size recommendations for future experiments

## Business Application
This type of analysis helps businesses make data-driven decisions by:
- Determining if design changes positively impact user behavior
- Quantifying the improvement in conversion rates
- Identifying potential patterns in user behavior over time
- Planning properly powered future experiments

## License
This project is provided for educational purposes. The dataset is sourced from Kaggle and is subject to their terms of use.