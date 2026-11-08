# Python-Capstone-Project
# Python Capstone Project – GDP, GDP Per Capita and International Trade Analysis

## Project Overview

This capstone project investigated whether a country's total GDP or GDP per capita has a stronger relationship with its international trade value.

Using economic data from the World Bank and UN Data sources, I analysed six countries (Australia, United States, China, India, Brazil and Vietnam) over a 10-year period from 2015 to 2024, producing a dataset of 60 country-year observations.

The main objective was to determine which economic indicator was more strongly associated with total international trade, measured as the combined value of exports and imports.

## 1. Data Extraction and Preparation

I collected economic datasets containing GDP, GDP per capita, exports and imports.

Using Python and PySpark, I:

- Loaded multiple CSV datasets into PySpark DataFrames.
- Examined the datasets for missing values, duplicates and inconsistent data types.
- Standardised country names to ensure consistency across datasets.
- Filtered the data to the six selected countries and the 2015–2024 period.
- Removed invalid records and converted numerical columns into appropriate data types.

## 2. Data Transformation and Integration

After cleaning the datasets, I used PySpark to transform and combine the economic indicators.

- Aggregated export and import values by country and year.
- Calculated **Total Trade = Exports + Imports**.
- Calculated trade balance as an additional derived metric, although it was not used in the final correlation analysis.
- Joined the GDP, GDP per capita and trade datasets using country and year.
- Created a consolidated dataset for statistical analysis and visualisation.

## 3. Exploratory Data Analysis and Visualisation

I used PySpark, Pandas and Matplotlib to investigate economic trends and relationships.

The visualisations included:

- GDP and GDP per capita comparisons across countries.
- Total trade trends between 2015 and 2024.
- Country-level averages of economic indicators.
- Scatter plots examining GDP against total trade.
- Scatter plots examining GDP per capita against total trade.
- Regression lines to illustrate the relationships between variables.

## 4. Statistical Analysis – Correlation and Linear Regression

To answer the research question, I performed Pearson correlation analysis and simple linear regression using PySpark MLlib.

I developed two separate regression models:

- **Model 1:** GDP as the independent variable and total trade as the dependent variable.
- **Model 2:** GDP per capita as the independent variable and total trade as the dependent variable.

### Key Findings

| Metric | GDP vs Total Trade | GDP Per Capita vs Total Trade |
|---|---|---|
| Pearson correlation (r) | 0.96 | 0.36 |
| R² | 0.92 | 0.13 |
| Relationship | Strong positive | Weak-to-moderate positive |

The results showed that total GDP had a substantially stronger relationship with international trade than GDP per capita.

GDP explained approximately 92% of the variation in total trade within the fitted simple regression model, compared with approximately 13% for GDP per capita.

## 5. Conclusion

The analysis demonstrated that the overall size of a country's economy was more strongly associated with its international trade value than its economic output per person.

Although GDP per capita provides valuable information about a country's economic development, GDP was a stronger predictor of total trade in this dataset.

The findings represent statistical associations rather than causal relationships and are limited to the six countries and 10-year period analysed.

## Skills Demonstrated

- Data extraction, cleaning and transformation using PySpark.
- Data integration using DataFrame operations and joins.
- Exploratory data analysis using Python, Pandas and Matplotlib.
- Statistical analysis using Pearson correlation.
- Predictive modelling using PySpark MLlib Linear Regression.
- Economic data interpretation and presentation of analytical findings.

**Technologies Used:** Python, PySpark, Spark SQL, Pandas, NumPy, Matplotlib, PySpark MLlib.
