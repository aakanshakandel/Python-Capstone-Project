# ============================================================
# Step 3: Hypothesis Formation & Testing
#
# Research Question:
# Does international trade relate more strongly to GDP
# or GDP per capita?
#
# Hypothesis:
# Countries with larger economies (higher GDP) tend to
# have higher total trade values.
#
# Tests included:
# 1. Pearson Correlation 
# 2. Linear Regression 
# 3. Comparison between different time periods
#    - 2015-2019
#    - 2020-2024
# ============================================================

import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, when
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression

# ------------------------------------------------
# START SPARK
# ------------------------------------------------
spark = SparkSession.builder.appName("Step3HypothesisTesting").getOrCreate()

# ------------------------------------------------
# PATHS
# ------------------------------------------------
HOME = os.path.expanduser("~")
INPUT_PATH = os.path.join(HOME, "outputs", "final_clean_single")
OUTPUT_DIR = os.path.join(HOME, "outputs", "step3_results")

# ------------------------------------------------
# LOAD CLEAN DATA
# ------------------------------------------------
df = spark.read.csv(INPUT_PATH, header=True, inferSchema=True)

# ------------------------------------------------
# CONVERT FORMATTED COLUMNS BACK TO NUMERIC
# ------------------------------------------------
df = (
    df.withColumn("Year", col("Year").cast("int"))
      .withColumn("gdp_total", col("GDP (USD, billions, 2dp)").cast("double"))
      .withColumn("gdp_pc", col("GDP per capita (USD, 2dp)").cast("double"))
      .withColumn("total_trade", col("Total Trade (USD, billions, 2dp)").cast("double"))
)

# ------------------------------------------------
# HYPOTHESIS
# ------------------------------------------------
print("\n=== HYPOTHESIS ===")
print("Countries with larger economies (higher GDP) tend to have higher total trade values.")
print("This analysis compares GDP vs Trade and GDP per capita vs Trade.")

# ============================================================
# TEST 1: PEARSON CORRELATION (STATISTICAL METHOD)
# ============================================================
corr_gdp_trade = df.stat.corr("gdp_total", "total_trade")
corr_gdppc_trade = df.stat.corr("gdp_pc", "total_trade")

print("\n=== TEST 1: PEARSON CORRELATION ===")
print(f"Correlation between GDP and Total Trade: {corr_gdp_trade:.4f}")
print(f"Correlation between GDP per Capita and Total Trade: {corr_gdppc_trade:.4f}")

if abs(corr_gdp_trade) > abs(corr_gdppc_trade):
    print("Result: GDP has the stronger correlation with Total Trade.")
else:
    print("Result: GDP per capita has the stronger correlation with Total Trade.")

# ============================================================
# TEST 2: LINEAR REGRESSION (MACHINE LEARNING MODEL)
# ============================================================

# -----------------------------
# Model A: GDP -> Trade
# -----------------------------
assembler_gdp = VectorAssembler(inputCols=["gdp_total"], outputCol="features")
df_gdp = assembler_gdp.transform(df).select("features", col("total_trade").alias("label"))

lr_gdp = LinearRegression(featuresCol="features", labelCol="label")
model_gdp = lr_gdp.fit(df_gdp)
summary_gdp = model_gdp.summary

print("\n=== TEST 2A: LINEAR REGRESSION (GDP -> TOTAL TRADE) ===")
print(f"R-squared: {summary_gdp.r2:.4f}")
print(f"Intercept: {model_gdp.intercept:.4f}")
print(f"Coefficient: {model_gdp.coefficients[0]:.4f}")

# -----------------------------
# Model B: GDP per capita -> Trade
# -----------------------------
assembler_gdppc = VectorAssembler(inputCols=["gdp_pc"], outputCol="features")
df_gdppc = assembler_gdppc.transform(df).select("features", col("total_trade").alias("label"))

lr_gdppc = LinearRegression(featuresCol="features", labelCol="label")
model_gdppc = lr_gdppc.fit(df_gdppc)
summary_gdppc = model_gdppc.summary

print("\n=== TEST 2B: LINEAR REGRESSION (GDP PER CAPITA -> TOTAL TRADE) ===")
print(f"R-squared: {summary_gdppc.r2:.4f}")
print(f"Intercept: {model_gdppc.intercept:.4f}")
print(f"Coefficient: {model_gdppc.coefficients[0]:.6f}")

if summary_gdp.r2 > summary_gdppc.r2:
    print("Result: GDP explains more variation in Trade than GDP per capita.")
else:
    print("Result: GDP per capita explains more variation in Trade than GDP.")

# ============================================================
# TEST 3: COMPARISON BETWEEN DIFFERENT TIME PERIODS
# ============================================================

df_period = df.withColumn(
    "period",
    when(col("Year").between(2015, 2019), "2015-2019")
    .otherwise("2020-2024")
)

period_summary = (
    df_period.groupBy("period")
    .agg(
        avg("gdp_total").alias("avg_gdp"),
        avg("gdp_pc").alias("avg_gdp_per_capita"),
        avg("total_trade").alias("avg_total_trade")
    )
    .orderBy("period")
)

print("\n=== TEST 3: TIME PERIOD COMPARISON ===")
period_summary.show(truncate=False)

# ============================================================
# SAVE RESULTS
# ============================================================

# Save correlation results
corr_results = spark.createDataFrame(
    [
        ("GDP vs Total Trade", float(corr_gdp_trade)),
        ("GDP per Capita vs Total Trade", float(corr_gdppc_trade))
    ],
    ["relationship", "pearson_correlation"]
)

corr_results.coalesce(1).write.mode("overwrite").option("header", True) \
    .csv(os.path.join(OUTPUT_DIR, "correlation_results"))

# Save regression results
regression_results = spark.createDataFrame(
    [
        ("GDP -> Total Trade", float(summary_gdp.r2), float(model_gdp.intercept), float(model_gdp.coefficients[0])),
        ("GDP per Capita -> Total Trade", float(summary_gdppc.r2), float(model_gdppc.intercept), float(model_gdppc.coefficients[0]))
    ],
    ["model", "r_squared", "intercept", "coefficient"]
)

regression_results.coalesce(1).write.mode("overwrite").option("header", True) \
    .csv(os.path.join(OUTPUT_DIR, "regression_results"))

# Save time-period comparison
period_summary.coalesce(1).write.mode("overwrite").option("header", True) \
    .csv(os.path.join(OUTPUT_DIR, "time_period_comparison"))

# ============================================================
# FINAL INTERPRETATION
# ============================================================
print("\n=== FINAL INTERPRETATION ===")
if abs(corr_gdp_trade) > abs(corr_gdppc_trade) and summary_gdp.r2 > summary_gdppc.r2:
    print("The hypothesis is supported.")
    print("Trade is more strongly related to GDP than GDP per capita.")
    print("This suggests that overall economic size plays a larger role in trade activity than income per person.")
else:
    print("The results are mixed.")
    print("Review the correlation and regression outputs to determine which variable has the stronger relationship with Trade.")

print(f"\nResults saved in: {OUTPUT_DIR}")
print("\n=== STEP 3 COMPLETED SUCCESSFULLY ===")

spark.stop()

