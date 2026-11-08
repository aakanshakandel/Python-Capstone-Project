# ============================================================
# Step 1: Data Preprocessing and Cleaning
#
# Key processing steps:
# - Standardise country names
# - Remove null and invalid values
# - Filter selected countries and years
# - Aggregate exports, imports, total trade, and trade balance
# - Merge trade data with GDP and GDP per capita
# - Save clean outputs for later analysis
# ============================================================

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, upper, when, sum as Fsum, lit, format_string,
    trim, count, isnan
)

spark = SparkSession.builder.appName("Step1Simple").getOrCreate()

# -----------------------------
# CONFIG
# -----------------------------
# COUNTRIES: countries included in the analysis
# BILLION: scaling factor used to convert large USD values into billions
COUNTRIES = ["United States", "Australia", "China", "Brazil", "India", "Vietnam"]
START, END = 2015, 2024
BILLION = 1_000_000_000.0

# -----------------------------
# 1. LOAD FILES
# -----------------------------
# Load the three CSV files into Spark DataFrames
trades = spark.read.csv("data/trades.csv", header=True, inferSchema=True)
gdp_total = spark.read.csv("data/GDP,PPP.csv", header=True, inferSchema=True)
gdp_pc = spark.read.csv("data/GDP,PPP,percapita.csv", header=True, inferSchema=True)

# -----------------------------
# 2. FIX COLUMN NAMES 
# -----------------------------
# Some datasets may use "Country" while others use "Country or Area".
# This step ensures a consistent column name so later joins work correctly.
if "Country" in trades.columns and "Country or Area" not in trades.columns:
    trades = trades.withColumnRenamed("Country", "Country or Area")

if "Country" in gdp_total.columns and "Country or Area" not in gdp_total.columns:
    gdp_total = gdp_total.withColumnRenamed("Country", "Country or Area")

if "Country" in gdp_pc.columns and "Country or Area" not in gdp_pc.columns:
    gdp_pc = gdp_pc.withColumnRenamed("Country", "Country or Area")

# -----------------------------
# 3. BASIC CLEANING
# -----------------------------
# Trim country names in case there are spaces
trades = trades.withColumn("Country or Area", trim(col("Country or Area")))
gdp_total = gdp_total.withColumn("Country or Area", trim(col("Country or Area")))
gdp_pc = gdp_pc.withColumn("Country or Area", trim(col("Country or Area")))

# Standardise country names
trades = trades.withColumn(
    "Country or Area",
    when(col("Country or Area") == "USA", "United States")
    .when(col("Country or Area") == "Viet Nam", "Vietnam")
    .otherwise(col("Country or Area"))
)

gdp_total = gdp_total.withColumn(
    "Country or Area",
    when(col("Country or Area") == "USA", "United States")
    .when(col("Country or Area") == "Viet Nam", "Vietnam")
    .otherwise(col("Country or Area"))
)

gdp_pc = gdp_pc.withColumn(
    "Country or Area",
    when(col("Country or Area") == "USA", "United States")
    .when(col("Country or Area") == "Viet Nam", "Vietnam")
    .otherwise(col("Country or Area"))
)


# -----------------------------------
# 4. KEEP ONLY REQUIRED TRADE COLUMNS
# -----------------------------------
# Select only the columns needed for the analysis.
# - cast Year to integer
# - standardise Flow values to uppercase
# - cast Trade (USD) to numeric
# - keep only selected countries and years
# - keep only EXPORT and IMPORT records
# - remove null and invalid negative trade values

tr = (
    trades.select("Country or Area", "Year", "Flow", "Trade (USD)")
    .withColumn("Year", col("Year").cast("int"))
    .withColumn("Flow", upper(trim(col("Flow"))))
    .withColumn("Trade (USD)", col("Trade (USD)").cast("double"))
    .filter(col("Country or Area").isin(COUNTRIES))
    .filter(col("Year").between(START, END))
    .filter(col("Flow").isin("EXPORT", "IMPORT"))
    .filter(col("Trade (USD)").isNotNull())
    .filter(col("Trade (USD)") >= 0)
)

# -----------------------------
# 5. COUNTRY ORDER
# -----------------------------
# Create a custom ordering expression so countries always appear
# in the same sequence in outputs and previews.

order_expr = (
    when(col("Country or Area") == COUNTRIES[0], 1)
    .when(col("Country or Area") == COUNTRIES[1], 2)
    .when(col("Country or Area") == COUNTRIES[2], 3)
    .when(col("Country or Area") == COUNTRIES[3], 4)
    .when(col("Country or Area") == COUNTRIES[4], 5)
    .when(col("Country or Area") == COUNTRIES[5], 6)
    .otherwise(99)
)

# -----------------------------
# 6. ORDER ROW-LEVEL TRADES
# -----------------------------
# Sort row-level trade data by:
# 1. chosen country order
# 2. year in descending order
# 3. flow type
tr_ordered = (
    tr.withColumn("country_order", order_expr)
      .orderBy(col("country_order").asc(), col("Year").desc(), col("Flow").asc())
      .drop("country_order")
)

# Convert row-level trade to BILLIONS of USD(2dp)
tr_ordered = tr_ordered.withColumn(
    "Trade (USD, billions, 2dp)",
    format_string("%.2f", col("Trade (USD)") / lit(BILLION))
).drop("Trade (USD)")

# Save ordered row-level trades
tr_ordered.coalesce(1).write.mode("overwrite").option("header", True) \
    .csv("outputs/trades_ordered_clean")

print("\n=== ROW-LEVEL TRADES PREVIEW (first 10 rows) ===")
tr_ordered.show(10, truncate=False)

# -----------------------------
# 7. AGGREGATE TRADE VALUES
# -----------------------------
# Group by country and year, then separately sum exports and imports.
# This allows calculation of:
# - total trade = exports + imports
# - trade balance = exports - imports

tr_agg = (
    tr.groupBy("Country or Area", "Year")
      .agg(
          Fsum(when(col("Flow") == "EXPORT", col("Trade (USD)")).otherwise(0)).alias("exports"),
          Fsum(when(col("Flow") == "IMPORT", col("Trade (USD)")).otherwise(0)).alias("imports")
      )
      .withColumn("total_trade", col("exports") + col("imports"))
      .withColumn("trade_balance", col("exports") - col("imports"))
)

# -----------------------------
# 8. CLEAN GDP TOTAL
# -----------------------------
# Keep only required columns, cast data types, filter selected countries/years,
# and remove null or invalid negative values.

gdp_total_clean = (
    gdp_total.select("Country or Area", "Year", "Value")
    .withColumn("Year", col("Year").cast("int"))
    .withColumn("gdp_total", col("Value").cast("double"))
    .drop("Value")
    .filter(col("Country or Area").isin(COUNTRIES))
    .filter(col("Year").between(START, END))
    .filter(col("gdp_total").isNotNull())
    .filter(col("gdp_total") >= 0)
)

# -----------------------------
# 9. CLEAN GDP PER CAPITA
# -----------------------------
# Keep only required columns, cast data types, filter selected countries/years,
# and remove null or invalid negative values.
gdp_pc_clean = (
    gdp_pc.select("Country or Area", "Year", "Value")
    .withColumn("Year", col("Year").cast("int"))
    .withColumn("gdp_pc", col("Value").cast("double"))
    .drop("Value")
    .filter(col("Country or Area").isin(COUNTRIES))
    .filter(col("Year").between(START, END))
    .filter(col("gdp_pc").isNotNull())
    .filter(col("gdp_pc") >= 0)
)

# -----------------------------
# 10. OPTIONAL DATA QUALITY CHECKS
# -----------------------------
# These checks count null values in each cleaned dataset.
# For numeric columns, isnan() is also checked where relevant.

print("\n=== NULL CHECKS ===")
tr.select([
    count(when(col(c).isNull() | isnan(c), c)).alias(c) if c == "Trade (USD)"
    else count(when(col(c).isNull(), c)).alias(c)
    for c in tr.columns
]).show()

gdp_total_clean.select([
    count(when(col(c).isNull() | isnan(c), c)).alias(c) if c == "gdp_total"
    else count(when(col(c).isNull(), c)).alias(c)
    for c in gdp_total_clean.columns
]).show()

gdp_pc_clean.select([
    count(when(col(c).isNull() | isnan(c), c)).alias(c) if c == "gdp_pc"
    else count(when(col(c).isNull(), c)).alias(c)
    for c in gdp_pc_clean.columns
]).show()

# -----------------------------
# 11. BUILD FINAL MERGED DATASET
# -----------------------------
# Merge aggregated trade data with GDP total and GDP per capita
# using country and year as the common join keys.

final_raw = (
    tr_agg.join(gdp_total_clean, on=["Country or Area", "Year"], how="left")
          .join(gdp_pc_clean, on=["Country or Area", "Year"], how="left")
)

# Order final dataset by country then descending year
final_raw = (
    final_raw.withColumn("country_order", order_expr)
             .orderBy(col("country_order").asc(), col("Year").desc())
             .drop("country_order")
)

# Format numeric metrics for presentation-ready output.
# GDP and trade values are converted to billions of USD.
# GDP per capita is kept in USD
final = (
    final_raw
    .withColumn("GDP (USD, billions, 2dp)",
                format_string("%.2f", col("gdp_total") / lit(BILLION)))
    .withColumn("GDP per capita (USD, 2dp)",
                format_string("%.2f", col("gdp_pc")))
    .withColumn("Exports (USD, billions, 2dp)",
                format_string("%.2f", col("exports") / lit(BILLION)))
    .withColumn("Imports (USD, billions, 2dp)",
                format_string("%.2f", col("imports") / lit(BILLION)))
    .withColumn("Total Trade (USD, billions, 2dp)",
                format_string("%.2f", col("total_trade") / lit(BILLION)))
    .withColumn("Trade Balance (USD, billions, 2dp)",
                format_string("%.2f", col("trade_balance") / lit(BILLION)))
    .select(
        "Country or Area", "Year",
        "GDP (USD, billions, 2dp)",
        "GDP per capita (USD, 2dp)",
        "Exports (USD, billions, 2dp)",
        "Imports (USD, billions, 2dp)",
        "Total Trade (USD, billions, 2dp)",
        "Trade Balance (USD, billions, 2dp)"
    )
)

# Save final dataset
final.coalesce(1).write.mode("overwrite").option("header", True) \
    .csv("outputs/final_clean_single")

print("\n=== FINAL MERGED PREVIEW (first 20 rows) ===")
final.show(20, truncate=False)

print("\n=== STEP 1 COMPLETED SUCCESSFULLY ===")

# Stop Spark session after processing is complete
spark.stop()

