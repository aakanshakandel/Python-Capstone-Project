# ============================================================
# Step 2: Exploratory Data Analysis & Visualisation
#
# Graphs produced:
# 1. Average GDP per Capita by Country (2015-2024)
# 2. Average GDP by Country (2015-2024)
# 3. Total Trade vs GDP per Capita (2015-2024, all years)
# 4. Total Trade vs GDP (2015-2024, all years)
# 5. Average Total Trade vs Average GDP per Capita
# 6. Average Total Trade vs Average GDP
#
# ============================================================

import matplotlib
matplotlib.use("Agg")   
# Uses a non-interactive backend because the Linux VM
# does not have a graphical display.

import matplotlib.pyplot as plt
# matplotlib is used to create all visualisations in this script:
# bar charts and scatter plots, and to save them as images.

import pandas as pd
# pandas is used to convert PySpark DataFrames to Pandas DataFrames.
# matplotlib cannot plot Spark DataFrames directly.

import numpy as np
# numpy is used to calculate the regression line (line of best fit)
# using numpy.polyfit() and numpy.poly1d().

import os 
# Used to manage file paths and create the folder where graphs are saved

from pyspark.sql import SparkSession
# SparkSession starts the PySpark environment
# so we can load and analyse the dataset.

from pyspark.sql.functions import col, avg
# col() is used to reference and convert Spark DataFrame columns
# avg() is used to calculate averages across years.


# ------------------------------------------------
# START SPARK
# ------------------------------------------------
spark = SparkSession.builder.appName("Step2EDA").getOrCreate()

# ------------------------------------------------
# PATHS
# ------------------------------------------------
HOME = os.path.expanduser("~")
INPUT_PATH = os.path.join(HOME, "outputs", "final_clean_single")
GRAPH_DIR = os.path.join(HOME, "outputs", "graphs")

os.makedirs(GRAPH_DIR, exist_ok=True)

# ------------------------------------------------
# LOAD CLEAN DATA
# ------------------------------------------------
df = spark.read.csv(INPUT_PATH, header=True, inferSchema=True)

# ------------------------------------------------
# CONVERT FORMATTED COLUMNS BACK TO NUMERIC
# ------------------------------------------------
# In Step 1, some values were saved in formatted presentation form.
# This step converts them back to numeric columns so they can be used
# for calculations, scatter plots, and regression lines.
df = (
    df.withColumn("Year", col("Year").cast("int"))
      .withColumn("gdp_pc", col("GDP per capita (USD, 2dp)").cast("double"))
      .withColumn("gdp_total", col("GDP (USD, billions, 2dp)").cast("double"))
      .withColumn("total_trade", col("Total Trade (USD, billions, 2dp)").cast("double"))
)

# ------------------------------------------------
# COUNTRY ORDER
# ------------------------------------------------
# Define a fixed country order so that countries appear consistently
country_order = ["United States", "Australia", "China", "Brazil", "India", "Vietnam"]

# ------------------------------------------------
# COUNTRY AVERAGES
# ------------------------------------------------
# Calculate the average GDP per capita, average GDP, and average total trade
# for each country across the full 2015-2024 period.
# Used for bar charts
avg_df = df.groupBy("Country or Area").agg(
    avg("gdp_pc").alias("avg_gdp_pc"),
    avg("gdp_total").alias("avg_gdp"),
    avg("total_trade").alias("avg_trade")
)

# Convert the aggregated Spark DataFrame to pandas
# matplotlib cannot plot Spark DataFrames directly

pdf_avg = avg_df.toPandas()

# Add a custom sort order column so the countries follow the chosen sequence.
pdf_avg["sort_order"] = pdf_avg["Country or Area"].apply(
    lambda x: country_order.index(x) if x in country_order else 999
)

pdf_avg = pdf_avg.sort_values("sort_order").reset_index(drop=True)

# ------------------------------------------------
# FULL DATASET FOR YEAR-LEVEL SCATTER PLOTS
# ------------------------------------------------
# Keep the year-level observations for each country.
# This dataset is used in Graph 3 and Graph 4 so that every year
# from 2015 to 2024 is included as a separate point.
pdf = df.select(
    "Country or Area",
    "Year",
    "gdp_pc",
    "gdp_total",
    "total_trade"
).toPandas()

# ------------------------------------------------
# GRAPH 1
# Average GDP per Capita by Country
# ------------------------------------------------
plt.figure(figsize=(10, 6))
plt.bar(pdf_avg["Country or Area"], pdf_avg["avg_gdp_pc"])
plt.title("Average GDP per Capita (USD) by Country (2015-2024)")
plt.xlabel("Country")
plt.ylabel("Average GDP per Capita (USD)")
plt.xticks(rotation=45)
plt.tight_layout()

graph1_path = os.path.join(GRAPH_DIR, "graph1_avg_gdp_per_capita.png")
plt.savefig(graph1_path, dpi=300, bbox_inches="tight")
plt.close()

# ------------------------------------------------
# GRAPH 2
# Average GDP by Country
# ------------------------------------------------
plt.figure(figsize=(10, 6))
plt.bar(pdf_avg["Country or Area"], pdf_avg["avg_gdp"])
plt.title("Average GDP (USD) by Country (2015-2024)")
plt.xlabel("Country")
plt.ylabel("Average GDP (Billions USD)")
plt.xticks(rotation=45)
plt.tight_layout()

# Define output path and save the graph.
graph2_path = os.path.join(GRAPH_DIR, "graph2_avg_gdp.png")
plt.savefig(graph2_path, dpi=300, bbox_inches="tight")
plt.close()

# ------------------------------------------------
# GRAPH 3
# Total Trade vs GDP per Capita
# ------------------------------------------------
plt.figure(figsize=(10, 6))

countries = pdf["Country or Area"].unique()

# Plot each country's yearly observations separately.
for country in countries:
    subset = pdf[pdf["Country or Area"] == country]
    plt.scatter(
        subset["gdp_pc"],
        subset["total_trade"],
        label=country,
        alpha=0.7
    )

# Calculate a linear regression line (best-fit line)
x1 = pdf["gdp_pc"]
y1 = pdf["total_trade"]
z1 = np.polyfit(x1, y1, 1)
p1 = np.poly1d(z1)

x1_sorted = np.sort(x1)

# Plot the best-fit line on top of the scatter plot.
plt.plot(
    x1_sorted,
    p1(x1_sorted),
    color="black",
    linewidth=2,
    label="Best Fit Line"
)

plt.title("Total Trade (USD) vs GDP per Capita (USD) (2015-2024)")
plt.xlabel("GDP per Capita (USD)")
plt.ylabel("Total Trade (Billions USD)")
plt.legend(title="Country")
plt.tight_layout()

# Define output path and save the graph.
graph3_path = os.path.join(GRAPH_DIR, "graph3_gdp_per_capita_vs_total_trade_all_years.png")
plt.savefig(graph3_path, dpi=300, bbox_inches="tight")
plt.close()

# ------------------------------------------------
# GRAPH 4
# Total Trade vs GDP
# ------------------------------------------------
plt.figure(figsize=(10, 6))

for country in countries:
    subset = pdf[pdf["Country or Area"] == country]
    plt.scatter(
        subset["gdp_total"],
        subset["total_trade"],
        label=country,
        alpha=0.7
    )

x2 = pdf["gdp_total"]
y2 = pdf["total_trade"]
z2 = np.polyfit(x2, y2, 1)
p2 = np.poly1d(z2)

x2_sorted = np.sort(x2)

plt.plot(
    x2_sorted,
    p2(x2_sorted),
    color="black",
    linewidth=2,
    label="Best Fit Line"
)

plt.title("Total Trade (USD) vs GDP (USD) (2015-2024)")
plt.xlabel("GDP (Billions USD)")
plt.ylabel("Total Trade (Billions USD)")
plt.legend(title="Country")
plt.tight_layout()

graph4_path = os.path.join(GRAPH_DIR, "graph4_gdp_vs_total_trade_all_years.png")
plt.savefig(graph4_path, dpi=300, bbox_inches="tight")
plt.close()

# ------------------------------------------------
# GRAPH 5
# Average Total Trade vs Average GDP per Capita
# ------------------------------------------------
plt.figure(figsize=(10, 6))

for i, row in pdf_avg.iterrows():
    plt.scatter(
        row["avg_gdp_pc"],
        row["avg_trade"],
        label=row["Country or Area"],
        s=80,
        alpha=0.8
    )
    plt.annotate(
        row["Country or Area"],
        (row["avg_gdp_pc"], row["avg_trade"]),
        xytext=(5, 5),
        textcoords="offset points"
    )

x3 = pdf_avg["avg_gdp_pc"]
y3 = pdf_avg["avg_trade"]
z3 = np.polyfit(x3, y3, 1)
p3 = np.poly1d(z3)

x3_sorted = np.sort(x3)

plt.plot(
    x3_sorted,
    p3(x3_sorted),
    color="black",
    linewidth=2,
    label="Best Fit Line"
)

plt.title("Average Total Trade (USD) vs Average GDP per Capita (USD) (2015-2024)")
plt.xlabel("Average GDP per Capita (USD)")
plt.ylabel("Average Total Trade (Billions USD)")
plt.legend(title="Country")
plt.tight_layout()

graph5_path = os.path.join(GRAPH_DIR, "graph5_avg_gdp_per_capita_vs_avg_total_trade.png")
plt.savefig(graph5_path, dpi=300, bbox_inches="tight")
plt.close()

# ------------------------------------------------
# GRAPH 6
# Average Total Trade vs Average GDP
# ------------------------------------------------
plt.figure(figsize=(10, 6))

for i, row in pdf_avg.iterrows():
    plt.scatter(
        row["avg_gdp"],
        row["avg_trade"],
        label=row["Country or Area"],
        s=80,
        alpha=0.8
    )
    plt.annotate(
        row["Country or Area"],
        (row["avg_gdp"], row["avg_trade"]),
        xytext=(5, 5),
        textcoords="offset points"
    )

x4 = pdf_avg["avg_gdp"]
y4 = pdf_avg["avg_trade"]
z4 = np.polyfit(x4, y4, 1)
p4 = np.poly1d(z4)

x4_sorted = np.sort(x4)

plt.plot(
    x4_sorted,
    p4(x4_sorted),
    color="black",
    linewidth=2,
    label="Best Fit Line"
)

plt.title("Average Total Trade (USD) vs Average GDP (USD) (2015-2024)")
plt.xlabel("Average GDP (Billions USD)")
plt.ylabel("Average Total Trade (Billions USD)")
plt.legend(title="Country")
plt.tight_layout()

graph6_path = os.path.join(GRAPH_DIR, "graph6_avg_gdp_vs_avg_total_trade.png")
plt.savefig(graph6_path, dpi=300, bbox_inches="tight")
plt.close()

# ------------------------------------------------
# PRINT OUTPUT LOCATIONS
# ------------------------------------------------
# Print confirmation so the user knows the script completed successfully
# and where each graph was saved.
print("\n=== STEP 2 COMPLETED SUCCESSFULLY ===")
print(f"Graph 1 saved to: {graph1_path}")
print(f"Graph 2 saved to: {graph2_path}")
print(f"Graph 3 saved to: {graph3_path}")
print(f"Graph 4 saved to: {graph4_path}")
print(f"Graph 5 saved to: {graph5_path}")
print(f"Graph 6 saved to: {graph6_path}")

spark.stop()

