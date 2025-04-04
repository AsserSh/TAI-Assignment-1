# Load required packages
using Pkg
Pkg.status()
Pkg.add(["DataFrames", "CSV", "GLM", "Statistics", "Plots"])
using DataFrames, CSV, GLM, Statistics, Plots

df = CSV.read("C:/Users/Asser/Downloads/global_food_wastage_dataset.csv", DataFrame)

# Filter only "Fruits & Vegetables" category
df_fruit = filter(row -> row."Food Category" == "Fruits & Vegetables", df)