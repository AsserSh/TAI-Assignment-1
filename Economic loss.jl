# Load required packages
using Pkg
Pkg.status()
Pkg.add(["DataFrames", "CSV", "GLM", "Statistics", "Plots"])
using DataFrames, CSV, GLM, Statistics, Plots

df = CSV.read("C:/Users/Asser/Downloads/global_food_wastage_dataset.csv", DataFrame)

# Filter only "Fruits & Vegetables" category
df_fruit = filter(row -> row."Food Category" == "Fruits & Vegetables", df)


# Select relevant columns
rename!(df_fruit, Dict(
    "Economic Loss (Million \$)" => :Economic_Loss,
    "Total Waste (Tons)" => :Total_Waste,
    "Avg Waste per Capita (Kg)" => :Avg_Waste_per_Capita,
    "Population (Million)" => :Population,
    "Household Waste (%)" => :Household_Waste
))

#Building a linear regressiion
model = lm(@formula(Economic_Loss ~ Total_Waste + Avg_Waste_per_Capita + Population + Household_Waste), df_fruit)

println(model)