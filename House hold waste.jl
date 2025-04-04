using Pkg
Pkg.status()
Pkg.add(["DataFrames", "CSV", "GLM", "Statistics", "Plots"])
using DataFrames, CSV, GLM, Statistics, Plots


# Load the dataset
file_path = "C:\\Users\\Asser\\Downloads\\global_food_wastage_dataset.csv"
df = CSV.read(file_path, DataFrame)


# Filter for Dairy Products only
dairy_df = filter(row -> row."Food Category" == "Dairy Products", df)

# Convert Year to a continuous variable (though it's categorical, we'll treat it as continuous for simplicity)
dairy_df.Year = parse.(Float64, string.(dairy_df.Year))

rename!(dairy_df, Dict(
    "Economic Loss (Million \$)" => :Economic_Loss,
    "Total Waste (Tons)" => :Total_Waste,
    "Avg Waste per Capita (Kg)" => :Avg_Waste_per_Capita,
    "Population (Million)" => :Population,
    "Household Waste (%)" => :Household_Waste
))

#Building a linear regressiion
model = lm(@formula(Household_Waste ~ Total_Waste + Economic_Loss + Avg_Waste_per_Capita + Population + Year), dairy_df)

# Display model results
println("\nRegression Results:")
println(model)