# Load required packages
using Pkg
Pkg.add(["DataFrames", "CSV", "MLJ", "StatsPlots", "Plots", "Random", "Imbalance", "CategoricalArrays", "DecisionTree", "MLJLinearModels", "MLJFlux"])
using DataFrames, CSV, MLJ, StatsPlots, Plots, Random, Imbalance, CategoricalArrays, DecisionTree,MLJLinearModels, MLJFlux

Pkg.add("MLJDecisionTreeInterface")  # For RandomForestClassifier
Pkg.add("XGBoost")                   # For XGBoostClassifier
Pkg.add("MLJLinearModels")           # For LogisticClassifier
Pkg.add("MLJFlux") 
Pkg.add("MLJXGBoostInterface")

# Load the data
data_path = "C:\\Users\\Asser\\Downloads\\fetal_health.csv"
df = CSV.read(data_path, DataFrame)


# Exploratory Data Analysis (EDA)
    function explore_data(df)
        println("Data dimensions: ", size(df))
        println("\nSample records:")
        println(first(df, 5))
    
        println("\nStatistics summary:")
        println(describe(df))
        
        println("\nMissing values per column:")
        println(combine(df, names(df) .=> (x -> sum(ismissing.(x))) .=> names(df)))
        
        # Distribution of target variable
        target_dist = combine(groupby(df, :fetal_health), nrow => :count)
        println("\nTarget variable distribution:")
        println(target_dist)
        
        # Visualization
        p1 = bar(target_dist.fetal_health, target_dist.count, 
                xlabel="Fetal Health", ylabel="Count", 
                title="Target Distribution", legend=false)
        
        correlation_matrix = cor(Matrix(df[:, Not(:fetal_health)]))
        p2 = heatmap(names(df[:, Not(:fetal_health)]), 
                    names(df[:, Not(:fetal_health)]), 
                    correlation_matrix, 
                    title="Feature Correlation", color=:viridis)
        
        # Access columns with their exact names (with spaces)
        p3 = histogram(df[!,"baseline value"], 
                      xlabel="Baseline Value", ylabel="Count", 
                      title="Baseline Value Distribution", legend=false)
        p4 = histogram(df[!,"accelerations"], 
                      xlabel="Accelerations", ylabel="Count", 
                      title="Accelerations Distribution", legend=false)
       
        plot(p1, p2, p3, p4, layout=(2, 2), size=(1000, 800))
    end

# Data Preprocessing
function prepare_data(df)
    df.fetal_health = categorical(df.fetal_health)
    y = df.fetal_health
    X = select(df, Not(:fetal_health))
    
    normalizer = Standardizer()
    model = machine(normalizer, X)
    fit!(model)
    X = MLJ.transform(model, X)
    
    return X, y
end

X, y = prepare_data(df)

# Handling class imbalance
function balance_classes(X, y)
    class_counts = combine(groupby(DataFrame(y=y), :y), nrow => :count)
    println("\nClass distribution before balancing:")
    println(class_counts)
    
    X_matrix = Float32.(Matrix(X))
    y_vector = Int.(y.refs)
    
    X_resampled, y_resampled = smote(X_matrix, y_vector; k=5, ratios=Dict(1=>1.0, 2=>1.0, 3=>1.0))
    
    X_final = DataFrame(X_resampled, names(X))
    y_final = categorical(y_resampled)
    
    println("\nClass distribution after balancing:")
    println(combine(groupby(DataFrame(y=y_final), :y), nrow => :count))
    
    return X_final, y_final
end

X_balanced, y_balanced = balance_classes(X, y)

# Splitting data
function train_test_split(X, y; test_size=0.2, seed=42)
    Random.seed!(seed)
    n = nrow(X)
    test_indices = randperm(n)[1:round(Int, test_size * n)]
    train_indices = setdiff(1:n, test_indices)
    
    return X[train_indices, :], X[test_indices, :], y[train_indices], y[test_indices]
end

X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced)


# Model evaluation - corrected version
function assess_model(model, X_train, X_test, y_train, y_test)
    m = machine(model, X_train, y_train)
    fit!(m)
    
    # Get both probabilistic predictions and class predictions
    proba_predictions = predict(m, X_test)
    class_predictions = predict_mode(m, X_test)
    
    acc = accuracy(class_predictions, y_test)
    conf_matrix = confusion_matrix(class_predictions, y_test)
    report = classification_report(class_predictions, y_test)
    
    println("\nModel: ", nameof(typeof(model)))
    println("Accuracy: ", round(acc, digits=3))
    println("\nConfusion Matrix:")
    println(conf_matrix)
    println("\nClassification Report:")
    println(report)
    
    heatmap(levels(y_test), levels(y_test), MLJ.confmat(class_predictions, y_test).mat, 
            xlabel="Predicted", ylabel="Actual", title="Confusion Matrix", color=:viridis)
    
    return m, acc, conf_matrix, report
end

# Hyperparameter tuning and model assessment - corrected version
best_model, best_acc = nothing, 0.0
model_results = []

for model in my_models
    model_name = nameof(typeof(model))
    
    if model_name == "RandomForestClassifier"
        tuned = TunedModel(
            model=model,
            tuning=Grid(resolution=3),
            resampling=CV(nfolds=5),
            ranges=[
                range(model, :n_trees, lower=50, upper=150),
                range(model, :max_depth, lower=3, upper=10)
            ]
        )
    elseif model_name == "XGBoostClassifier"
        tuned = TunedModel(
            model=model,
            tuning=Grid(resolution=3),
            resampling=CV(nfolds=5),
            ranges=[
                range(model, :max_depth, lower=3, upper=7),
                range(model, :eta, lower=0.1, upper=0.3)
            ]
        )
    else
        tuned = model
    end
    
    mach = machine(tuned, X_train, y_train)
    fit!(mach)
    class_predictions = predict_mode(mach, X_test)  # Use predict_mode for classifiers
    acc = accuracy(class_predictions, y_test)
    
    push!(model_results, (name=model_name, accuracy=acc, model=mach))
    
    if acc > best_acc
        best_acc, best_model = acc, mach
    end
end


# Load models one by one
RandomForestClassifier = @load RandomForestClassifier pkg=DecisionTree verbosity=1
XGBoostClassifier = @load XGBoostClassifier verbosity=1
LogisticClassifier = @load LogisticClassifier pkg=MLJLinearModels verbosity=1
NeuralNetworkClassifier = @load NeuralNetworkClassifier pkg=MLJFlux verbosity=1

my_models = [
    RandomForestClassifier(),
    XGBoostClassifier(),
    LogisticClassifier(),
    NeuralNetworkClassifier()
]


# Results summary
println("\nModel Comparison:")
for result in model_results
    println("$(result.name): Accuracy = $(round(result.accuracy, digits=3))")
end

println("\nTop Model: $(nameof(typeof(best_model.model))) with accuracy $(round(best_acc, digits=3))")

# Feature importance (if applicable)
if nameof(typeof(best_model.model)) in ["RandomForestClassifier", "XGBoostClassifier"]
    println("\nFeature Importance:")
    
    # For RandomForest
    if nameof(typeof(best_model.model)) == "RandomForestClassifier"
        fi = feature_importances(best_model.model, X_train, y_train)
        fi_df = DataFrame(Feature=names(X_train), Importance=fi.importance)
    # For XGBoost
    else
        importance = xgboost_importance(best_model.model)
        fi_df = DataFrame(Feature=names(X_train), Importance=importance)
    end
    
    sort!(fi_df, :Importance, rev=true)
    println(fi_df)
    
    bar(fi_df.Feature, fi_df.Importance, 
        xlabel="Features", ylabel="Importance", 
        title="Feature Importance", legend=false, 
        xtickfont=font(8, :horizontal), size=(800, 400))
end

# Save best model
model_path = "best_fetal_health_model.jlso"
MLJ.save(model_path, best_model)
println("\nBest model saved at $model_path")

# Predict on sample data
sample = X_test[1:1, :]
predicted = predict_mode(best_model, sample)
actual = y_test[1]
println("\nSample prediction:")
println("Predicted: ", predicted)
println("Actual:    ", actual)
println("Match:     ", predicted[1] == actual)