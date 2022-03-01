using MLUtils

function r2(; y_true::Vector{Float64}, y_pred::Vector{Float64})
    y_bar = sum(y_true)/length(y_true)
    return 1.0 - sum((y_true .- y_pred) .^2) / sum((y_true .- y_bar).^2)
end

r_squared = r2
