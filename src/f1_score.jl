@doc """
f1\\_score(; y\\_proba::Vector{Float64}, y\\_true::Vector{Int64})

May be used to look for an optimal threshols to maximize F1-score

**returns** named tuple

````
return (
    f1::Vector{Float64}, 
    threshold::Vector{Float64}
)
````
"""
function f1_score(;y_proba::Vector{Float64}, y_true::Vector{Int64})

    zip(y_proba[:, 2], y_true) |> collect |> sort |> x -> begin

        f1 = Any[]
        len = length(y_true)
        
        cnt_p = (y_true |> frequencies)[1]
        cnt_n = len - cnt_p

        tp = cnt_p
        fp = cnt_n

        fn = 0
        tn = 0
        t = 0.0

        for (i, r) in enumerate(x)
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1_ = 2.0 / (1.0 / precision + 1.0 / recall)

            push!(f1, (f1_, t))
            
            t, y = r
            tp -= y
            fn += y
            tn += (1 - y)
            fp -= (1 - y)
        end

        return (
            f1 = [v[1] for v in f1], 
            threshold = [v[2] for v in f1]
        )
    end
end
