"""
# TfIdf\\_Transformer

TfIdf\\_Transformer.fit(X; smooth=true, norm=:l2, use_idf=true)

Parameters:
  - X::Matrix{Float64} - is a terms count matrix (2D), where X(i,j) >= 0  
  - norm::Symbol - could be either :l1 or :l2(default)  
  - use_idf::Bool - do not use inverse document frequency, default: true 
  - smooth::Bool - adds 1 to tf(t) and df(d) befor calculation, default: true
**returns** named tuple as a model   

TfIdf_Transformer.transform(m, X)

Parameters:
  - X::Matrix{Float64} - is a terms counts matrix, X(i,j) must be >= 0,
  - m::NamedTuple - fitted transformer, returned by TfIdf_Transformer.fit
**returns** transformed data

Example:
````julia
X = [0 1 1 1 0 0 1 0 1
     0 2 0 1 0 1 1 0 1
     1 0 0 1 1 0 1 1 1
     0 1 1 1 0 0 1 0 1]

tfidf = TfIdf_Transformer.fit(X)
X_transformed = TfIdf_Transformer.transform(tfidf, X)
````
"""
module TfIdf_Transformer

    function fit(X::Matrix{Int64}; smooth::Bool=true, norm=:l2, use_idf::Bool=true)
        if use_idf
            if smooth
                idf = log.((size(X, 1) +1 ) ./ (sum(X .> 0, dims=1) .+1)) .+ 1
            else
                idf = log.( size(X, 1) ./sum(X .> 0, dims=1)) .+ 1
            end
        else
            idf = nothing
        end

        model =  (
            smooth=smooth,
            norm=norm,
            use_idf=use_idf,
            idf=idf
        )
        return model
    end

    function transform(model, X::Matrix{Int64})
        if model.use_idf
            freq_mx = X ./ sum(X, dims=2)
            tfidf = freq_mx .* model.idf
            if model.norm == :l2
                return tfidf ./ sqrt.(sum(tfidf .^2, dims=2))
            else
                return tfidf ./ sum(abs.(tfidf), dims=2)
            end
        else
            freq_mx = X ./ sum(X, dims=2)
            if model.norm == :l2
                return freq_mx ./ sqrt.(sum(freq_mx .^2, dims=2))
            else
                return freq_mx ./ sum(abs.(freq_mx), dims=2)
            end
        end
    end

end
