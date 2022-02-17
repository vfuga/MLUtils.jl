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
