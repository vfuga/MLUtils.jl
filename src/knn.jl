
module KNN

    using Lazy

    _metrics = Dict{Symbol, Any}()

    function knn(X::Matrix{Float64}; max_k::Int64=5, metric=nothing, max_distance=nothing)
        # get default metric

        # @show ((metric == :cityblock) | (metric == :manhattan)) # & (max_distance === nothing)

        neighbors = Vector{Any}(undef, size(X, 1))

        if ((metric === nothing) | (metric == :euclidean)) & (max_distance === nothing)
            Threads.@threads for i in 1:size(X, 1)
                r = sqrt.(sum((X .- X[i, :]') .^2; dims=2))[:, 1]
                ix = sortperm(r)[1:max_k+1]
                ix = [j for j in ix if j != i]
                neighbors[i] = (index=i, dist=r[ix], neighbors=ix)
            end
        elseif ((metric == :cityblock) | (metric == :manhattan)) & (max_distance === nothing)
            Threads.@threads for i in 1:size(X, 1)
                r = sum(abs.(X .- X[i, :]'); dims=2)[:, 1]
                ix = sortperm(r)[1:max_k+1]
                ix = [j for j in ix if (j != i)]
                neighbors[i] = (index=i, dist=r[ix], neighbors=ix)
            end
        elseif (metric == :cos) & (max_distance === nothing)
            denominators = sum(X .* X; dims=2)[:, 1]
            for i in 1:size(X, 1)
                r = 1.0 .- sum(X .* X[i, :]'; dims=2)[:, 1] ./ sqrt.(denominators .*  (X[i, :]' * X[i, :]))
                ix = sortperm(r)[1:max_k+1]
                ix = [j for j in ix if (j != i)]
                neighbors[i] = (index=i, dist=r[ix], neighbors=ix)
            end
        elseif (metric == :cos) & (max_distance > 0.0)
            denominators = sum(X .* X; dims=2)[:, 1]
            for i in 1:size(X, 1)
                r = 1.0 .- sum(X .* X[i, :]'; dims=2)[:, 1] ./ sqrt.(denominators .*  (X[i, :]' * X[i, :]))
                ix = sortperm(r)[1:max_k+1]
                ix = [j for j in ix if (j != i) & (r[j] < max_distance)]
                neighbors[i] = (index=i, dist=r[ix], neighbors=ix)
            end
        elseif ((metric == :cityblock) | (metric == :manhattan)) & (max_distance >= 0.0)
            Threads.@threads for i in 1:size(X, 1)
                r = sum(abs.(X .- X[i, :]'); dims=2)[:, 1]
                ix = sortperm(r)[1:max_k+1]
                ix = [j for j in ix if (j != i) & (r[j] < max_distance)]
                neighbors[i] = (index=i, dist=r[ix], neighbors=ix)
            end
        elseif ((metric === nothing) | (metric == :euclidean)) & (max_distance >= 0.0)
            Threads.@threads for i in 1:size(X, 1)
                r = sqrt.(sum((X .- X[i, :]') .^2; dims=2))[:, 1]
                ix = sortperm(r)[1:max_k+1]
                ix = [j for j in ix if (j != i) & (r[j] < max_distance)]
                neighbors[i] = (index=i, dist=r[ix], neighbors=ix)
            end
        end

        return neighbors
    end

end

using .KNN

X = rand(10000, 20)
@time neighbors = Main.KNN.knn(X; max_k = 5)
@time neighbors = Main.KNN.knn(X; max_k = 5, max_distance=5.0, metric=:euclidean)
@time neighbors = Main.KNN.knn(X; max_k = 5, metric= :manhattan)
@time neighbors = Main.KNN.knn(X; max_k = 5, metric= :manhattan, max_distance=.4)
@time neighbors = Main.KNN.knn(X; max_k = 5, metric= :cos)
@time neighbors = Main.KNN.knn(X; max_k = 5, metric= :cos, max_distance=.2)
