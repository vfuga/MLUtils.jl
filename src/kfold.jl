using Random

@doc """
StratifiedKfold(y::Vector{Int64}, k::Int64) for *binary classification*

**Parameters:**
- y::Vector{Int}, yᵢ ∈ {0, 1}
- k::Int, number of folds

"""
function StratifiedKfold(y::Vector{Int64}, k::Int64)
    all_ix = collect(1:length(y))

    positives_ix = Random.shuffle([i for (i, v) in enumerate(y) if v == 1])
    negatives_ix = Random.shuffle([i for (i, v) in enumerate(y) if v == 0])
    
    p_batch_sz = Int64(round(length(positives_ix)/k))
    n_batch_sz = Int64(round(length(negatives_ix)/k))

    folds = []

    for i in 1:k
        start = 1 + (i-1)*p_batch_sz
        stop = i == k ? length(positives_ix) : i * p_batch_sz
        append!(folds, [positives_ix[start: stop]])
    end

    for i in 1:k
        start = 1 + (i-1)*n_batch_sz
        stop = i == k ? length(negatives_ix) : i * n_batch_sz
        append!(folds[i], negatives_ix[start: stop])
    end

    return [
        (train_indices=Random.shuffle(setdiff(all_ix, f)), 
            test_indices=f)
        for f in folds
    ]
end
