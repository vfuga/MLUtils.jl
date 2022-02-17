using Lazy

function roc_auc(; y_true::Vector{Int}, y_proba::Vector{Float64})
    if y_true |> Lazy.frequencies |> length == 1
        error("Only one class in y_true. At least two classes are expected.")
    end
    data = @>> (zip(y_true, y_proba) |> collect)  sort(; by= x -> x[2])
    # @>> data foreach(println)
    # TPR = TP / (TP + FN)
    # FPR = FP / (FP + TN)

    tp = (y_true |> frequencies)[1]
    fn = 0
    fp = length(y_true) - tp
    tn = 0
    t = 0.0

    roc = []

    for r in data
        tpr = tp / (tp + fn)
        fpr = fp / (fp + tn)
        
        push!(roc, (tpr, fpr, t))

        y, t = r
        if y == 1
            tp -= 1
            fn += 1
        else
            fp -= 1
            tn += 1
        end
    end
    roc = roc[end:-1:1]
    tpr = [r[1] for r in roc]
    fpr = [r[2] for r in roc]
    thresh = [r[3] for r in roc]

    curve = collect(zip([fp for fp in fpr[1:end-1]], [fp for fp in fpr[2:end]], [r for r in tpr[1:end-1]]))
    auc = sum(map(a -> (a[2] - a[1])*a[3], curve))

    roc = (auc=auc, tpr=tpr, fpr=fpr, thresh=thresh)
    return roc
end

