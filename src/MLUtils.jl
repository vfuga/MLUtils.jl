module MLUtils

include("tf_idf.jl")
include("roc_auc.jl")
include("f1_score.jl")
include("kfold.jl")
include("knn.jl")
include("r2.jl")

export TfIdf_Transformer
export roc_auc
export f1_score
export StratifiedKfold
export KNN
export r2, r_squared

end # module
