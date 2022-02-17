# MLUtils.jl

Example:
========
````Julia
using MLUtils

X = [0 1 1 1 0 0 1 0 1
     0 2 0 1 0 1 1 0 1
     1 0 0 1 1 0 1 1 1
     0 1 1 1 0 0 1 0 1]
m = TfIdf_Transformer.fit(X; smooth=true, norm=:l2)
X_ = TfIdf_Transformer.transform(m, X)

#      0.0       0.469791  0.580286  0.384085  0.0       0.0       0.384085  0.0       0.384085
#      0.0       0.687624  0.0       0.281089  0.0       0.538648  0.281089  0.0       0.281089
#      0.511849  0.0       0.0       0.267104  0.511849  0.0       0.267104  0.511849  0.267104
#      0.0       0.469791  0.580286  0.384085  0.0       0.0       0.384085  0.0       0.384085
````
