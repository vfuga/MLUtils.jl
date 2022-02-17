using Test
using MLUtils

X = [0 1 1 1 0 0 1 0 1;
0 2 0 1 0 1 1 0 1;
1 0 0 1 1 0 1 1 1;
0 1 1 1 0 0 1 0 1]


function smooth_l2_idf()
    T = [
        0.0       0.469791  0.580286  0.384085  0.0       0.0       0.384085  0.0       0.384085
        0.0       0.687624  0.0       0.281089  0.0       0.538648  0.281089  0.0       0.281089
        0.511849  0.0       0.0       0.267104  0.511849  0.0       0.267104  0.511849  0.267104
        0.0       0.469791  0.580286  0.384085  0.0       0.0       0.384085  0.0       0.384085
    ]
    m = TfIdf_Transformer.fit(X; smooth=true, norm=:l2)
    return all(round.(TfIdf_Transformer.transform(m, X);digits=6) .== T)
end  

@test smooth_l2_idf()

function smooth_l1_idf()
    T = [
        0.0       0.213315  0.263487  0.174399  0.0       0.0       0.174399  0.0       0.174399
        0.0       0.33226   0.0       0.135822  0.0       0.260274  0.135822  0.0       0.135822
        0.219033  0.0       0.0       0.1143    0.219033  0.0       0.1143    0.219033  0.1143
        0.0       0.213315  0.263487  0.174399  0.0       0.0       0.174399  0.0       0.174399
    ]
    m = TfIdf_Transformer.fit(X; smooth=true, norm=:l1)
    return all(round.(TfIdf_Transformer.transform(m, X);digits=6) .== T)
end  
@test smooth_l1_idf()

function smooth_l1_noidf()
    T = [
        0.0       0.2       0.2  0.2       0.0       0.0       0.2       0.0       0.2
        0.0       0.333333  0.0  0.166667  0.0       0.166667  0.166667  0.0       0.166667
        0.166667  0.0       0.0  0.166667  0.166667  0.0       0.166667  0.166667  0.166667
        0.0       0.2       0.2  0.2       0.0       0.0       0.2       0.0       0.2
    ]
    m = TfIdf_Transformer.fit(X; smooth=true, norm=:l1, use_idf=false)
    return all(round.(TfIdf_Transformer.transform(m, X);digits=6) .== T)
end  

@test smooth_l1_noidf()


function smooth_l2_noidf()
    T = [
        0.0       0.447214  0.447214  0.447214  0.0       0.0       0.447214  0.0       0.447214
        0.0       0.707107  0.0       0.353553  0.0       0.353553  0.353553  0.0       0.353553
        0.408248  0.0       0.0       0.408248  0.408248  0.0       0.408248  0.408248  0.408248
        0.0       0.447214  0.447214  0.447214  0.0       0.0       0.447214  0.0       0.447214
    ]
    m = TfIdf_Transformer.fit(X; smooth=true, norm=:l2, use_idf=false)
    return all(round.(TfIdf_Transformer.transform(m, X);digits=6) .== T)
end  

@test smooth_l2_noidf()

