import MLUtils
using Test

@testset "roc_auc" begin
    # Binary classification
    y_true = [0, 1]
    y_score = [0.0, 1]
    @test roc_auc(; y_true=y_true, y_proba=y_score).auc == 1.0
    # assert_array_almost_equal(tpr, [0, 0, 1])
    # assert_array_almost_equal(fpr, [0, 1, 1])
    # assert_almost_equal(roc_auc, 1.0)

    y_true = [0, 1]
    y_score = [1.0, 0]
    # assert_array_almost_equal(tpr, [0, 1, 1])
    # assert_array_almost_equal(fpr, [0, 0, 1])
    @test roc_auc(; y_true=y_true, y_proba=y_score).auc == 0.0   
    #assert_almost_equal(roc_auc, 0.0)

    y_true = [1, 0]
    y_score = [1.0, 1]
    # assert_array_almost_equal(tpr, [0, 1])
    # assert_array_almost_equal(fpr, [0, 1])
    # @test roc_auc(; y_true=y_true, y_proba=y_score).auc == 0.5 skip=true
    #assert_almost_equal(roc_auc, 0.5)

    y_true = [1, 0]
    y_score = [1, 0.0]
    # assert_array_almost_equal(tpr, [0, 0, 1])
    # assert_array_almost_equal(fpr, [0, 1, 1])
    # assert_almost_equal(roc_auc, 1.0)
    @test roc_auc(; y_true=y_true, y_proba=y_score).auc == 1.0

    y_true = [1, 0]
    y_score = [0.5, 0.5]
    #    assert_array_almost_equal(tpr, [0, 1])
    #    assert_array_almost_equal(fpr, [0, 1])
    # assert_almost_equal(roc_auc, 0.5)
    # @test roc_auc(; y_true=y_true, y_proba=y_score).auc == 0.5 skip=true

    y_true = [0, 0]
    y_score = [0.25, 0.75]
    # assert UndefinedMetricWarning because of no positive sample in y_true
    # expected_message = (
    #     "No positive samples in y_true, true positive value should be meaningless"
    # )
    # assert_array_almost_equal(tpr, [0.0, 0.5, 1.0])
    # assert_array_almost_equal(fpr, [np.nan, np.nan, np.nan])
    # @test roc_auc(; y_true=y_true, y_proba=y_score).auc  broken=true # exception expected

    y_score = round.(rand(10),; digits=3)
    y_true = rand(0:1, 10)

    y_score = [0.442, 0.909, 0.529, 0.551, 0.491, 0.396, 0.88, 0.936, 0.437, 0.543]
    y_true = [0, 0, 1, 1, 0, 1, 1, 0, 1, 0]
    @test roc_auc(; y_true=y_true, y_proba=y_score).auc ≈ 0.32
    @test roc_auc(; y_true=1 .- y_true, y_proba=y_score).auc ≈ 0.68

    y_true = [1, 1, 1, 0, 1, 1, 0, 0, 1, 0]
    y_score = [0.719, 0.705, 0.389, 0.206, 0.653, 0.323, 0.529, 0.868, 0.015, 0.36]
    @test roc_auc(; y_true=y_true, y_proba=y_score).auc == .5

    y_true = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    y_score = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    @test roc_auc(; y_true=y_true, y_proba=y_score).auc == .5

    y_true = [0, 1, 0, 0, 0, 1, 0, 0, 0, 0]
    y_score = [0.083, 0.105, 0.339, 0.461, 0.086, 0.497, 0.008, 0.019, 0.716, 0.042]
    @test roc_auc(; y_true=y_true, y_proba=y_score).auc == .75
    @test roc_auc(; y_true=1 .- y_true, y_proba=y_score).auc == .25

end