import MLUtils
using Test
using Random

@testset "StratifiedKfold" begin
    y = Random.shuffle(append!(zeros(Int64, 900), ones(Int64, 100)))
    folds = StratifiedKfold(y, 3)
        
    @test sum(y) == 100
    @test length(intersect(folds[1].test_indices, folds[2].test_indices)) == 0
    @test length(intersect(folds[1].test_indices, folds[3].test_indices)) == 0
    @test length(intersect(folds[2].test_indices, folds[3].test_indices)) == 0

    for i in 1:3
        @test sum(y[folds[i].train_indices]) + sum(y[folds[i].test_indices]) == 100
    end
    
    @test length(intersect(folds[1].train_indices, folds[1].test_indices)) == 0
    @test length(intersect(folds[2].train_indices, folds[2].test_indices)) == 0
    @test length(intersect(folds[3].train_indices, folds[3].test_indices)) == 0

    @test length(Set(union(folds[1].train_indices,folds[2].train_indices,folds[3].train_indices))) == 1000
    @test sum(y[collect(Set(union(folds[1].test_indices, folds[2].test_indices, folds[3].test_indices)))]) == 100

end