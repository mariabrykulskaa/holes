using Ripserer
using LinearAlgebra
using Plots
using Distances
using NPZ
using Printf
function get_representatives(data, k, metric=Distances.euclidean)
    diagram_cocycles = ripserer(data; reps=true, metric=metric)
    
    k_ = min(size(diagram_cocycles[2])[1], k)
    
    most_persistent_co = diagram_cocycles[2][end-k_+1:end]
    
    filtration = diagram_cocycles[2].filtration
    cycles = [reconstruct_cycle(filtration, mpc) for mpc in most_persistent_co]
    
    @printf("Выбрано %d циклов из %d точек\n", k_, length(data))
    return diagram_cocycles, cycles
end

function save_intervals(filename::String, intervals)
    bd_mat = isempty(intervals) ? Array{Float64}(undef, 0, 2) :
        reduce(vcat, [reshape([bd[1], bd[2]], 1, 2) for bd in intervals])
    npzwrite(filename, bd_mat)
end

function process_data(
    data_path::String = "avaricsent_dictionary_cbow_last_subset.npz",  
    dir_path::String = "words/AV",                             
    chunk_num::String = "my_subset",                           
    metric = Distances.cosine_dist                             
)
    mkpath(dir_path)
    
    vars = npzread(data_path)
    
    words = collect(keys(vars))
    data = [vars[w] for w in words]
    
    @printf("Всего загружено %d слов (точек)\n", length(data))

    diagram, cycles = get_representatives(data, 500, metric)
    
    plt = plot(diagram)
    display(plt)
    
    save_intervals("$(dir_path)/h0_$(chunk_num).npy", diagram[1])
    save_intervals("$(dir_path)/h1_$(chunk_num).npy", diagram[2])
    
    scycles = [reduce(vcat, [reshape(collect(vertices(sx)), 1, :) for sx in cycle]) for cycle in cycles]
    sscycles = isempty(scycles) ? Array{Int64}(undef, 0, 1) : reduce(vcat, scycles)
    sscycles = Matrix{Int64}(sscycles)
    
    npzwrite("$(dir_path)/ru_word_holes_$(chunk_num).npy", sscycles)
    
    return diagram, cycles
end

diagram, cycles = process_data("avaricsent_dictionary_cbow_last_subset.npz", "words/AV", "my_subset", Distances.cosine_dist)
