using Ripserer
using Distances
using LinearAlgebra
using NPZ
using Printf

function get_representatives(data, k; metric=Distances.euclidean)
    dgms = ripserer(data; reps=true, metric=metric)
    h1 = dgms[2]
    
    if isempty(h1)
        @printf("Не найдено ни одного цикла в H₁.\n")
        return dgms, []
    end
    
    k_ = min(length(h1), k)
    most_persistent = h1[end-k_+1:end]
    
    filtration = h1.filtration
    
    cycles = [reconstruct_cycle(filtration, mpc) for mpc in most_persistent]
    @printf("Найдено %d циклов (из H₁) на выборке из %d точек.\n", k_, length(data))
    
    return dgms, cycles
end

function save_holes_to_txt(cycles, embedding_keys, X, filename)
    open(filename, "w") do io
        if isempty(cycles)
            @warn "Список циклов H₁ пуст. Файл $(filename) будет пустым или содержать заметку."
            println(io, "Нет найденных 1-мерных дыр (H₁).")
            return
        end
        
        for (i, cycle) in enumerate(cycles)
            all_verts = Set{Int}()
            for simplex in cycle
                v_inds = vertices(simplex)
                union!(all_verts, v_inds)
            end
            
            hole_description = String[]
            for v in sort(collect(all_verts))
                word = embedding_keys[v]
                coords = X[v, :]
                coords_str = join(coords, " ")
                push!(hole_description, "$word")
            end
            
            line_str = join(hole_description, " | ")
            println(io, line_str)
        end
    end
    @printf("Данные о циклах H₁ сохранены в '%s'.\n", filename)
end
function main()
    npz_data = npzread("russian_nofraglit_SVD_dict_subset.npz")
    
    embedding_keys = collect(keys(npz_data))
    
    X = hcat([npz_data[k] for k in embedding_keys]...)'
    X = Matrix(X) 
    
    n_points = size(X, 1)
    data = [X[i, :] for i in 1:n_points]
    @info "Считано $n_points точек, размерность = $(size(X,2))."
    
    dgms, cycles = get_representatives(data, 20; metric=Distances.euclidean)
    
    save_holes_to_txt(cycles, embedding_keys, X, "holes_bounds_SVD_8.txt")
end

main()