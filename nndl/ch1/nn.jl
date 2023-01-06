import Random, .Iterators.partition

"神经网络"
struct Network
    weights::Vector{Matrix{Float64}}
    Network(sizes::Integer...) = new([Random.randn(sizes[i], sizes[i-1] + 1) for i in eachindex(sizes)[2:end]])
end

"sigmoid function"
σ(z) = 1 / (1 + exp(-z))

"diff sigmoid function"
dσ(z) = (x -> x * (1 - x))(σ(z));

"前馈"
feedforward(network::Network, input::Vector) = foldl((acc, w) -> σ.(w * [acc; 1]), network.weights; init=input)

"数据集"
DataSet = AbstractVector{Tuple{AbstractVector{<:Real},Real}}

"随机梯度下降"
function sgd(
    network::Network,
    training_data::DataSet,
    η::Real,
    epochs::Integer,
    mini_batch_size::Integer,
    test_data::Union{DataSet,Nothing}=nothing,
)
    for epoch in 1:epochs
        Random.shuffle!(training_data)
        for mini_batch in partition(training_data, mini_batch_size)
            _update_mini_batch!(network, mini_batch, η)
        end
        if test_data !== nothing
            @show evaluate(network, test_data)
        else
            @show epoch
        end
    end
end

"小批量更新"
function _update_mini_batch!(
    network::Network,
    mini_batch::DataSet,
    η::Real,
)
    η /= length(mini_batch)
    ∇w = [zeros(size(w)) for w in network.weights]
    for (x, y) in mini_batch
        ∇w .+= backprop(network, x, y)
    end
    network.weights .-= η .* ∇w
end

"反向传播"
function backprop(
    network::Network,
    x::AbstractVector{<:Real},
    y::Real,
)
    ∇w = [zeros(size(w)) for w in network.weights]

    as = [x] # activations 保存每层所有激活值
    zs = []  # 保存每层的所有 z
    for w in network.weights
        z = w * [as[end]; 1]
        push!(zs, z)
        a = σ.(z)
        push!(as, a)
    end
    δ = (as[end] - y) .* dσ.(zs[end])
    ∇w[end] = δ * [as[end-1]; 1]'
    for l in 2:length(network.weights)
        δ = (network.weights[end-l+1]'*δ)[1:end-1] .* dσ.(zs[end-l])
        ∇w[end-l+1] = δ * [as[end-l]; 1]
    end
    ∇w
end
