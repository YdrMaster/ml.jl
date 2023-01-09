# 反向传播的 4 个公式：
#
# - BP0: δ{l,j} = ∂C/∂z{l,j}
#   > 误差 δ 定义为第 l 层的第 j 个神经元激活前的输出对损失的影响
#   > C: cost function 是一个向量输入的标量函数，或者说一个多元函数
#
# - BP1: δ{L,j} = ∂C/∂z{L,j} = ∂C/∂a{L,j} ⋅ ∂a{L,j}/∂z{L,j} = ∂C/∂a{L,j} ⋅ dσ(z{L,j})
#   > 假设所有的激活函数都是 σ，输出层的误差可以计算出来
#   - BP1a: δ{L} = ∇C .* dσ.(z{L})
#
# - BP2: δ{l} = [δ{l,j} for j in 0:J{l}]
#             = [(δ{l+1} ⋅ w{l+1}[:,i]) ⋅ dσ(z{l,j}) for j in 0:J{l}]
#             = [(w{l+1}[:,i] ⋅ δ{l+1}) for j in 0:J{l}] .* dσ.(z{L})
#             = (w{l+1}'δ{l+1}) .* dσ.(z{L})
#   > 其中每个 δ{l,j} 这样展开：
#   > δ{l,j} = ∂C/∂z{l,j}                                              | 定义
#   >        = Σ{i}(∂C/∂z{l+1,i} ⋅ ∂z{l+1,i}/∂a{l,j} ⋅ ∂a{l,j}/∂z{l,j}) | 链式法则
#   >        = Σ{i}(δ{l+1,i} ⋅ w{l+1,i,j} ⋅ dσ(z{l,j}))                 | 代换
#   >        = Σ{i}(δ{l+1,i} ⋅ w{l+1,i,j}) ⋅ dσ(z{l,j})                 | 非迭代项拿出求和符号
#   >        = (δ{l+1} ⋅ w{l+1}[:,i]) ⋅ dσ(z{l,j})                      | 就是 l+1 层误差点乘 l+1 层权重矩阵的一列，再乘 σ'

import Random, .Iterators.partition

"神经网络"
struct Network
    weights::Vector{Matrix{Float64}}
    Network(sizes::Integer...) = new([Random.randn(sizes[i], sizes[i-1] + 1) for i in eachindex(sizes)[2:end]])
end

"sigmoid function"
σ(z::Real) = 1 / (1 + exp(-z))

"diff sigmoid function"
dσ(z::Real) = (x -> x * (1 - x))(σ(z));

"前馈"
feedforward(network::Network, input::AbstractVector{<:Real}) = foldl((acc, w) -> σ.(w * [acc; 1]), network.weights; init=input)

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
        ∇w .+= _backprop(network, x, y)
    end
    network.weights .-= η .* ∇w
end

"反向传播"
function _backprop(
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
