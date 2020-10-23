Network{T} = Vector{Matrix{T}}
Network(sizes::Vector{Int}) = [randn(sizes[i], sizes[i - 1] + 1) for i in 2:length(sizes)]

"神经网络的规模"
structof(network::Network) =            # 神经网络的规模等于
    [[size(w, 2) - 1 for w in network]; # 权重矩阵每项的列数
     size(network[end], 1)]             # 最后一项的行数

"获得 `l-1` 层的第 `k` 个神经元到 `l` 层第 `j` 个神经元的连接强度"
w(network::Network, j::Integer, k::Integer, l::Integer) = 
    network[l - 1][j,k]

"获得 `l` 层的第 `j` 个神经元的偏置"
b(network::Network, j::Integer, l::Integer) =
    network[l - 1][j,end]

sigmoid(z::Real) = 1 / (1 + exp(-z))
dsigmoid(z::Real) = (exp_z = exp(-z); exp_z / (1 + exp_z)^2)

"""
    feedforward(network::Network, input::Vector)

前馈

## 参数
- `network`: 神经网络对象
- `input`: 输入向量
"""
function feedforward(network::Network, input::Vector)
    output = [sigmoid.(input)]
    for w in network
        push!(output, sigmoid.(w * [output[end]; 1]))
    end
    output
end

import Random

"""
    sgd!(network::Network, training_data, epochs, mini_batch_size, eta, test_data)

用随机梯度下降训练神经网络

## 参数
- `network`: 神经网络对象
- `training_data`: 训练集
- `epochs`: 使用完整训练集的次数
- `mini_batch_size`: 批量规模
- `η`: 学习率
- `test_data`: 测试集
"""
function sgd!(
    network::Network,
    training_data::Vector{T},
    epochs::Integer,
    mini_batch_size::Integer,
    η::Real,
    test_data::Union{Vector,Nothing}=nothing
) where T <: Real
    # 分配内存
    ∇w = [zeros(size(w)) for w in network]
    as = [zeros(n) for n in structof(network)]
    zs = [zeros(n) for n in structof(network)]
    # 全部训练集使用 `epochs` 次
    for _ in 1:epochs
        Random.shuffle!(training_data)
        # 训练集分为 `length(training_data) ÷ mini_batch_size` 批
        @time for i in 1:(length(training_data) ÷ mini_batch_size)
            fill!.(∇w, .0)
            # 用每组数据执行反向传播
            for (input, label) in training_data[(1:mini_batch_size) .+ (i - 1)mini_batch_size]
                as[1][:] = input
                zs[1][:] = sigmoid.(input)
                backprop!(network, as, zs, label)
                # ∇w .+= ∇w
            end
            network .-= η * ∇w
        end
    end

    if test_data === nothing
    else
    end
end
        
"""
    backprop!(network, as, zs)

反向传播

## 参数
- `network`: 神经网络对象
- `as`: 带权输入存储
- `zs`: 输出存储
- `label`: 标签
"""
function backprop!(
    network::Network,
    as::Vector{Vector{T}},
    zs::Vector{Vector{T}},
    label::Vector{T}
) where T <: Real
    # 前向传播
    for (i, w) in enumerate(network)
        as[i + 1][:] = w * [zs[i]; 1]
        zs[i + 1][:] = sigmoid.(as[i + 1])
    end
    δL = (as[end] .- label) .* dsigmoid.(zs[end])
end