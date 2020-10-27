import Random, LinearAlgebra

mat_mul! = LinearAlgebra.mul!
Network{T} = Vector{Matrix{T}}

"""
    Network(sizes)

用随机初始权构造神经网络
每行是一个神经元
"""
Network(sizes::Vector{Int}) = [Random.randn(sizes[i], sizes[i - 1] + 1) for i in 2:length(sizes)]

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
    output = input
    for w in network
        output = sigmoid.(w * [output; 1])
    end
    output
end

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
    training_data::Vector,
    epochs::Integer,
    mini_batch_size::Integer,
    η::Real,
    test_data::Union{Vector,Nothing}=nothing
)
    η /= mini_batch_size
    # 分配内存
    ∇w = [zeros(size(w)) for w in network]
    network_struct = structof(network)
    as = [ones(n + 1) for n in network_struct]
    zs = [zeros(n) for n in network_struct[2:end]]
    δs = [zeros(n) for n in network_struct[2:end]]
    # 全部训练集使用 `epochs` 次
    for _ in 1:epochs
        Random.shuffle!(training_data)
        # 训练集分为 `length(training_data) ÷ mini_batch_size` 批
        @time for i in 1:(length(training_data) ÷ mini_batch_size)
            # 用每组数据执行反向传播
            for (input, label) in view(training_data, (1:mini_batch_size) .+ (i - 1)mini_batch_size)
                as[1][1:end - 1] = input
                # 前向传播
                for (i, w) in enumerate(network)
                    mat_mul!(zs[i], w, as[i])
                    map!(sigmoid, as[i + 1], zs[i])
                    map!(dsigmoid, δs[i], zs[i])
                end
                # 反向传播
                @views @. δs[end] *= as[end][1:end - 1] - label
                for i in length(network):-1:2
                    @views δs[i - 1] .*= transpose(network[i][:,1:end - 1]) * δs[i]
                end
                for i in 1:length(∇w)
                    mat_mul!(∇w[i], δs[i], transpose(as[i]), -η, 1)
                end
            end
            for (w, x) in zip(network, ∇w) 
                w .+= x 
                x .= 0
            end
        end
        
        if test_data === nothing
        else
            m = count(test_data) do (input, label) argmax(feedforward(network, input)) == argmax(label) end
            n = length(test_data)
            println("$(100m / n)%($m / $n)")
        end
    end
end
