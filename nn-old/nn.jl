import Random, LinearAlgebra

mat_mul! = LinearAlgebra.mul!
Network{T} = Vector{Matrix{T}}

"""
    Network(sizes)

用随机初始权构造神经网络
每行是一个神经元
"""
Network(sizes::Vector{Int}) = [Random.randn(sizes[i], sizes[i-1] + 1) for i in 2:length(sizes)]

"神经网络的规模"
structof(network::Network) =
    [
        [size(w, 2) - 1 for w in network] # 权重矩阵每项的列数
        size(network[end], 1)             # 最后一项的行数
    ]

"获得 `l-1` 层的第 `k` 个神经元到 `l` 层第 `j` 个神经元的连接强度"
w(network::Network, j::Integer, k::Integer, l::Integer) =
    network[l-1][j, k]

"获得 `l` 层的第 `j` 个神经元的偏置"
b(network::Network, j::Integer, l::Integer) =
    network[l-1][j, end]

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
    training_data::Matrix,
    epochs::Integer,
    mini_batch_size::Integer,
    η::Real,
    test_data::Union{Matrix,Nothing}=nothing
)
    η /= mini_batch_size
    n = size(training_data, 2)
    # 分配内存
    ∇w = [zeros(size(w)) for w in network]
    network_struct = structof(network)
    println(network_struct)
    as = [ones(n + 1) for n in network_struct]
    zs = [zeros(n) for n in network_struct[2:end]]
    δs = [zeros(n) for n in network_struct[2:end]]
    buffer_mat_mul = [zeros(n) for n in network_struct[2:end]]
    errors = Float64[]
    # 全部训练集使用 `epochs` 次
    for _ in 1:epochs
        indices = Random.shuffle(1:n)
        # 训练集分为 `length(training_data) ÷ mini_batch_size` 批
        @time for i in 1:(n÷mini_batch_size)
            @views _indices = indices[(1:mini_batch_size).+(i-1)mini_batch_size]
            # 用每组数据执行反向传播
            for index in _indices
                @views as[1][1:end-1] .= training_data[1:network_struct[1], index]
                # 前向传播
                for (i, w) in enumerate(network)
                    mat_mul!(zs[i], w, as[i])
                    map!(sigmoid, as[i+1], zs[i])
                    map!(dsigmoid, δs[i], zs[i])
                end
                # 反向传播
                @views δs[end] .*= as[end][1:end-1] .- training_data[network_struct[1]+1:end, index]
                for i in length(network):-1:2
                    @views mat_mul!(buffer_mat_mul[i-1], transpose(network[i][:, 1:end-1]), δs[i])
                    δs[i-1] .*= buffer_mat_mul[i-1]
                end
                for i in eachindex(∇w)
                    mat_mul!(∇w[i], δs[i], transpose(as[i]), -η, 1)
                end
            end
            for (w, x) in zip(network, ∇w)
                w .+= x
                x .= 0
            end
        end

        if test_data !== nothing
            n = size(test_data, 2)
            m = count(1:n) do i
                # 二进制自动编码
                # a = [round(UInt8, x) for x in feedforward(network, test_data[1:network_struct[1],i])]
                # a == test_data[network_struct[1] + 1:end,i]
                # 概率分布
                argmax(feedforward(network, test_data[1:network_struct[1], i])) == argmax(test_data[network_struct[1]+1:end, i])
            end
            push!(errors, m / n)
            # GR.plot(errors)
            println("$(100m / n)%($m / $n)")
        end
    end
end
