include("nn.jl")
if (!@isdefined training_data)
    # 读取数据
    data = open("train-images.idx3-ubyte", "r") do file
        _ = ntoh(read(file, UInt32))
        n = ntoh(read(file, UInt32))
        r = ntoh(read(file, UInt32))
        c = ntoh(read(file, UInt32))
        reshape([read(file, UInt8) / 255 for i in 1:n * r * c], (r * c, n))
    end
    # 读取标签
    labels = open("train-labels.idx1-ubyte", "r") do file
        _ = ntoh(read(file, UInt32))
        n = ntoh(read(file, UInt32))
        [read(file, UInt8) for i in 1:n]
    end
    # 构造监督训练集
    training_data = [data; [label == n ? 1 : 0 for n in 0:9, label in labels]]
end
# 构造神经网络
network = Network([size(data, 1), 60, 10])
# 训练网络
sgd!(network, training_data[:,1:50000], 100, 10, 4, training_data[:,50001:end])
