"生成一组二维环状数据。"
function make_circle(number::Integer, noise::Real, radius::Real=1.0)
    [radius * f(2π * i / number) + noise * randn() for f in [cos, sin], i in 1:number]
end
