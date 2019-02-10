set JULIA_NUM_THREADS=4
julia

using Pkg
Pkg.add("CuArrays")
Pkg.test("CuArrays") # fails



using CUDAdrv
using CUDAnative
using CuArrays

M = 10^5
@time CuArray(rand(M))
@time rand(M)


N = 2^20
x = fill(1.0f0, N)  # a vector filled with 1.0 (Float32)
y = fill(2.0f0, N)  # a vector filled with 2.0

y .+= x             # increment each element of y with the corresponding element of x

# check that we got the right answer
using Test
@test all(y .== 3.0f0)

cufill(1.0f0,3)




function sequential_add!(y, x)
    for i in eachindex(y, x)
        @inbounds y[i] += x[i]
    end
    return nothing
end

fill!(y, 2)
sequential_add!(y, x)
@test all(y .== 3.0f0)



# parallel implementation
function parallel_add!(y, x)
    Threads.@threads for i in eachindex(y, x)
        @inbounds y[i] += x[i]
    end
    return nothing
end

fill!(y, 2)
parallel_add!(y, x)
@test all(y .== 3.0f0)


using BenchmarkTools
@btime sequential_add!($y, $x)
@btime parallel_add!($y, $x)


#######################################################################################
# CUDA ARRAY SUMMING
#######################################################################################

using CuArrays

x_d = cufill(1.0f0, N)  # a vector stored on the GPU filled with 1.0 (Float32)
y_d = cufill(2.0f0, N)  # a vector stored on the GPU filled with 2.0
y_d .+= x_d
@test all(Array(y_d) .== 3.0f0)

function add_broadcast!(y, x)
    CuArrays.@sync y .+= x
    return
end

@btime add_broadcast!(y_d, x_d)



#############################################
# Writing a gpu kernel
#############################################
using CUDAnative

N = 2^20
x_d = cufill(1.0f0, N)  # a vector stored on the GPU filled with 1.0 (Float32)
y_d = cufill(2.0f0, N)  

function gpu_add1!(y, x)
    for i = 1:length(y)
        @inbounds y[i] += x[i]
    end
    return nothing
end

fill!(y_d, 2)
@cuda gpu_add1!(y_d, x_d)
@test all(Array(y_d) .== 3.0f0)


function bench_gpu1!(y, x)
    CuArrays.@sync begin
        @cuda gpu_add1!(y, x)
    end
end

@btime bench_gpu1!(y_d, x_d)

# Matrix multiplication 

N = 1000
x_d = cufill(1.0f0, (N,N))
y_d = cufill(1.0f0, (N,N))

function mult_broadcast!(y, x)
    CuArrays.@sync y * x
    return
end

@btime mult_broadcast!(y_d, x_d)

x_h = fill(1.0f0, (N,N))
y_h = fill(1.0f0, (N,N))

@btime x_h * y_h


z= Array(x_d*y_d)

@test all(Array(y_d) .== 3.0f0)


# FFT
using FFTW
N = 500
@time Ax_h = fft(fill(1.0f0, (N,N)))
@time Ax_d = fft(cufill(1.0f0, (N,N)))

@time ifft(Ax_h)
@time ifft(Ax_d)


@time x_d * x_d * x_d * x_d * x

for i in 1:

@time x_h * x_h * x_h * x_h








using CUDAdrv
using CUDAnative
using CuArrays

@time cu(randn(10^5))
@time randn(CuArray{Float32}, 1)

@time rand(5000,5000)

using GPUArrays



function test()
	n = 10
	return cumsum(randn(n))/sqrt(n)	
end


@time a = cu(cumsum(randn(10^7))/sqrt(10^7))
@time a = cumsum(randn(10^7))/sqrt(10^7)


function mc(x)
	return cu(cumsum(randn(10^6))/sqrt(10^6))
end

function mc(x)
	return cumsum(randn(10^6))/sqrt(10^6)
end

@time map(mc,ones(10))









N = 1000
x_d = cufill(1.0f0, (N,N))

m = 10

p = x_d


function test(N,m)
x_d = cufill(1.0f0, (N,N))
p = x_d
for i in 1:m
	p = p * x_d 
end
return p
end

@cuda test()

@time test(1000,200)


function test2(N,m)
x_d = fill(1.0f0, (N,N))
p = x_d
for i in 1:m
	p = p * x_d 
end
return p
end

@time test2(1000,200)