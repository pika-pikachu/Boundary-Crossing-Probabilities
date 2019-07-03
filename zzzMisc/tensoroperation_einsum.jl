using TensorOperations
using Einsum

x = randn(30,40)
y = randn(30,40,30,30)

#   0.002536 seconds (13 allocations: 7.656 KiB)
@time @tensor begin 
	M[k,l] := x[i,j]*y[i,j,k,l]
end

# 0.230833 seconds (5.47 M allocations: 101.138 MiB, 6.91% gc time)
@time @einsum M[k,l] := x[i,j]*y[i,j,k,l]


x = randn(500,500)
y = randn(100,100,100,100)
z = x


x = zeros(500,500)

@time @einsum M[i,k] := x[i,j]*z[j,k]
@time x*z
@time @tensor begin 
	M[i,k] := x[i,j]*z[j,k]
end


using SparseArrays

x = spzeros(3,3)