using Distributions
using Bridge
using PyPlot

T = 1.
z = 1.
n = 10^4

tau_T = rand(InverseGaussian(T, T^2))
t_mesh = 0:tau_T/(n-1):tau_T

B1 = sample(t_mesh, WienerBridge(tau_T, z)) 
B2 = sample(t_mesh, WienerBridge(tau_T, z)) 

plot(B1.yy, B2.yy, linewidth = 1)
plt[:axis]([-2.5, 2.5, -2.5, 2.5])
plt[:xlabel](L"W_1(t)")
plt[:ylabel](L"W_2(t)") 


# Generating a Cauchy process

function run_max(X)
Y = zeros(length(X))
run_max = 0
for i in 1:length(X)
	Y[i] = max(X[i],run_max)
	run_max = Y[i] 
end
return Y
end

function B2_gen(B1)
B1M = run_max(B1)
t_b2_mesh = unique(B1M)
B2_vec = zeros(length(t_b2_mesh))
Zi = randn(length(t_b2_mesh))
for i in 2:length(t_b2_mesh)
	B2_vec[i] = B2_vec[i-1] + Zi[i]*sqrt(t_b2_mesh[i] - t_b2_mesh[i-1])
end
return [t_b2_mesh, B2_vec]
end

n = 10^5
B1 = cumsum([0; randn(n-1)/sqrt(n)]) # running time up to 1
C = B2_gen(B1)
plt[:step](C[1], C[2])


function monte_carlo_1(n = 10^4)
B1 = cumsum([0; randn(n-1)/sqrt(n)])
B1M = run_max(B1)
t_b2_mesh = unique(B1M)
C = B2_gen(t_b2_mesh)
plt[:step](t_b2_mesh,C)
end


monte_carlo_1()




n = 10^5
X = cumsum(rand(Cauchy(0,1/n),n))./n
t_mesh = range(0, stop = 1, length = n)
plot(t_mesh, X)

function monte_carlo(n = 10^5)
	X = cumsum(rand(Cauchy(0,1/n),n))./n
	t_mesh = range(0, stop = 1, length = n)
	plot(t_mesh, X)
end

monte_carlo()