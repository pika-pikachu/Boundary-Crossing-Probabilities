using StatsBase 
using Distributions
using PyPlot

# Daniels Boundary
function g(t)
	return 0.5 - t*log(0.25*(1 + sqrt( 1 + 8*exp(-1/t) ) ) )
end

# Exact non-BCP
function exact_limit(T = 1, theta = 1)
	if T == 0 
	  return 0
	end
	a1 = a2 = 1/2
	a = 1
	b1 = cdf(Normal(0 ,sqrt(T)),g(T)) 
	b2 = a1*cdf(Normal(theta,sqrt(T)),g(T)) 
	b3 = a2*cdf(Normal(2*theta,sqrt(T)),g(T)) 
	return (b1 - (b2 + b3)/a)
end


#generates RW with n steps starting from 0 
#stepsize h  = dt
# dt = 1/n (e.g. 2 steps would mean time intervals of 1/2)
function X(n)
dt = 1/n
h = 1/n
values = (-(7*n):(7*n))*h
f(x) = exp(-x^2/dt/2)/sqrt(2*pi*dt)*h
prob =  Weights(map(f, values))
xi = sample(values, prob, n)
return cumsum([0; xi])
end

# X(5)

# n = 100
# t_vec = 0:(1/n):1
# mapg = map(g, t_vec)
# plot(t_vec, mapg)

function mult(X)
X2 = -X # antithetic variable
n = length(X)-1
vec1 = zeros(n)
vec2 = zeros(n)
t_vec = 0:(1/n):1
	for i = 1:n
		if (X[i+1] < g(t_vec[i+1])) & (X[i] < g(t_vec[i]))
			vec1[i] = 1 - exp( -2 * n * ( g(t_vec[i+1]) - X[i+1] ) * ( g(t_vec[i]) - X[i] ) )
		else
			vec1[i] = 0
		end
		if (X2[i+1] < g(t_vec[i+1])) & (X2[i] < g(t_vec[i]))
			vec2[i] = 1 - exp( -2 * n * ( g(t_vec[i+1]) - X2[i+1] ) * ( g(t_vec[i]) - X2[i] ) )
		else
			vec2[i] = 0
		end
	end
#	close()
#	figure()
#	plot(t_vec, X)
#	mapg = map(g, t_vec)
#	plot(t_vec, mapg)
	return (prod(vec1) + prod(vec2))/2
end



function monte_carlo(N, n)
p_vec = zeros(N)
	for i in 1:N
		p_vec[i] = mult(X(n))
	end
	return abs(mean(vec) - exact_limit())
end

function guideplot(N, s = 0.05)
	plt[:yscale]("log")
	plt[:xscale]("log")
	grid("on")
	plot(1:N, 1 ./((1:N).^0.5)*s, label = L"$1/\sqrt{n}$", ls = "--", color = "black")
	plot(1:N, 1 ./(1:N)*s  ,	     label = L"1/n",     ls = "--", color = "black")
	plot(1:N, 1 ./((1:N).^2)*s ,   label = L"1/n^2",   ls = "--", color = "black")
	plot(1:N, 1 ./((1:N).^4)*s ,   label = L"1/n^4",   ls=  "--", color = "black")
end




function dyn_conv_plot(n, N)
i = 1
n_mesh = unique(floor.(Int,exp.( log(N)*( 0:(1/(n-1)):1 ) ) ) ) # cuts up the x - log axis uniformly to save computation
bcp_vec = zeros(length(n_mesh))
while 1 == 1
	for j in 1:length(n_mesh)
		bcp_vec[j] = bcp_vec[j]*i/(i+1) +  mult(X(n_mesh[j]))/(i+1)
	end
	if i % 10000 == 0
		# fitting linear model
		error_vec = abs.(bcp_vec - ones(length(n_mesh)) .* exact_limit())
		x = log.(n_mesh)
		y = log.(error_vec)

		beta_1 = cov(x,y)/var(x)
		# beta_0 = mean(y) - beta_1*mean(x)
		beta_0 = y[1] - beta_1*x[1]

		y_fit = exp.( beta_0*ones(N) + beta_1*log.(1:N) )

		plt[:cla]()
		guideplot(N)
		plot(n_mesh, error_vec, marker = "o")	
		plot(1:N, y_fit)	
		plt[:pause](0.0001)
		print("iteration: ", i , " slope: ", beta_1, "\r")
	end
	i += 1
end
end

dyn_conv_plot(10,100)