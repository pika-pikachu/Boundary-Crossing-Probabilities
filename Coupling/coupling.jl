using StatsBase 
using PyPlot
using Distributions

@doc """
	g(t = 1) 

Returns the Daniel's boundary at time t. 
""" -> 
function g(t)
	return 0.5 - t*log(0.25*(1 + sqrt( 1 + 8*exp(-1/t) ) ) )
end

@doc """
	exact_limit(T = 1 , theta = 1) 

Returns the exact probability that a Brownian motion crosses Daniel's boundary with parameter theta
""" -> 
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

@doc """
	generate(n) 

Generates 4 different approximations of BM from the same uniform random vector:

X_h: Markov chain approximation
X: Exact BM at time partitions 
X_snap: Snaps the BM to h-grid
X_snap_incr: Cumulative sum of h-grid snapped increments
""" -> 
function generate(n)
dt = 1/n
h = 1/n
values = (-(7*n):(7*n))*h
f(x) = exp(-x^2/dt/2)/sqrt(2*pi*dt)*h
prob =  Weights(map(f, values))
cdf_h = cumsum(prob)
U = rand(n)
Xi = zeros(n)
Xi_h = zeros(n)
dif = zeros(n)
Xi = quantile.(Normal(0,sqrt(dt)), U)
Xi_snap = round.(quantile.(Normal(0,sqrt(dt)), U)/h)*h
	for i in 1:n
		Xi_h[i] = values[U[i] .< cdf_h][1] 
		dif[i] = abs(Xi_h[i] - Xi[i])
	end
X_h = cumsum([0; Xi_h])
X = cumsum([0; Xi])
X_snap = zeros(n+1)
X_snap_incr = cumsum([0; Xi_snap]) 
	for i in 1:length(X)
		X_snap[i] = round(X[i]/h)*h
	end
return [X_h, X, X_snap, X_snap_incr]
end

# n = 20
# s = generate(n)
# diff_h = abs.(s[2] - s[1])
# diff_snap = abs.(s[2] - s[3])
# plt[:hist](diff_h, bins = 40, alpha = 0.3)

# for n in 1:5:20
# 	plt[:hist](generate2(n,10000)[1], bins = 40, density = true, cumulative = true, alpha = 0.3)
# end

# plt[:hist](diff_h, bins = 40, density = true, cumulative = true, alpha =0.3)
# plt[:hist](diff_snap, bins = 40, density = true)

@doc """
	mc(n, N, replace = true, path = true) 

Plots an exact BM and 3 different approximations of BM from the same uniform random vector and their error from the true path

n: Number of time partitions
N: Number of paths generated
replace: (true) flush the plotting window each time
path: (true) plot the paths

mc(30, 10, replace = false, path = true)
""" -> 
function mc(n, N = 1; replace = true, path = true)
h = 1/n
a1 = 0.1
a = 0.3
if replace == true
	plt[:cla]()
end
for i in 1:N
W_n = generate(n)
	if path == true
		plot(0:1/n:1, W_n[1], label = "X_h", color = "red", alpha = a1)
		plot(0:1/n:1, W_n[2], label = "X", color = "black", alpha = a1)
		plot(0:1/n:1, W_n[3], label = "X_snap", color = "blue", alpha = a1)
		plot(0:1/n:1, W_n[4], label = "X_incr_snap", color = "green",   ls=  "--", alpha = a1)	
	end
	plot(0:1/n:1, abs.(W_n[1] - W_n[2]), label = "diff X_h", color = "red",   ls=  "--", alpha = a)
	plot(0:1/n:1, abs.(W_n[3] - W_n[2]), label = "diff X_snap", color = "blue",   ls=  "--", alpha = a)
	plot(0:1/n:1, abs.(W_n[4] - W_n[2]), label = "diff X_snap_incr", color = "green",   ls=  "--", alpha = a)
	plot(0:1/n:1, -ones(n+1)/n, color = "black")
	plot(0:1/n:1, zeros(n+1), color = "black")
	plot(0:1/n:1, ones(n+1)/n, color = "black")
end
#legend()
xticks(0:1/n:1)
yticks(((-3*n):(3*n))*h)
grid("on", lw = 0.5)
end

@doc """
	mult(X) 

Returns the probability a discretely monitored X (assumed to be BM) crosses a n-piecewise-linear boundary
Uses 

n: Number of time partitions
N: Number of paths generated
replace: (true) flush the plotting window each time
path: (true) plot the paths

""" -> 
function mult(X, antithetic = true)
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
	if antithetic = true
		return (prod(vec1) + prod(vec2))/2
	else 
		return prod(vec1)
	end

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
bcp_vec_Xh = zeros(length(n_mesh))
bcp_vec_X = zeros(length(n_mesh))
bcp_vec_snap = zeros(length(n_mesh))
bcp_vec_incr_snap = zeros(length(n_mesh))
while 1 == 1
	for j in 1:length(n_mesh)
		S = generate(n_mesh[j])
		Xh_j = S[1] 
		X_j = S[2]
		Xsnap_j = S[3]
		snap_incr_j = S[4]
		bcp_vec_Xh[j] = bcp_vec_Xh[j]*i/(i+1) +  mult(Xh_j)/(i+1)
		bcp_vec_X[j] = bcp_vec_X[j]*i/(i+1) +  mult(X_j)/(i+1)
		bcp_vec_snap[j] = bcp_vec_snap[j]*i/(i+1) +  mult(Xsnap_j)/(i+1)
		bcp_vec_incr_snap[j] = bcp_vec_incr_snap[j]*i/(i+1) +  mult(snap_incr_j)/(i+1)
	end
	if i % 10000 == 0
		# fitting linear model
		error_vec_Xh = abs.(bcp_vec_Xh - ones(length(n_mesh)) .* exact_limit())
		error_vec_X = abs.(bcp_vec_X - ones(length(n_mesh)) .* exact_limit())
		error_vec_snap = abs.(bcp_vec_snap - ones(length(n_mesh)) .* exact_limit())
		error_vec_incr_snap = abs.(bcp_vec_incr_snap - ones(length(n_mesh)) .* exact_limit())
		#dif = abs.(bcp_vec2 - bcp_vec1) 
		x = log.(n_mesh)
		y = log.(error_vec_Xh)

		beta_1 = cov(x,y)/var(x)
		# beta_0 = mean(y) - beta_1*mean(x)
		beta_0 = y[1] - beta_1*x[1]

		y_fit = exp.( beta_0*ones(N) + beta_1*log.(1:N) )

		plt[:cla]()
		guideplot(N)
		plot(n_mesh, error_vec_Xh, marker = "o", label = "X_h error", color = "red")	
		plot(n_mesh, error_vec_X, marker = "o", label = "X error", color = "black")	
		plot(n_mesh, error_vec_snap, marker = "o", label = "X snap error", color = "blue")	
		plot(n_mesh, error_vec_incr_snap, marker = "o", label = "X incr snap error", color = "purple")
		plot(1:N, y_fit)	
		plt[:pause](0.0001)
		print("iteration: ", i , " slope: ", beta_1, "\r")
		legend()
	end
	i += 1
end
end

dyn_conv_plot(10,100)



