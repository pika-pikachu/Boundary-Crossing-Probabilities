using Distributions 

@doc """
	g(t) 

Returns the Daniel's boundary at time t. 
""" -> 
function g(t)
	return 0.5 - t*log(0.25*(1 + sqrt( 1 + 8*exp(-1/t) ) ) )
end

@doc """
	exact_limit(T = 1) 

Returns the exact probability that a Brownian motion crosses Daniel's boundary
""" ->
function exact_limit(T = 1)
	if T == 0 
	  return 0
	end
	a1 = a2 = 1/2
	a = 1
	b1 = cdf(Normal(0 ,sqrt(T)), g(T)) 
	b2 = a1*cdf(Normal(1,sqrt(T)),g(T)) 
	b3 = a2*cdf(Normal(2,sqrt(T)),g(T)) 
	return 1 - (b1 - (b2 + b3)/a)
end

@doc """
	g(t, x0, t0, n)

n-piecewise linear approximation of g(t)
n: Maximum number of boundary partitions
""" -> 
function g(t, x0, t0, n = 0)
if n == 0
	return g(t + t0) - x0
else
	ti = floor.(t.*n)./n
	m = n.*(g(ti + 1 ./n) - g(ti))
	c = g(ti) - ti.*m
	return m.*(t-t0) + c
end
end

# t0 = 0
# N = 1000
# t_vec = range(t0; stop = 1, length = N)
# vector = zeros(N)
# for i in 1:N
# 	vector[i] = g(t_vec[i], 0, t0, 3) 
# end

# plot(t_vec, vector)

function K_gen(N, n = 0, T = 1, x0 = 0, t0 = 0)
h = (T-t0)/N
K = zeros(N,N)
function b(t)
	# return (g(t,x0,n) - g(t0,x0,n))/(t-t0)
	return 0
end
for j = 1:N
	for i = 1:j
		t = j*h
		s = (2i-1)*h/2
		K[j,i] = cdf(Normal(0,1), ( g(s,x0,t0,n) - g(t,x0,t0,n) )/sqrt(t-s)) + 
				exp( -2*b(t)*( g(t,x0,t0,n) - g(s,x0,t0,n) - (t-s)*b(t) ) )*
				cdf(Normal(0,1), ( g(s,x0,t0,n) - g(t,x0,t0,n) + 2*(t-s)*b(t) )/sqrt(t-s))
	end
end
return K
end


function F_vec(N, n = 0, T = 1, x0 = 0, t0 = 0)
h = (T-t0)/N
F = zeros(N)
function b(t)
	# return (g(t,x0,n) - g(t0,x0,n))/(t-t0)
	return 0
end
for i = 1:N
	t = i*h
	F[i] = cdf(Normal(0,1), -g(t,x0,t0,n)/sqrt(t)) + 
			exp(-2*b(t)*(g(t,x0,t0,n) - t*b(t)))*cdf(Normal(0,1), (-g(t,x0,t0,n) + 2t*b(t))/sqrt(t))
end
return F
end

@doc """
	BCP_LD(N, n, T, x0, t0)

Returns approximated boundary crossing probability (Loader & Deely 1987)
N: Number of partitions
n: Number of piecewise linear partitions. Set to 0 to keep the original boundary.
T: Terminal time
x0: initial position of BM
t0: inital time of BM
""" -> 
function BCP_LD(N, n = 0, T = 1, x0 = 0, t0 = 0)
	return sum(K_gen(N, n, T, x0, t0)\F_vec(N, n, T, x0, t0))
end




using PyPlot
# using PyCall
# @pyimport matplotlib.patches as patch

@doc """
	guideplot(N, s)

Returns a plot with 4 convergence lines n^{-1/2}, n^{-1}, n^{-2}, n^{-4}
N: Maximum number of boundary partitions
s: constant scaling
""" -> 
function guideplot(N, s = 0.05)
	# plt.xscale("log")
	# plt.yscale("log")
	plt[:xscale]("log")
	plt[:yscale]("log")
	grid("on", lw = 0.5)
	plot(1:N, 1 ./((1:N).^0.5)*s, label = L"$1/\sqrt{n}$", ls = "--", color = "black")
	plot(1:N, 1 ./(1:N)*s, label = L"1/n", ls = "--", color = "black")
	# plot(1:N, 1 ./((1:N).^1.5)*s, label = L"1/n^1.5", ls = "--", color = "black")
	plot(1:N, 1 ./((1:N).^2)*s, label = L"1/n^2", ls = "--", color = "black")
	# plot(1:N, 1 ./((1:N).^3.5)*s, label = L"1/n^3.5", ls=  "--", color = "black")
	plot(1:N, 1 ./((1:N).^4)*s, label = L"1/n^4", ls=  "--", color = "black")
	# plot(1:N, 1 ./((1:N).^6)*s, label = L"1/n^6", ls=  "--", color = "black")
	xticks(unique(vcat(1:10, 10:10:100)))
end


@doc """
	converge(n, N, p, T, x0, lb)

Returns a convergence plot of the MC approximation towards the true solution
n: number of equally spaced points
N: maximum number of boundary partitions

converge(10,2000)
""" -> 
function converge(n, N, p = 1, T = 1, x0 = 0, lb = -3)
limit = exact_limit()
n_mesh = unique(floor.(Int,exp.( log(N)*( 0:(1/(n-1)):1 ) ) ) ) # cuts up the x -l og axis uniformly to save computation
bcp_vec = zeros(length(n_mesh))
for i in 1:length(n_mesh)
	bcp_vec[i] = abs(BCP_LD(n_mesh[i]) - limit)
end
figure()
guideplot(N, bcp_vec[1])
plot(n_mesh, bcp_vec, marker="o")
xlabel("n")
ylabel(L"|P_n - P|")
title(L"Error, $|P - P_n|$")
end