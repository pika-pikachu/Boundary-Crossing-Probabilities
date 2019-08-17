using PyPlot
# using PyCall
# @pyimport matplotlib.patches as patch
using CPUTime # used for timing
using QuadGK # used for quadrature in the lower boundary estimate
using Optim # finding minimum of the function


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
	pvec = [0.5, 1, 1.5, 2, 3, 4]
	for i in 1:length(pvec)
		plot(1:N, 1 ./( (1:N).^(pvec[i]) )*s, ls = "--", color = "black")
		plt[:text](N + 0.3, 0.5*s/N^(pvec[i]) , join(["n^-", pvec[i]]), fontsize = 9)
	end	
	xticks(unique(vcat(1:10, 10:10:100, 100:100:1000, 1000:1000:10000)))
end


@doc """
	converge(n, N, p, T, x0, lb, γ, lock_h2, error_bounds)

Returns a convergence plot of the MC approximation towards the true solution
n: number of equally spaced points
N: maximum number of boundary partitions
p: h = γn^{-p}
T: Time horizon
x0: initial point
lb: lower boundary truncation
γ: h = γn^{-p}
lock_h2: guideplot locking last point to h^2

converge(10,100, p = 1, lb = -3)
converge(10,100, p= 0.6, lb = -3, error_bounds = 1) # last partition mod
""" -> 
function converge(n, N; 
	p = 1, 
	T = 1, 
	x0 = 0, 
	lb = -3, 
	γ = 1, 
	lock_h2 = 1, 
	error_bounds = 0)
limit = exact_limit()

# cuts up the x-log axis uniformly to save computation
n_mesh = unique(floor.(Int,exp.( log(N)*( 0:(1/(n-1)):1 ) ) ) ) 
bcp_vec = zeros(length(n_mesh))
time_vec = zeros(length(n_mesh))
	for i in 1:length(n_mesh)
		CPUtic()
		bcp_vec[i] = abs(BCP(n_mesh[i], γ/n_mesh[i]^p, T, x0, lb) - limit)
		time_vec[i] = CPUtoc()
	end
figure()

# locking the guide grid, useful when using the last partition mod
if lock_h2 == 1
	guideplot(N, N^2*bcp_vec[end])
else	
	guideplot(N, N^(2*p)*bcp_vec[end])
end

# plot of the error convergence
plot(n_mesh, bcp_vec, marker="o", label = L"|P_n - P|")

if error_bounds == 1
# scaling constant error
	plot(2:N, collect(2:N) .* log.(2:N) .* exp.(-pi^2 .* ( collect(2:N) ./ γ ) .^(2*p) ./ collect(2:N) ), color = "red" , label = "scal. error")

	# daniels boundary estimates
	K = 0.69314718 # bound on first derivative
	gamma = 2 # bound on second derivative of boundary
	plot(1:N, 1 ./ collect(1:N) .^2 .* (0.625*K + 0.5) .* gamma ./ 2, label = "bd err." )

	# lower boundary error 
	g_min = Optim.minimum(optimize(g,0,T))
	I, err = quadgk(s -> 2*ccdf(Normal(0,1), (abs(lb) + g_min)/sqrt(T-s))*abs((abs(lb) + g_min))*exp(-(abs(lb) + g_min)^2/(2*s))/sqrt(2*pi*s^3), 0, T, rtol = 1e-5)
	plot(1:N, ones(N).*I, color = "green", label = "trunc. err.")
	axis([1, N+1, I/10, 1])	
end
# time elapsed plotting
	for i in 1:length(n_mesh)
		plt[:text](n_mesh[i] + 0.3, bcp_vec[i], join([time_vec[i],"s"]), fontsize = 9)
	end

legend()
xlabel(L"n")
ylabel(L"|P_n - P|")
title( join(["Error convergence ","p=",p, ","," lb=",lb]))
end


