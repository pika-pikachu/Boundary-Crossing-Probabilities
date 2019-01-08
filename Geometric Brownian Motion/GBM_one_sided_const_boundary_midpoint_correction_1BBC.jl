using PyPlot
using Distributions

struct MeshParams
	n::Int64   
    h::Float64
    T::Float64
end

function Phi(z)
	return cdf(Normal(),z)
end

@doc """
	exact_GBM(T)

Returns the exact probability that a Geometric Brownian Motion process crosses the boundary g(t) 
""" -> 
function exact_GBM(T = 1)
	h = exp(1)
	r = 0
	σ = 1
	x0 = 1
	return 1 - (Phi(((σ^2/2 - r)*T + log(h/x0))/σ/sqrt(T)) - exp((2r - σ^2)*log(h/x0)/σ^2)*Phi(((σ^2/2 - r)*T - log(h/x0))/σ/sqrt(T)))
end

@doc """
	bbb(x0, x1, t0, t1, T, μ, σ, g) 

Returns the exact probability that a Brownian bridge starting at x0 and ending at x1 survives 
a two piecewise linear boundary approximation of g(t): g(t), t0  <= t <= t1. 
μ and σ are the coefficients for the diffusion SDE dXt = μ(Xt)dt + σ(Xt)dWt.
""" -> 
function bbb(x0, x1, t0, t1, T, μ::Function, σ::Function, g::Function)
        dt = t1 - t0
        # gt1 = (g(t1,T)-μ(t0,x0)*dt)/σ(t0,x0)
        gt1 = g(t1,T)/σ(t0,x0)
        gt0 = g(t0,T)
        return 1 - exp(-2/dt*log(gt0/x0)*(gt0*(gt1 - gt0) + log(gt0/x1)))
end

# bbb(1.9,
# 	1.9,
# 	0,
# 	1/1,
# 	1,
# 	(t,x) -> 0, 
# 	(t,x) -> 1,
# 	(t,T) -> exp(1))

# function bbb(x0, x1, t0, t1, T, μ::Function, σ::Function, g::Function)
#         dt = t1 - t0
#         # gt1 = (g(t1,T)-μ(t0,x0)*dt)/σ(t0,x0)
#         gt1 = g(t1,T)/σ(t0,x0)
#         gt0 = g(t0,T)
#         return 1
# end


@doc """
	transprob(x, y, p, h)

Returns the exact transition probability of an Ornstein Uhlenbeck process
x: starting position
y: ending position
p: mesh parameters
""" -> 
function transprob(x, y, p::MeshParams, μ::Function, σ::Function)
	s = sqrt(p.n)/y/sqrt(2*pi)*exp(-(log(y/x)+1/2/p.n)^2/2*p.n)*p.h
	if s > 1
		return 1 
	else 
		return s
	end
end

# n = 50
# test(z) = transprob(2/n, z, MeshParams(n, 1/n, 1), (t,x) -> 0, (t,x) -> 1)
# x = map(test, 3:(-1/n):(1/n))
# plot(3:(-1/n):(1/n),x)

# sum(x)


@doc """
	pmatrix0(p, μ, σ, g)

Returns the transition probability matrix of the Markov chain approximation of an Ornstein Uhlenbeck process
from time 0 to time 1/n
p: mesh parameters
""" -> 
function pmatrix0(p::MeshParams, μ::Function, σ::Function, g::Function)
	x0 = 1 # initial point`
	ymesh = (g(1/p.n,p.T)-p.h/2):(-p.h):(sqrt(p.n)*p.h)
	# ymesh = (g(1/p.n,p.T)-p.h/2):(-p.h):p.h
	yvec = zeros(length(ymesh))
		for j = 1:length(ymesh)
			yvec[j] = bbb(x0, ymesh[j], 0, 1/p.n, p.T, μ, σ, g)*transprob(x0, ymesh[j], p, μ, σ)
		end
	return yvec
end

# x = pmatrix0(MeshParams(50, 1/50, 1),  (t,x) -> 0, (t,x) -> 1, (t,T) -> exp(1))
# plot(x)

@doc """
	pmatrix

Returns the transition probability matrix of the Markov chain approximation of an Ornstein Uhlenbeck process
from time i/n to time (i+1)/n
i: ith time partition 
""" -> 
function pmatrix(i::Int, p::MeshParams, μ::Function, σ::Function, g::Function)
	jrange = (g(i/p.n,p.T)-p.h/2):(-p.h):(sqrt(p.n)*p.h)
	# jrange = (g(i/p.n,p.T)-p.h/2):(-p.h):p.h
	krange = (g((i+1)/p.n,p.T)-p.h/2):(-p.h):(sqrt(p.n)*p.h)
	# krange = (g((i+1)/p.n,p.T)-p.h/2):(-p.h):p.h
	if jrange[end] == 0
		jrange = jrange[1:(end-1)]
	end
	if krange[end] == 0
		krange = krange[1:(end-1)]
	end
	M = zeros(length(jrange),length(krange))
		for j = 1:length(jrange)
			for k = 1:length(krange)
				M[j, k] = bbb(jrange[j], krange[k], i/p.n, (i+1)/p.n, p.T, μ, σ, g)*transprob(jrange[j], krange[k], p, μ, σ)
			end
		end
	return M
end

# x1 = pmatrix(1, MeshParams(50, 1/50, 1),  (t,x) -> 0, (t,x) -> 1, (t,T) -> exp(1))
# x2 = pmatrix(40, MeshParams(50, 1/50, 1),  (t,x) -> 0, (t,x) -> 1, (t,T) -> exp(1))

# n = 50
# h = 1/n
# xmesh = (exp(1)-h/2):(-h):0 

# transprob(xmesh[end],xmesh[end], MeshParams(n, h, 1), (t,x) -> 0, (t,x) -> 1)

# xmesh[end]

#  transprob(0.3, z, MeshParams(30, 1/30, 1), (t,x) -> 0, (t,x) -> 1)

# transprob()

# plt[:imshow](log.(log.(x1 .+ 1) ))


@doc """
	BCP

Returns the approximated boundary crossing probability
p: mesh parameters
""" -> 
function BCP(p::MeshParams, μ::Function, σ::Function, g::Function)
    if g(p.T, p.T) < p.h
        return 1
    end
	prob = transpose(pmatrix0(p, μ, σ, g))
	for i = 1:(p.n-1)
		prob = prob*pmatrix(i, p, μ, σ, g)
		print("iteration ",i," sum ",sum(prob),"\n")
	end
	return 1 - (sum(prob))
end

BCP(MeshParams(50,1/50,1), 
		  (t,x) -> 0, 
		  (t,x) -> 1,
		  (t,T) -> exp(1)
		  )

@doc """
	guideplot(N, s)

Returns a plot with 4 convergence lines n^{-1/2}, n^{-1}, n^{-2}, n^{-4}
N: Maximum number of boundary partitions
s: constant scaling
""" -> 
function guideplot(N, s = 0.05)
	plt[:xscale]("log")
	plt[:yscale]("log")
	grid("on")
	plot(1:N, 1 ./((1:N).^0.5)*s, label = L"$1/\sqrt{n}$", ls = "--", color = "black")
	plot(1:N, 1 ./(1:N)*s ,	     label = L"1/n",     ls = "--", color = "black")
	plot(1:N, 1 ./((1:N).^2)*s,   label = L"1/n^2",   ls = "--", color = "black")
	plot(1:N, 1 ./((1:N).^4)*s,   label = L"1/n^4",   ls=  "--", color = "black")
end


@doc """
	converge(n, N)

Returns a convergence plot of the MC approximation towards the true solution
n: number of equally spaced points
N: maximum number of boundary partitions
""" -> 
function converge(n, N)
limit = exact_GBM()
n_mesh = unique(floor.(Int,exp.( log(N)*( 0:(1/(n-1)):1 ) ) ) )
bcp_vec = zeros(length(n_mesh))
for i in 1:length(n_mesh)
	@time bcp_vec[i] = abs(BCP(MeshParams(n_mesh[i],1/sqrt(n_mesh[i]),1), 
		  (t,x) -> 0, 
		  (t,x) -> 1,
		  (t,T) -> exp(1)
		  ) - limit)
	# print(bcp_vec[i], "\n")
end
figure()
guideplot(N, bcp_vec[1])
plot(n_mesh, bcp_vec, marker="o")
xlabel("n")
ylabel(L"|P_n - P|")
title(L"Error, $|P - P_n|$")
#return bcp_vec
end

converge(10,100)