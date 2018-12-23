using PyPlot
using Distributions

struct MeshParams
	n::Int64   
    h::Float64
    T::Float64
    lb::Float64
end

@doc """
	exact_OU(T)

Returns the exact probability that a Ornstein Uhlenbeck process crosses the boundary g(t) = α + h exp(-κ t)
""" -> 
function exact_OU(T = 1)
	h = 1
	κ = 1
	α = 0
	σ = 1
	return 2(1 - cdf(Normal(0,1), (α + h)/σ/sqrt((exp(2*κ*T)-1)/2κ))) 
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
        return 1 - exp(-2(gt1 - x1)*(gt0 - x0)/dt)
end

@doc """
	transprob(x, y, p, h)

Returns the exact transition probability of an Ornstein Uhlenbeck process
x: starting position
y: ending position
p: mesh parameters
""" -> 
function transprob(x, y, p::MeshParams, μ::Function, σ::Function)
	return pdf(Normal(x*exp(-1/p.n), sqrt((1-exp(-2/p.n))/2)), y)*p.h
end

@doc """
	C(x, lb_trunc, p, μ, σ)

Combines the transition probabilities of the lower tail
x: starting position
p: mesh parameters
""" -> 
function C(x, lb_trunc, p::MeshParams, μ::Function, σ::Function)
	ymesh = lb_trunc:(-p.h):(-10)
	l = length(ymesh)
	yvec = zeros(l)
	for j = 1:l
		yvec[j] = transprob(x, ymesh[j], p, μ, σ)
	end
	return sum(yvec)
end

@doc """
	pmatrix0(p, μ, σ, g)

Returns the transition probability matrix of the Markov chain approximation of an Ornstein Uhlenbeck process
from time 0 to time 1/n
p: mesh parameters
""" -> 
function pmatrix0(p::MeshParams, μ::Function, σ::Function, g::Function)
	ymesh = (g(1/p.n,p.T)-p.h/2):(-p.h):p.lb
	lb_trunc = ymesh[end]
	yvec = zeros(length(ymesh))
		for j = 1:(length(ymesh)-1)
			yvec[j] = bbb(0, ymesh[j], 0, 1/p.n, p.T, μ, σ, g)*transprob(0, ymesh[j], p, μ, σ)
		end
	yvec[end] = bbb(0, lb_trunc, 0, 1/p.n, p.T, μ, σ, g)*C(0, lb_trunc, p, μ, σ) 
	return yvec
end

@doc """
	pmatrix(i, p, h, T, lb)

Returns the transition probability matrix of the Markov chain approximation of an Ornstein Uhlenbeck process
from time i/n to time (i+1)/n
i: ith time partition 
""" -> 
function pmatrix(i::Int, p::MeshParams, μ::Function, σ::Function, g::Function)
	jrange = (g(i/p.n,p.T)-p.h/2):(-p.h):p.lb 
	krange = (g((i+1)/p.n,p.T)-p.h/2):(-p.h):p.lb
	lb_trunc = krange[length(krange)]
	M = zeros(length(jrange),length(krange))
		for j = 1:(length(jrange)-1)
			for k = 1:(length(krange)-1)
					M[j, k] = bbb(jrange[j], krange[k], i/p.n, (i+1)/p.n, p.T, μ, σ, g)*transprob(jrange[j], krange[k], p, μ, σ)
			end
			M[j, length(krange)] = bbb(jrange[j], lb_trunc, i/p.n, (i+1)/p.n, p.T, μ, σ, g)*C(jrange[j],lb_trunc,p,μ,σ)
		end
	M[length(jrange), length(krange)] = 1
	return M
end

@doc """
	BCP(n::Int, h, T, lb)

Returns the approximated boundary crossing probability
p: mesh parameters
""" -> 
function BCP(p::MeshParams, μ::Function, σ::Function, g::Function)
    if g(p.T, p.T) - p.lb < p.h
        return 1
    end
	prob = transpose(pmatrix0(p, μ, σ, g))
	for i = 1:(p.n-1)
		prob = prob*pmatrix(i, p, μ, σ, g)
	end
	return 1 - (sum(prob))
end


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
limit = exact_OU()
n_mesh = unique(floor.(Int,exp.( log(N)*( 0:(1/(n-1)):1 ) ) ) )
bcp_vec = zeros(length(n_mesh))
for i in 1:length(n_mesh)
	@time bcp_vec[i] = abs(BCP(MeshParams(n_mesh[i],1/n_mesh[i],1,-3), 
		  (t,x) -> 0, 
		  (t,x) -> 1,
		  (t,T) -> exp(-t) 
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
