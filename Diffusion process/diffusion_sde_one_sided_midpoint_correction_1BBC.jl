using PyPlot
using Distributions

struct MeshParams
	n::Int64   
	h::Float64
    T::Float64
    lb::Float64
end

# function J(a1, b1, a2, b2, t1, t2, x, T)
# 	h = (a1 + b1*t1)/t1 - x/t2
# 	T = sqrt(t2*t1/(t2-t1))
# 	a2d = b2 + (a2 - x)/t2
# 	a1d = a1*(b1 + (a1 - x)/t2)
# 	J = 1 - cdf(Normal(),h*T) + exp(-2*a2*a2d)*
# 	    cdf(Normal(), (h - 2*a2d)*T ) +
# 	    exp(-2*a1d)* cdf(Normal(), h*T - 2*a1/T) -
# 	    exp(-2*a1d + (4*a1- 2*a2)*a2d )*
# 	    cdf(Normal(), (h - 2*a2d)*T - 2*a1/T )
# 	return J
# end
	
# function bbb(x0, x1, t0, t2, T, μ::Function, σ::Function, g::Function)
#         t1 = (t0 + t2)/2 
#         dt = t2 - t0
#         gt2 = (g(t2,T)-μ(t0,x0)*dt)/σ(t0,x0)
#         gt1 = (g(t1,T)-μ(t0,x0)*dt/2)/σ(t0,x0)
#         gt0 = g(t0,T)
#         b1 = (gt1 - gt0)/(t1 - t0) 
#         a1 = gt0 - x0
#         b2 = (gt2 - gt1)/(t2 - t1)
#         a2 = gt2 - b2*(t2 - t0) - x0
#         return 1 - J(a1, b1, a2, b2, t1 - t0, t2 - t0, x1 - x0, T)
# end

function bbb(x0, x1, t0, t1, T, μ::Function, σ::Function, g::Function)
        dt = t1 - t0
        # gt1 = (g(t1,T)-(μ(t0,x0)+μ(t0,x1))/2*dt)/σ(t0,x0)
        #gt1 = (g(t1,T)-μ(t0,x0)*dt)/σ(t0,x0)
        gt1 = g(t1,T)/σ(t0,x0)
        gt0 = g(t0,T)
        return 1 - exp(-2/dt*(gt1 - x1)*(gt0 - x0))
end

function transprob(x, y, t, p::MeshParams, μ::Function, σ::Function)
	return sqrt(p.n/2/pi)/σ(t,x)*exp(-p.n/2/(σ(t,x)^2)*(y - μ(t,x)/p.n - x)^2)*p.h
end

function C(x, lb_trunc, p::MeshParams, μ::Function, σ::Function)
	ymesh = lb_trunc:(-p.h):(-12)
	l = length(ymesh)
	yvec = zeros(l)
	for j = 1:l
		yvec[j] = transprob(x, ymesh[j], 1/p.n, p, μ, σ)
	end
	return sum(yvec)
end

function pmatrix0(p::MeshParams, μ::Function, σ::Function, g::Function)
	ymesh = (g(1/p.n,p.T)-p.h/2):(-p.h):p.lb
	l = length(ymesh)
	lb_trunc = ymesh[end]
	yvec = zeros(l)
		for j = 1:(l-1)
			yvec[j] = bbb(0, ymesh[j], 0, 1/p.n, p.T, μ, σ, g)*transprob(0, ymesh[j], 1/p.n, p, μ, σ)
		end
	yvec[end] = bbb(0, lb_trunc, 0, 1/p.n, p.T, μ, σ, g)*C(0, lb_trunc, p, μ, σ) 
	return yvec
end

function pmatrix(i::Int, p::MeshParams, μ::Function, σ::Function, g::Function)
	jrange = (g(i/p.n,p.T)-p.h/2):(-p.h):p.lb 
	krange = (g((i+1)/p.n,p.T)-p.h/2):(-p.h):p.lb
	lb_trunc = krange[length(krange)]
	M = zeros(length(jrange),length(krange))
		for j = 1:(length(jrange)-1)
			for k = 1:(length(krange)-1)
					M[j, k] = bbb(jrange[j], krange[k], i/p.n, (i+1)/p.n, p.T, μ, σ, g)*transprob(jrange[j], krange[k], i/p.n, p, μ, σ)
			end
			M[j, length(krange)] = bbb(jrange[j], lb_trunc, i/p.n, (i+1)/p.n, p.T, μ, σ, g)*C(jrange[j],lb_trunc,p,μ,σ)
		end
	M[length(jrange), length(krange)] = 1
	return M
end

function BCP(;n::Int64, T::Float64, lb::Float64, h_func::Function, μ::Function, σ::Function, g::Function)
	h = h_func(n)
	p = MeshParams(n, h, T, lb)
    if g(p.T, p.T) - p.lb < p.h
        return 1
    end    
	prob = transpose(pmatrix0(p, μ, σ, g))
	for i = 1:(p.n-1)
		prob = prob*pmatrix(i, p, μ, σ, g)
	end
	return 1 - (sum(prob))
end

#daniels boundary
@time BCP(n = 40,
		  T = 1.0,
		  lb = -3.0,
		  h_func = (n) -> 1/n,
		  μ = (t,x) -> 0, 
		  σ = (t,x) -> 1,
		  g = (t,T) -> 0.5 - t*log(0.25*(1 + sqrt( 1 + 8*exp(-1/t) ) ) )/sqrt(T) 
		  )

# OU process
a = @time BCP(n = 50,
		  T = 1.0,
		  lb = -3.0, 
		  h_func = (n) -> 1/n,
		  μ = (t,x) -> -x, 
		  σ = (t,x) -> 1,
		  g = (t,T) -> exp(-t) 
		  )

function exact_BM(T = 1)
	if T == 0 
	  return 0
	end
	a = 1/2
	k1 = k2 = 1/2
	psi(t, T) = a - t/(2a)*log(k1/2 +  sqrt( k1^2/4 + k2*exp(-4a^2/t) ) )/sqrt(T) 
	b1 = cdf(Normal(0 ,sqrt(T)), psi(T,1)) 
	b2 = k1*cdf(Normal(2a,sqrt(T)), psi(T,1)) 
	b3 = k2*cdf(Normal(4a,sqrt(T)),psi(T,1)) 
	return 1 - (b1 - (b2 + b3))
end

function exact_OU(T = 1)
	h = 1
	κ = 1
	α = 0
	σ = 1
	return 2(1 - cdf(Normal(0,1), (α + h)/σ/sqrt((exp(2*κ*T)-1)/2κ))) 
end

function guideplot(N, s = 0.05)
	plt[:xscale]("log")
	plt[:yscale]("log")
	grid("on", lw = 0.5)
	plot(1:N, 1 ./((1:N).^0.5)*s, label = L"$1/\sqrt{n}$", ls = "--", color = "black")
	plot(1:N, 1 ./(1:N)*s, label = L"1/n", ls = "--", color = "black")
	plot(1:N, 1 ./((1:N).^2)*s, label = L"1/n^2", ls = "--", color = "black")
	plot(1:N, 1 ./((1:N).^4)*s, label = L"1/n^4", ls=  "--", color = "black")
	xticks(unique(vcat(1:10, 10:10:100)))
end

function BM_converge(;n::Int64, N::Int64, h_func::Function)
limit = exact_BM()
n_mesh = unique(floor.(Int,exp.( log(N)*( 0:(1/(n-1)):1 ) ) ) )
bcp_vec = zeros(length(n_mesh))
for i in 1:length(n_mesh)
	@time bcp_vec[i] = 
	abs(BCP(
		n = n_mesh[i],
		T = 1.0, 
		lb = -3.0,
		h_func = h_func,
		μ = (t,x) -> 0, 
		σ = (t,x) -> 1,
		g = (t,T) -> 0.5 - t*log(0.25*(1 + sqrt( 1 + 8*exp(-1/t) ) ) )/sqrt(T)
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

#BM_converge(n = 10, N =50, h_func = (n) -> 1/sqrt(n))

function OU_converge(;n::Int64, N::Int64, h_func::Function)
limit = exact_OU()
n_mesh = unique(floor.(Int,exp.( log(N)*( 0:(1/(n-1)):1 ) ) ) )
bcp_vec = zeros(length(n_mesh))
for i in 1:length(n_mesh)
	@time bcp_vec[i] = 
	abs(BCP(
		n = n_mesh[i],
		T = 1.0, 
		lb = -3.0,
		h_func = h_func,
		μ = (t,x) -> -x, 
		σ = (t,x) -> 1,
		g = (t,T) -> exp(-t)
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

OU_converge(n = 8, N = 100, h_func = (n) -> 1/n^1.5)


