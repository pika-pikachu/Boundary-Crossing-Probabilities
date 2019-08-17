using PyPlot
# using PyCall
# @pyimport matplotlib.patches as patch
using CPUTime


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
	converge(n, N, p, T, x0, lb, γ)

Returns a convergence plot of the MC approximation towards the true solution
n: number of equally spaced points
N: maximum number of boundary partitions

converge(10,100,1,1,0,-3,1)
""" -> 
function converge(n, N, p = 1, T = 1, x0 = 0, lb = -3, γ = 1)
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
# guideplot(N, bcp_vec[1])
guideplot(N, N^(2*p)*bcp_vec[end])
plot(n_mesh, bcp_vec, marker="o")

# time elapsed plotting
	for i in 1:length(n_mesh)
		plt[:text](n_mesh[i] + 0.3, bcp_vec[i], join([time_vec[i],"s"]), fontsize = 9)
	end
xlabel(L"n")
ylabel(L"|P_n - P|")
title("Convergence Plot")
end


