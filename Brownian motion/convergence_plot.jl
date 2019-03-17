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
	plt[:xscale]("log")
	plt[:yscale]("log")
	grid("on", lw = 0.5)
	plot(1:N, 1 ./((1:N).^0.5)*s, label = L"$1/\sqrt{n}$", ls = "--", color = "black")
	plot(1:N, 1 ./(1:N)*s, label = L"1/n", ls = "--", color = "black")
	plot(1:N, 1 ./((1:N).^1.5)*s, label = L"1/n^1.5", ls = "--", color = "black")
	plot(1:N, 1 ./((1:N).^2)*s, label = L"1/n^2", ls = "--", color = "black")
	plot(1:N, 1 ./((1:N).^3.5)*s, label = L"1/n^3.5", ls=  "--", color = "black")
	plot(1:N, 1 ./((1:N).^4)*s, label = L"1/n^4", ls=  "--", color = "black")
	# plot(1:N, 1 ./((1:N).^6)*s, label = L"1/n^6", ls=  "--", color = "black")
	xticks(unique(vcat(1:10, 10:10:100)))
end


@doc """
	converge(n, N, p, T, x0, lb)

Returns a convergence plot of the MC approximation towards the true solution
n: number of equally spaced points
N: maximum number of boundary partitions

converge(10,100,1,1,0,-3,6)
""" -> 
function converge(n, N, p = 1, T = 1, x0 = 0, lb = -3, γ = 1, lb2 = -4, lb_trans = 10^6)
limit = exact_limit()
n_mesh = unique(floor.(Int,exp.( log(N)*( 0:(1/(n-1)):1 ) ) ) ) # cuts up the x -l og axis uniformly to save computation
bcp_vec = zeros(length(n_mesh))
for i in 1:length(n_mesh)
	if i > lb_trans
		lb = lb2
	end
	bcp_vec[i] = abs(BCP(n_mesh[i], γ/n_mesh[i]^p, T, x0, lb) - limit)
end
figure()
guideplot(N, bcp_vec[1])
plot(n_mesh, bcp_vec, marker="o")
xlabel("n")
ylabel(L"|P_n - P|")
title(L"Error, $|P - P_n|$")
end