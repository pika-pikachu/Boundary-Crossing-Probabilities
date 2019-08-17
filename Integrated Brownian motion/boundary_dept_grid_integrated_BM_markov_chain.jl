# Markov chain approximation of Integrated Brownian motion
# June 23rd 2019
#
# DESCRIPTION:
#
# Uses rectangular sum approximation of the integrated Brownian motion. The grid size is made dependent
# on the boundary. This ensures the boundary lies on the grid, and smooth convergence. 
#
# reference for animations:
# https://genkuroki.github.io/documents/Jupyter/20170624%20Examples%20of%20animations%20in%20Julia%20by%20PyPlot%20and%20matplotlib.animation.html
# animation save location: C:\Users\Vincent\AppData\Local\Julia-1.0.1
# 

using TensorOperations
using PyCall
@pyimport matplotlib.animation as anim
using PyPlot
using IJulia

function g(t)
	return 0.4
end

@doc """
	transprob(x, y, dt, h)

Returns the transition probability of Brownian motion
x: starting position
y: ending position
dt: time step size
h: space step size
""" -> 
function transprob(x, y, dt, h)
	return exp(-(y-x)^2/(2*dt))/sqrt(2*pi*dt)*h
end

@doc """
	pmatrix01(n, h, x_inf, y_inf)

Returns the transition probability matrix of the Markov chain approximation of Brownian motion 
from time 0 to time 1/n
n: number of time partitions
h: space step size
""" ->
function pmatrix01(n::Int, x_inf = 3, y_inf = 2)
h = g(1)/n
U_x = floor(Int, x_inf/h)
L_x = ceil(Int, -x_inf/h)
U_y = n^2
L_y = ceil(Int,  -y_inf*n/h)
x_1_mesh = U_x:-1:L_x
y_1_mesh = U_y:-1:L_y
M = zeros(length(x_1_mesh), length(y_1_mesh))
	for i = 1:length(x_1_mesh)
		for j = 1:length(y_1_mesh)
			# if (y_1_mesh[j] == x_1_mesh[i]) && (h/n*(U - (y_1_mesh[j] -1)) < g(1/n))
			if (y_1_mesh[j] == x_1_mesh[i]) && (h/n*y_1_mesh[j] < g(1/n))
				M[i, j] = transprob(0, h*x_1_mesh[i], 1/n, h)
			end
		end
	end
return M
end


@doc """
	pmatrix_i(n, h, x_inf, y_inf)

Transition from (x_i, y_i) to (x_{i+1} , y_{i+1}). 
n: number of time partitions
h: space step size
""" -> 
function pmatrix_i(t_i, n::Int, x_inf = 3, y_inf = 2)
h = g(1)/n
U_x = floor(Int, x_inf/h)
L_x = ceil(Int,-x_inf/h)
U_y = n^2
L_y = ceil(Int, -y_inf*n/h)
x_0_mesh = U_x:-1:L_x
y_0_mesh = U_y:-1:L_y
x_1_mesh = U_x:-1:L_x
y_1_mesh = U_y:-1:L_y
M = zeros(length(x_0_mesh),length(y_0_mesh),length(x_1_mesh),length(y_1_mesh))
	for i = 1:length(x_0_mesh)
		for j = 1:length(y_0_mesh)
			for k = 1:length(x_1_mesh)
				for l = 1:length(y_1_mesh)
					# if (y_1_mesh[l] - y_0_mesh[j] == x_1_mesh[k]) && (h/n*(U - (y_1_mesh[l]-1)) < g((t_i+1)/n)) && (h/n*(U - (y_0_mesh[j]-1)) < g(t_i/n))
					if (y_1_mesh[l] - y_0_mesh[j]) == x_1_mesh[k] && (h/n*y_1_mesh[l] < g(1/n))
						M[i,j,k,l] = transprob(h*x_0_mesh[i], h*x_1_mesh[k], 1/n, h)
					end
				end
			end
		end
	end
return M
end

function anti_transpose(M)
	return reshape(M[end:-1:1], size(M,1), size(M,2))
end

function t_mat(n::Int, x_inf = 3, y_inf = 2)
P_old = pmatrix01(n, x_inf, y_inf)
	for i in 1:(n-1)
		Mi = pmatrix_i(i, n, x_inf, y_inf)
		P_new = zeros(size(Mi)[1], size(Mi)[2])
		@tensor begin 
			P_new[x1,y1] = P_old[x0,y0]*Mi[x0,y0,x1,y1]
		end
		P_old = P_new
	end
return P_old
end

# 24.919170 seconds (3.67 k allocations: 26.157 GiB, 13.79% gc time)


# x = t_mat(10, 1/10)


function guideplot(N, s = 0.05)
	# plt.xscale("log")
	# plt.yscale("log")
	plt[:xscale]("log")
	plt[:yscale]("log")
	grid("on", lw = 0.5)
	pvec = [0.5, 1, 1.5, 2, 3, 4]
	for i in 1:length(pvec)
		plot(1:N, 1 ./( (1:N).^(pvec[i]) )*s, ls = "--", color = "black")
		plt[:text](N + 0.01, s/N^(pvec[i]) , join(["n^-", pvec[i]]), fontsize = 9)
	end	
	xticks(unique(vcat(1:10, 10:10:100, 100:100:1000, 1000:1000:10000)))
end


function convergence(N = 10, γ = 2, x_inf = 3, y_inf = 2)
p_vec = zeros(N)
for n in 1:N	
	M = t_mat(n, x_inf, y_inf)
	p_vec[n] = sum(M)
end
p_diff = abs.(p_vec[2:end] - p_vec[1:(end-1)])
subplot(1,2,1)
	guideplot(N, 0.1)
	plot(1:(N-1), p_diff)
subplot(1,2,2)
	plot(1:N, p_vec)
end




# figure()
# plt[:imshow](P_old, extent = [-3,3,-3,3], cmap = "afmhot", alpha=1, aspect="auto", interpolation="none")
# title(L"Heat map $ X_t = (W_t, \int_0^t W_u du) $")
# xlabel(L"$W_t$")
# ylabel(L"$\int_0^t W_u du$")