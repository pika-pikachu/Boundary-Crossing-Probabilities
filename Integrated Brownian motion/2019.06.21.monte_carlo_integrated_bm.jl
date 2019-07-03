# Integral of Brownian motion
# June 20th 2019

using PyPlot
using Statistics

@doc """
	monte_carlo_bm(n)

Returns a single simulation of a Brownian motion, its integral - both rectangular and midpoint
n: Number of time partitions
""" -> 
function monte_carlo_bm(n = 10^3)
x = cumsum([0; randn(n-1)])/sqrt(n) # Brownian motion
# y = cumsum(x)./(1:n) # Riemann sum rectangle rule
y = cumsum(x)./n # Riemann sum rectangle rule
z = y .- (x./(2*n))  # Riemann sum midpoint rule
return [x, y, z]
end

@doc """
	monte_carlo_sim(n, N)

Returns simulations of a Brownian motion and its integral
n: Number of time partitions
N: Number of Monte Carlo paths
""" -> 
function monte_carlo_sim(n = 10^3, N = 100)
figure(figsize=(12, 10))
time_mesh = range(0, stop = 1, length = n)
for i in 1:N
	z_vec = monte_carlo_bm(n)
	BM = z_vec[1]
	MAVG_BM = z_vec[2]
	subplot(2,3,1)
		plot(time_mesh, BM, color = "black",alpha = 0.05 )
		title(L"$W_t$")
		xlabel(L"$t$")
		axis([0,1,-4,4])
	subplot(2,3,2)
		plot(time_mesh, MAVG_BM, color = "red", alpha = 0.05)
		title(L"$\int_0^t W_s ds$")
		xlabel(L"$t$")
		axis([0,1,-4,4])
	subplot(2,3,4)
		plot(BM,MAVG_BM, color = "black", alpha = 0.05 )
		xlabel(L"$W_t$")
		ylabel(L"$\int_0^t W_s ds$")
		axis([-4,4,-3,3])
end
subplot(2,3,3)
	z_vec = monte_carlo_bm(n)
	BM = z_vec[1]
	MAVG_BM = z_vec[2]
	MAVG_mdpt_BM = z_vec[3]
	plot(time_mesh, BM, color = "black" , label = L"$W_t$")
	plot(time_mesh, MAVG_BM, color = "red", alpha = 1, label = L"$\int_0^t W_s ds$")
	plot(time_mesh, MAVG_mdpt_BM, color = "blue", alpha = 1, label = L"$\int_0^t W_s ds$ MDPT")
	xlabel(L"$t$")
	legend()
	axis([0,1,-4,4])
subplot(2,3,5)
	plot(BM,MAVG_BM, color = "red", linewidth = "1" )
	xlabel(L"$W_t$")
	ylabel(L"$\int_0^t W_s ds$")
	for i in [1, floor(Int,n/4),floor(Int,n/2),floor(Int,3*n/4), n] 
		plt[:scatter](BM[i], MAVG_BM[i], color = "black")
		plt[:text](BM[i], MAVG_BM[i], join(["t=",i/n]), fontsize = 9)
	end
	axis([-4,4,-3,3])
end


# Boundary function
function gU(t)
	return 0.4
end


function gL(t)
	return -0.4
end

@doc """
	monte_carlo_crossing(n)

Simulates a trajectory to see if it hits a boundary
n: Number of time partitions
""" -> 
function monte_carlo_crossing(n = 10^3)
x = cumsum([0; randn(n-1)])/sqrt(n) # Brownian motion
Y_old = 0
	for i in 2:n 
		Y_new = Y_old + x[i]/n # rectangular rule
		if Y_new > gU(i/n) || Y_new < gL(i/n)
			return 1
		end
		Y_old = Y_new
	end
	# for i in 2:n 
	# 	Y_new = Y_old + (x[i]+x[i-1])/(2*n) # Trapezoidal rule
	# 	if Y_new > gU(i/n) || Y_new < gL(i/n)
	# 		return 1
	# 	end
	# 	Y_old = Y_new
	# end
return 0
end


@doc """
	monte_carlo_bcp(n)

Returns the survival probability
n: Number of time partitions
""" -> 
function monte_carlo_bcp(N = 10^3, n = 10^3)
mc_vec = zeros(N)
	for i in 1:N
		mc_vec[i] = monte_carlo_crossing(n)
	end
return 1 - mean(mc_vec)
end


# monte_carlo_bcp(10^4,10^3)

using PyPlot
function guideplot(N, s = 0.05)
	# plt.xscale("log")
	# plt.yscale("log")
	plt[:xscale]("log")
	plt[:yscale]("log")
	grid("on", lw = 0.5)
	pvec = [0.5, 1, 1.5, 2, 3, 4]
	for i in 1:length(pvec)
		plot(1:N, 1 ./( (1:N).^(pvec[i]) )*s, ls = "--", color = "black")
		plt[:text](N + 0.3, s/N^(pvec[i]) , join(["n^-", pvec[i]]), fontsize = 9)
	end	
	xticks(unique(vcat(1:10, 10:10:100, 100:100:1000, 1000:1000:10000)))
end


function dyn_conv_plot(n)
i = 1
n_mesh = 1:n 
bcp_vec = zeros(n)
for j in 1:n
	bcp_vec[j] = monte_carlo_crossing(n_mesh[j])
end
while 1 == 1 # non ending loop
	for j in 1:n
		bcp_vec[j] = bcp_vec[j]*i/(i+1) +  monte_carlo_crossing(n_mesh[j])/(i+1)
	end
	if i % 10000 == 0
		# fitting linear model
		dPn = abs.( bcp_vec[2:end] - bcp_vec[1:(end-1)] )
		dP_n_x_mesh = 2:n
		x = log.(dP_n_x_mesh)
		y = log.(dPn)

		beta_1 = cov(x,y)/var(x)
		# beta_0 = mean(y) - beta_1*mean(x)
		beta_0 = y[1] - beta_1*x[1]

		y_fit = exp.( beta_0*ones(n-1) + beta_1*log.(2:n) )

		plt[:cla]()
		guideplot(n, 0.1)
		plot(2:n, dPn, marker = "o")	
		plot(dP_n_x_mesh, y_fit)	
		plt[:pause](0.0001)
		print("iteration: ", i, " slope: ", beta_1 , " BCP: ", bcp_vec[end], "\r")
	end
	i += 1
end
end

dyn_conv_plot(80)