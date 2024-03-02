### A Pluto.jl notebook ###
# v0.19.37

using Markdown
using InteractiveUtils

# ╔═╡ c2c8340e-6dcc-485d-9080-067afce01d98
using Pkg

# ╔═╡ 7c966f54-5f54-4e54-b2b7-9a176bc18425
Pkg.activate(".")

# ╔═╡ 53db1055-1f08-44ae-b408-c913ca584796
using Dagitty, CSV, DataFrames, StanSample, StatsPlots, StatsBase, Distributions, RCall, Random

# ╔═╡ e8876f09-f2a4-4d82-a801-6ce4967c8118
using CategoricalArrays

# ╔═╡ b35464ca-d191-11ee-0596-9d83104cc8e8
md"""
Author: Curro Campuzano Jiménez (@currocam)
"""

# ╔═╡ 19415027-ccb4-4dbd-947a-b451cb8a9589
md"""
# Week 6
The theme of this homework is tadpoles. You must keep them alive.
"""

# ╔═╡ aba71a6a-cc92-40ac-bd08-be657e4da066
function reedfrogs()
  url = "https://raw.githubusercontent.com/rmcelreath/rethinking/master/data/reedfrogs.csv"
  CSV.read(download(url), DataFrame)
end

# ╔═╡ f1dca430-5ecd-433b-8810-569b49f9eac1
data = reedfrogs()

# ╔═╡ c35ab845-e480-45f6-891d-7c10456ec122
md"""
## Exercise 1
Conduct a prior predictive simulation for the Reedfrog model. By this I mean to simulate the prior distribution of tank survival probabilities αj. Start by using this prior:

$$
\begin{aligned}
S_j \sim \text{Binomial}(N_j, p_j)\\
\text{logit}(p_j) = \alpha_j\\
\alpha_j \sim N(\bar \alpha, \sigma)\\
\bar \alpha \sim N(0, 1)\\
\sigma \sim \exp(1)
\end{aligned}$$
Be sure to transform the αj values to the probability scale for plotting and summary.
How does increasing the width of the prior on σ change the prior distribution of αj?
You might try Exponential(10) and Exponential(0.1) for example.
"""

# ╔═╡ fd342621-481c-4344-b34e-21948b94d396
inv_logit = x -> 1 / (1+ exp(-x))

# ╔═╡ 7d9638a0-5f5a-4438-957f-9f93854b2157
function prior_prediction(n, rate)
	σ = rand(Exponential(rate), n)
	α_bar = rand(Normal(0, 1), n)
	α_j = rand.(Normal.(α_bar, σ))
	inv_logit.(α_j)
end

# ╔═╡ 5bc935c2-0ed9-4366-bd05-20ce2b790ba2
histogram(
	[
		prior_prediction(1000, 10),
		prior_prediction(1000, 1),
		prior_prediction(1000, 0.1)
	],
	xaxis = "Survival probability", 
	yaxis = "Density",
	xlim = (0, 1),
	labels = ["σ = exp(10)" "σ = exp(1)" "σ = exp(0.1)"]
)

# ╔═╡ dc8e5154-e64d-43e3-9e06-748e6521a2fe
md"""
As we can see, very flat prior induces very flat priors *before* applying the link function. After applying the link (logit), most of that flatness goes into 0 or 1, making it highly skewed. 
"""

# ╔═╡ c1329bc6-0884-4bc7-b2fe-e624adea9f43
md"""
## Exercise 2
Revisit the Reedfrog survival data, data(reedfrogs). Start with the varying
effects model from the book and lecture.
"""

# ╔═╡ b8046f68-eb8f-4492-a7b9-5c4559ce2013
stan_m1 = "
data {
  int<lower=0> N;
  array[N] int<lower=1> density;
  array[N] int<lower=1> surv;
}
parameters {
  array[N] real alpha;
  real alpha_bar;
  real sigma;
}
model {
	sigma ~ exponential(1);
	alpha_bar ~ normal(0, 1);
	alpha ~ normal(alpha_bar, sigma);	
	surv ~ binomial(density, inv_logit(alpha));
}
";

# ╔═╡ abcd20ac-5a54-4851-a972-980552df49ae
function m1(data)
  model = SampleModel("m1", stan_m1)
  data_dict = Dict(
    "N" => size(data, 1),
    "density" => data.density,
    "surv" => data.surv,
  )
  rc = stan_sample(model; data=data_dict)
  if success(rc)
    return model
  end
  nothing
end

# ╔═╡ 6b790ce4-b240-4190-a971-29bd01703b82
# ╠═╡ show_logs = false
post1 = read_samples(m1(data))

# ╔═╡ d053f1fb-4317-42d9-a045-edfe6b533d37
let
	scatter(
	data.propsurv, 
	inv_logit.([mean(post1[i]) for i in 1:nrow(data)]),
	labels = false,
	yaxis = "Posterior mean survival probability",
	xaxis = "Empirical survival probability"
	)
	Plots.abline!(1, 0, line=:dash, legend=false)
end

# ╔═╡ 91d5ba58-6e40-4cd8-a835-7aa304335c63
md"""
Then modify it to estimate the causal effects of the treatment variables pred and size, including how size might modify the effect of predation. An easy approach is to estimate an effect for each combination of pred and size. Justify your model with a DAG of this experiment.
"""

# ╔═╡ 1928a266-dd85-4975-aa15-9f724b8abc3a
drawdag(DAG(:Predation => :Survival, :Tank => :Survival, :Size => :Survival, :Density => :Survival))

# ╔═╡ 4593c1c5-5257-4a1d-8fe9-0adad1349534
stan_m2 = "
data {
  int<lower=0> N;
  int<lower=0> K;
  array[N] int<lower=1> density;
  array[N] int<lower=1> surv;
  array[N] int<lower=1> category;
}
parameters {
  array[N] real alpha;
  array[K] real beta;
  real sigma;
}
model {
	sigma ~ exponential(1);
	alpha ~ normal(0, sigma);	
	beta ~ normal(0, 1);
	for (n in 1:N) {
		surv[n] ~ binomial(density[n], inv_logit(alpha[n] + beta[category[n]]));
	}
}
";

# ╔═╡ 63cb8621-b8ba-413b-9549-9d7e39d66ac7
function m2(data)
  model = SampleModel("m2", stan_m2)
  categories = CategoricalArray(data.pred .* data.size)
  data_dict = Dict(
    "N" => size(data, 1),
	"K" => length(levels(categories)),
    "density" => data.density,
    "surv" => data.surv,
	"category" => levelcode.(categories)
  )
  rc = stan_sample(model; data=data_dict)
  if success(rc)
    return model
  end
  nothing
end

# ╔═╡ 07dd01e6-06e2-4730-ba40-877eab0cedf4
# ╠═╡ show_logs = false
post2 = read_samples(m2(data))

# ╔═╡ f1b08c71-9700-4c79-a57d-4f3ca24954e4
md"""
Now, we can explore the four $\beta$ coefficients:
"""

# ╔═╡ 1230cad0-cb47-41e2-beb2-8b613c500b0a
histogram(
	[[post2[key] for key in [Symbol("beta.$i") for i in 1:4]]],
	xaxis = "", 
	yaxis = "Density",
	layout = 4,
	color=[:gray :red :blue :green],
	label = false,
	title = [
		"No predator & big effect" "No predator & small effect" "Predator & big effect" "Predator & small effect"
	]
	)

# ╔═╡ b7dd6c46-2419-465f-b3e0-16bbecad8bf4
md"""
As expected, the absence of predator has an strong and clear positive effect in survival (top row of plots). Interestengly,the effect of predation is larger in large tapoles. 
"""

# ╔═╡ Cell order:
# ╟─b35464ca-d191-11ee-0596-9d83104cc8e8
# ╠═c2c8340e-6dcc-485d-9080-067afce01d98
# ╠═7c966f54-5f54-4e54-b2b7-9a176bc18425
# ╟─19415027-ccb4-4dbd-947a-b451cb8a9589
# ╠═53db1055-1f08-44ae-b408-c913ca584796
# ╠═aba71a6a-cc92-40ac-bd08-be657e4da066
# ╠═f1dca430-5ecd-433b-8810-569b49f9eac1
# ╟─c35ab845-e480-45f6-891d-7c10456ec122
# ╠═fd342621-481c-4344-b34e-21948b94d396
# ╠═7d9638a0-5f5a-4438-957f-9f93854b2157
# ╠═5bc935c2-0ed9-4366-bd05-20ce2b790ba2
# ╟─dc8e5154-e64d-43e3-9e06-748e6521a2fe
# ╟─c1329bc6-0884-4bc7-b2fe-e624adea9f43
# ╠═b8046f68-eb8f-4492-a7b9-5c4559ce2013
# ╠═abcd20ac-5a54-4851-a972-980552df49ae
# ╠═6b790ce4-b240-4190-a971-29bd01703b82
# ╠═d053f1fb-4317-42d9-a045-edfe6b533d37
# ╟─91d5ba58-6e40-4cd8-a835-7aa304335c63
# ╠═1928a266-dd85-4975-aa15-9f724b8abc3a
# ╠═e8876f09-f2a4-4d82-a801-6ce4967c8118
# ╠═4593c1c5-5257-4a1d-8fe9-0adad1349534
# ╠═63cb8621-b8ba-413b-9549-9d7e39d66ac7
# ╠═07dd01e6-06e2-4730-ba40-877eab0cedf4
# ╟─f1b08c71-9700-4c79-a57d-4f3ca24954e4
# ╠═1230cad0-cb47-41e2-beb2-8b613c500b0a
# ╟─b7dd6c46-2419-465f-b3e0-16bbecad8bf4
