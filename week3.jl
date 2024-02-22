### A Pluto.jl notebook ###
# v0.19.37

using Markdown
using InteractiveUtils

# ╔═╡ 2bf77562-7520-4319-ac0a-1d91e1a548fa
begin
  using Pkg
  Pkg.activate(".")
  using CSV
  using DataFrames
  using Dagitty
  using StanSample
  using StatsBase
  using StatsPlots
  using Distributions
end

# ╔═╡ 3ea44b70-bf7d-11ee-02db-3144b88214ab
md"""
#### Author: Curro Campuzano Jiménez (@currocam)
"""

# ╔═╡ 36a34cfb-1769-48e4-8507-863deac4c6d9
md"""
# Week 3
## Question 1

The first two problems are based on the same data. The data in data(foxes) are 116 foxes from 30 different urban groups in England. These fox groups are like street gangs. Group size (groupsize) varies from 2 to 8 individuals. Each group maintains its own (almost exclusive) urban territory. Some territories are larger than others. The area variable encodes this information. Some territories also have more avgfood than others. And food influences the weight of each fox.
"""

# ╔═╡ 39feeff7-8040-418c-a0ac-87b04e88c8b1
function Foxes()
  url = "https://raw.githubusercontent.com/rmcelreath/rethinking/master/data/foxes.csv"
  CSV.read(download(url), DataFrame)
end

# ╔═╡ 909e1c12-3ee1-4de4-ae71-31d7d0e9221a
foxes = Foxes()

# ╔═╡ e6bd5d00-bef8-4a8f-a038-bf8ffae60b78
md"""
Assume this DAG:
"""

# ╔═╡ cc982a03-ecca-46e6-a44e-c14b7971255f
DAG(:A => :F, :F => :G, :F => :W, :G => :W) |> drawdag

# ╔═╡ 08e863da-fa5c-49e0-b58d-f2d88f3f129c
md"""
where F is avgfood, G is groupsize, A is area, and W is weight.
Use the backdoor criterion and estimate the total causal influence of A on
F. *What effect would increasing the area of a territory have on the amount
of food inside it?*
"""

# ╔═╡ 6a0c9a7d-c4c5-4ee3-9a3c-351b87a27060
function model1(data)
	stan_model = "
data {
  int<lower=0> N;
  vector[N] area;
  vector[N] avgfood;
}
parameters {
  real beta;
  real sigma;
}
model {
  sigma ~ exponential(1);
  beta ~ normal(0, 1);
  avgfood ~ normal(beta * area, sigma);
}
";
	model = SampleModel("regression", stan_model)
  data_dict = Dict(
    "N" => size(data, 1),
    "area" => standardize(ZScoreTransform, data.area),
    "avgfood" => standardize(ZScoreTransform, data.avgfood)
  )
  rc = stan_sample(model; data=data_dict)
  if success(rc)
    return model
  end
  nothing
end

# ╔═╡ 00c7f72c-d43f-410c-b3cf-bc8bd16ef32c
# ╠═╡ show_logs = false
model1(foxes) |> describe

# ╔═╡ 419cdddc-979c-4872-a62c-ce81a12106b5
md"""
Although is not always the case, here the coefficient $beta$ describes the total causal influence of the area on the available food. An increase of 1sd on the area will roughly increase 0.88sd the average available food. 
"""

# ╔═╡ 9287c935-39c5-4d2f-bd89-8a7e85f9ea03
md"""
## Question 2
Infer the total causal effect of adding food F to a territory on the weight
W of foxes. Can you calculate the causal effect by simulating an intervention
on food?
"""

# ╔═╡ b8127e1f-915d-45cf-9704-32c336f91334
function model2(data)
	stan_model = "
data {
  int<lower=0> N;
  vector[N] avgfood;
  vector[N] weight;
}
parameters {
  real alpha;
  real beta;
  real sigma;
}
model {
  sigma ~ exponential(1);
  alpha ~ normal(0, 1);
  beta ~ normal(0, 1);
  weight ~ normal(alpha + beta * avgfood, sigma);
}
";
	model = SampleModel("regression", stan_model)
  data_dict = Dict(
    "N" => size(data, 1),
    "weight" => standardize(ZScoreTransform, data.weight),
    "avgfood" => standardize(ZScoreTransform, data.avgfood)
  )
  rc = stan_sample(model; data=data_dict)
  if success(rc)
    return model
  end
  nothing
end

# ╔═╡ 478b8a4d-92b1-4126-a92e-841be702291c
# ╠═╡ show_logs = false
m2 = model2(foxes);

# ╔═╡ e0cfe40f-3a0a-4d5b-a39b-e728247af49f
begin
	post2 = read_samples(m2)
	weight0 = Normal.(post2.alpha, post2.sigma) .|> rand
	weight1 = Normal.(post2.alpha + post2.beta, post2.sigma) .|> rand
	density(weight1- weight0)
	xaxis!("Effect of 1sd increase in F on weight")
	yaxis!("Density")
end

# ╔═╡ 4924930e-f0fe-4375-84d2-f7646f046206
md"""
Our results suggest that there is no increase on weight when increasing the available food. 

# Question 3
Infer the direct causal effect of adding food F to a territory on the weight
W of foxes. In light of your estimates from this problem and the previous
one, what do you think is going on with these foxes?
"""

# ╔═╡ a4fabef0-8dcb-4244-a0ca-92d134064760
function model3(data)
	stan_model = "
data {
  int<lower=0> N;
  vector[N] avgfood;
  vector[N] groupsize;
  vector[N] weight;
}
parameters {
  real alpha;
  real beta_g;
  real beta_f;
  real sigma;
}
model {
  sigma ~ exponential(1);
  alpha ~ normal(0, 1);
  beta_g ~ normal(0, 1);
  beta_f ~ normal(0, 1);
  weight ~ normal(alpha + beta_f * avgfood + beta_g * groupsize, sigma);
}
";
	model = SampleModel("regression", stan_model)
  data_dict = Dict(
    "N" => size(data, 1),
    "weight" => standardize(ZScoreTransform, data.weight),
    "avgfood" => standardize(ZScoreTransform, data.avgfood),
    "groupsize" => standardize(ZScoreTransform, float(data.groupsize))
  )
  rc = stan_sample(model; data=data_dict)
  if success(rc)
    return model
  end
  nothing
end

# ╔═╡ 061a957c-3e73-4089-8583-f8fbb80adbb8
# ╠═╡ show_logs = false
m3 = model3(foxes);

# ╔═╡ 2d2caa4b-fceb-4ceb-a9b2-839078cf4217
weight1_groupsize
	post3 = read_samples(m3)
	weight0_direct = Normal.(post3.alpha, post3.sigma) .|> rand
	weight1_groupsize = Normal.(post3.alpha + post3.beta_f, post3.sigma) .|> rand
	density(weight1_direct- weight0_direct)
	xaxis!("Effect of 1sd increase in F on weight")
	yaxis!("Density")
end

# ╔═╡ 2315bb9f-b6ad-475a-962e-67cc9001983e
begin
	weight0_groupsize = Normal.(post3.alpha, post3.sigma) .|> rand
	weight1_groupsize = Normal.(post3.alpha + post3.beta_g, post3.sigma) .|> rand
	density(weight1_groupsize- weight0_groupsize)
	xaxis!("Effect of 1sd increase in group size on weight")
	yaxis!("Density")
end

# ╔═╡ a2226c30-ceb0-416d-b800-c3558509c027
describe(m3)

# ╔═╡ 5a58d03d-b241-4407-9b8e-3be468d6c632
md"""
Although there is a lot of uncertainty, what we see is that the direct causal effect of adding food (once fixed the group size) is likely positive, and 1sd increase in the food would cause roughly between 0.26-0.90 sd increase in weight. In contrast, the direct effect of the groupsize on the weight is negative. 1sd increase in the group size would cause a decrease of 0.36-1.00 sd in the weight. Basically, both forces cancel each other. 

## Question 4
Suppose there is an unobserved confound that influences F
and G, like this:
"""

# ╔═╡ 71c239f5-41cf-418b-815b-a46e9eb9e3dc
DAG(:A => :F, :F => :G, :F => :W, :G => :W, :U => :F, :U => :G) |> drawdag

# ╔═╡ 6493fd3c-a15a-4f13-8ebc-397053f22c72
md"""
Assuming the DAG above is correct, again estimate both the total and direct
causal effects of F on W. What impact does the unobserved confound have?

Because of the backdoor criterion, the total causal effect will be biased, but not the direct effect (because we intervene in both F and G). Then, the previous direct effect of F in W is okay. 
"""

# ╔═╡ Cell order:
# ╠═3ea44b70-bf7d-11ee-02db-3144b88214ab
# ╠═2bf77562-7520-4319-ac0a-1d91e1a548fa
# ╠═36a34cfb-1769-48e4-8507-863deac4c6d9
# ╠═39feeff7-8040-418c-a0ac-87b04e88c8b1
# ╠═909e1c12-3ee1-4de4-ae71-31d7d0e9221a
# ╠═e6bd5d00-bef8-4a8f-a038-bf8ffae60b78
# ╠═cc982a03-ecca-46e6-a44e-c14b7971255f
# ╠═08e863da-fa5c-49e0-b58d-f2d88f3f129c
# ╠═6a0c9a7d-c4c5-4ee3-9a3c-351b87a27060
# ╠═00c7f72c-d43f-410c-b3cf-bc8bd16ef32c
# ╠═419cdddc-979c-4872-a62c-ce81a12106b5
# ╠═9287c935-39c5-4d2f-bd89-8a7e85f9ea03
# ╠═b8127e1f-915d-45cf-9704-32c336f91334
# ╠═478b8a4d-92b1-4126-a92e-841be702291c
# ╠═e0cfe40f-3a0a-4d5b-a39b-e728247af49f
# ╠═4924930e-f0fe-4375-84d2-f7646f046206
# ╠═a4fabef0-8dcb-4244-a0ca-92d134064760
# ╠═061a957c-3e73-4089-8583-f8fbb80adbb8
# ╠═2d2caa4b-fceb-4ceb-a9b2-839078cf4217
# ╠═2315bb9f-b6ad-475a-962e-67cc9001983e
# ╠═a2226c30-ceb0-416d-b800-c3558509c027
# ╠═5a58d03d-b241-4407-9b8e-3be468d6c632
# ╠═71c239f5-41cf-418b-815b-a46e9eb9e3dc
# ╠═6493fd3c-a15a-4f13-8ebc-397053f22c72
