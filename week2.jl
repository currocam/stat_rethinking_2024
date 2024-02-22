### A Pluto.jl notebook ###
# v0.19.37

using Markdown
using InteractiveUtils

# ╔═╡ e9b1ec8c-b6cb-11ee-0969-579d19504652
# ╠═╡ show_logs = false
begin
  using Pkg
  Pkg.activate(".")
  using Plots
  using CSV
  using DataFrames
  using Distributions
  using Dagitty
  using StanSample
  using StatsPlots
end

# ╔═╡ 1aff8e9b-209a-47b5-acd2-c2e23581f7b5
md"""
#### Author: Curro Campuzano Jiménez (@currocam)
"""

# ╔═╡ 489f45c6-974d-4c39-90ee-be218e40dc0a
md"""
# Week2

## Question 1
From the Howell1 dataset, consider only the people younger than 13 years
old. Estimate the causal association between age and weight. Assume that
age influences weight through two paths. First, age influences height, and
height influences weight. Second, age directly influences weight through age related changes in muscle growth and body proportions.

Draw the DAG that represents these causal relationships.
"""

# ╔═╡ e29d7d00-6c3e-4662-8707-f40e92d05005
g = DAG(:Age => :Height, :Height => :Weight, :Age => :Weight)

# ╔═╡ ce77b783-8c75-440e-b53f-d25c7ef757ee
drawdag(g)

# ╔═╡ 9ab20f92-ae18-42be-a07b-b0d835c74b31
md"""
And then write a generative simulation that takes age as an input and simulates height and weight, obeying the relationships in the DAG.
"""

# ╔═╡ 5e837e37-2a01-45ef-809b-3193e2ddb95d
truncated_normal(μ, σ) = truncated(Normal(μ, σ), lower=0)

# ╔═╡ fad0aa05-67d2-4004-aef1-fe37d56c8f43
function simulation(age::Int)
  α = 10
  β = 0.1
  γ = 2
  σ = 10
  σ2 = 2
  H = truncated_normal(α * age, σ) |> rand
  W = truncated_normal(γ * age + β * H, σ2) |> rand
  return [H, W]
end

# ╔═╡ 331981a6-9411-478f-b649-365caeb17c24
begin
  ages_sim = rand(DiscreteUniform(2, 13), 200)
  sim_data = DataFrame(hcat(simulation.(ages_sim)...)', [:height, :weight])
  sim_data.age = ages_sim
  sim_data
end

# ╔═╡ 4b6caf18-014f-446b-aba9-eeb3ba39c3d8
md"""
Let's visualize the data. 
"""

# ╔═╡ 7e3d943f-fabf-46b2-a79f-8fcf97ea16d7
begin
  scatter(sim_data.age, sim_data.weight, label="Simulated data", legend=:topleft)
  xlabel!("Age in years")
  ylabel!("Weight (Kg)")
end

# ╔═╡ 60d2e0a9-76c6-474b-bca5-c5c914bc4645
md"""
## Question 2

Estimate the total causal effect of each year of growth on weight. First, we are going to define the model using stan. 
"""

# ╔═╡ e78c4c12-dfb5-4df5-90c9-fa274d114f8b
stan_model = "
data {
  int<lower=0> N;
  vector[N] Age;
  vector[N] W;
}
parameters {
  real alpha;
  real beta;
  real sigma;
}
model {
  alpha ~ normal(5, 1);
  beta ~ uniform(0, 10);
  sigma ~ exponential(1);
  W ~ normal(alpha + beta * Age, sigma);
}
";

# ╔═╡ 3dbd1d0a-975a-40ec-8ef6-5ed57a9976b1
function fit(data)
  model = SampleModel("regression", stan_model)
  data_dict = Dict(
    "N" => size(data, 1),
    "Age" => data.age,
    "W" => data.weight
  )
  rc = stan_sample(model; data=data_dict)
  if success(rc)
    return model
  end
  nothing
end

# ╔═╡ a5e311d4-278a-4258-ba58-cdebb29e34b1
md"""
### Simulated data
Before using real data, let's do it with the simulated data
"""

# ╔═╡ dda1a083-83a8-40de-9703-c1eedaf48ff9
# ╠═╡ show_logs = false
sim_data |> fit |> describe

# ╔═╡ 213e48a5-e8f9-4223-8906-d4e5dcc37d2d
md"""
The model seems to be working alright, although $\alpha$ doesn't look alright to me. We can estimate the total causal effect using the $\beta$ paramater. 

### Real data
We download the data
"""

# ╔═╡ b5a71427-e6ad-4405-9046-0aa068607705
function Howell()
  url = "https://raw.githubusercontent.com/rmcelreath/rethinking/master/data/Howell1.csv"
  CSV.read(download(url), DataFrame)
end

# ╔═╡ d937f3cf-d6d7-404f-9f1d-ee078a2f0960
howell = filter(:age => x -> x < 13, Howell())

# ╔═╡ ac938fc3-c942-4e6f-b5ca-902dbb5944f2
md"""
And now, we fit the model. 
"""

# ╔═╡ 9b434802-e443-469d-8c15-f2a128c6c1c9
# ╠═╡ show_logs = false
howell_model = fit(howell);

# ╔═╡ 885c4d29-8898-4708-ae18-fd95a9798404
md"""
We can expect the parameters. The $\beta$ corresponds to the total causal effect age in weight.
"""

# ╔═╡ f815c70f-c2cc-4146-8965-89ff677d6771
describe(howell_model)

# ╔═╡ d995233a-e5c1-4f02-819f-387a0960c37a
md"""
We can do it better by visualizing a few lines from the posterior with the data. 
"""

# ╔═╡ a6e40368-9d58-494d-8de3-3891f3da6e82
post = read_samples(howell_model, :dataframe);

# ╔═╡ 23602a27-a721-40cf-bd4d-58ffa17eabde
begin
  scatter(howell.age, howell.weight, label="Data", legend=:topleft)
  for i in 1:20
    plot!(howell.age, post.alpha[i] .+ post.beta[i] .* howell.age, legend=nothing)
  end
  xlabel!("Age in years")
  ylabel!("Weight (Kg)")
end

# ╔═╡ 4003e2df-cbb8-4c91-bc7b-bc15bb7a952b
md"""
All of them are very close together and fit the data well. Another option would beto visualize the distribution of the $\beta$ parameter. 
"""

# ╔═╡ f99adb4b-acac-4124-971b-9c11acf6b1bd
density(post.beta, label ="Posterior distribution of beta")

# ╔═╡ 50d468c9-ed40-4299-980b-5c13c043c189
md"""
## Question 3

The data in data(Oxboys) (rethinking package) are growth records for 26 boys measured over 9 periods. I want you to model their growth. Specifically, model the increments in growth from one period (Occasion in the data table) to the next. Each increment is
simply the difference between height in one occasion and height in the previous occasion. Since none of these boys shrunk during the study, all of the growth increments are greater than zero. Estimate the posterior distribution of these increments. Constrain the distribution so it is always positive—it should not be possible for the model to think that boys can shrink from year to year. Finally compute the posterior distribution of the total growth over all 9 occasions.
"""

# ╔═╡ b9f8e03b-210c-4484-bc99-ed9543905bb2
function Oxboys()
  url = "https://raw.githubusercontent.com/rmcelreath/rethinking/master/data/Oxboys.csv"
  CSV.read(download(url), DataFrame)
end

# ╔═╡ 843e71f9-d6b3-45cf-9f27-213d4fab5683
oxboys_raw = Oxboys()

# ╔═╡ 4626a594-6897-4060-b7f7-64b3f5c2f56d
md"""
First, we compute the difference in height and age.
"""

# ╔═╡ f98a8877-7755-4019-a5f1-06e527d3fd01
differences(vct) = [vct[i]-vct[i-1] for i in 2:length(vct)]

# ╔═╡ 9bb5ddbc-f892-425e-bea2-886ec60b92df
oxboys = combine(
	groupby(oxboys_raw, :Subject),
	:height => differences => :ΔHeight,
)

# ╔═╡ 4f6e513d-65eb-4331-9dfc-e3c7a1eba3ce
md"""

My generative model:

$$\begin{aligned}
\Delta_{\text{Age}} \sim \text{HalfNormal}(\alpha, \sigma)\\
\alpha \sim \text{Uniform}(0, 10)\\
\sigma \sim \text{exp}(1)\\
\end{aligned}$$
"""

# ╔═╡ e1ecca17-8fb9-4aba-bea7-01e6590a57b2
stan_model2 = "
data {
  int<lower=0> N;
  vector<lower=0> [N] DeltaHeight;
}
parameters {
  real alpha;
  real sigma;
}
model {
  alpha ~ uniform(0, 10);
  sigma ~ exponential(2);
  DeltaHeight ~ normal(alpha, sigma);
}
";

# ╔═╡ 7b3b4c5c-0377-4c72-8b30-8ff067dbed81
function fit2(data)
  model = SampleModel("truncated_normal", stan_model2)
  data_dict = Dict(
    "N" => size(data, 1),
    "DeltaHeight" => data.ΔHeight,
  )
  rc = stan_sample(model; data=data_dict)
  if success(rc)
    return model
  end
  nothing
end

# ╔═╡ 32f86b7b-424d-46f8-8238-028d7ad3b600
oxboys_model = fit2(oxboys);

# ╔═╡ 6707856c-68e7-4296-98f8-1d0954fa2116
describe(oxboys_model)

# ╔═╡ 3aa1fc9d-0e30-4195-a2aa-19b4bf050800
post2 = read_samples(oxboys_model);

# ╔═╡ 260e334b-1323-4379-8665-16a885c12b31
post_total_growth = rand.(Normal.(post2.alpha, post2.sigma), 9) .|> sum

# ╔═╡ 3785268b-a53c-4f2d-96cf-ab933bb7a9fa
density(post_total_growth, label="Posterior total growth")

# ╔═╡ Cell order:
# ╠═1aff8e9b-209a-47b5-acd2-c2e23581f7b5
# ╠═e9b1ec8c-b6cb-11ee-0969-579d19504652
# ╠═489f45c6-974d-4c39-90ee-be218e40dc0a
# ╠═e29d7d00-6c3e-4662-8707-f40e92d05005
# ╠═ce77b783-8c75-440e-b53f-d25c7ef757ee
# ╠═9ab20f92-ae18-42be-a07b-b0d835c74b31
# ╠═5e837e37-2a01-45ef-809b-3193e2ddb95d
# ╠═fad0aa05-67d2-4004-aef1-fe37d56c8f43
# ╠═331981a6-9411-478f-b649-365caeb17c24
# ╠═4b6caf18-014f-446b-aba9-eeb3ba39c3d8
# ╠═7e3d943f-fabf-46b2-a79f-8fcf97ea16d7
# ╠═60d2e0a9-76c6-474b-bca5-c5c914bc4645
# ╠═e78c4c12-dfb5-4df5-90c9-fa274d114f8b
# ╠═3dbd1d0a-975a-40ec-8ef6-5ed57a9976b1
# ╠═a5e311d4-278a-4258-ba58-cdebb29e34b1
# ╠═dda1a083-83a8-40de-9703-c1eedaf48ff9
# ╠═213e48a5-e8f9-4223-8906-d4e5dcc37d2d
# ╠═b5a71427-e6ad-4405-9046-0aa068607705
# ╠═d937f3cf-d6d7-404f-9f1d-ee078a2f0960
# ╠═ac938fc3-c942-4e6f-b5ca-902dbb5944f2
# ╠═9b434802-e443-469d-8c15-f2a128c6c1c9
# ╠═885c4d29-8898-4708-ae18-fd95a9798404
# ╠═f815c70f-c2cc-4146-8965-89ff677d6771
# ╠═d995233a-e5c1-4f02-819f-387a0960c37a
# ╠═a6e40368-9d58-494d-8de3-3891f3da6e82
# ╠═23602a27-a721-40cf-bd4d-58ffa17eabde
# ╠═4003e2df-cbb8-4c91-bc7b-bc15bb7a952b
# ╠═f99adb4b-acac-4124-971b-9c11acf6b1bd
# ╠═50d468c9-ed40-4299-980b-5c13c043c189
# ╠═b9f8e03b-210c-4484-bc99-ed9543905bb2
# ╠═843e71f9-d6b3-45cf-9f27-213d4fab5683
# ╠═4626a594-6897-4060-b7f7-64b3f5c2f56d
# ╠═f98a8877-7755-4019-a5f1-06e527d3fd01
# ╠═9bb5ddbc-f892-425e-bea2-886ec60b92df
# ╠═4f6e513d-65eb-4331-9dfc-e3c7a1eba3ce
# ╠═e1ecca17-8fb9-4aba-bea7-01e6590a57b2
# ╠═7b3b4c5c-0377-4c72-8b30-8ff067dbed81
# ╠═32f86b7b-424d-46f8-8238-028d7ad3b600
# ╠═6707856c-68e7-4296-98f8-1d0954fa2116
# ╠═3aa1fc9d-0e30-4195-a2aa-19b4bf050800
# ╠═260e334b-1323-4379-8665-16a885c12b31
# ╠═3785268b-a53c-4f2d-96cf-ab933bb7a9fa
