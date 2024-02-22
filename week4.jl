### A Pluto.jl notebook ###
# v0.19.37

using Markdown
using InteractiveUtils

# ╔═╡ ebf977b2-9891-4682-a552-0a1f94a00d37
using Pkg; Pkg.activate(".")

# ╔═╡ 5d30627f-54a8-4c4f-a0d0-a90f4f73f28a
using CSV, DataFrames, StanSample, StatsBase, StatsPlots, Distributions, RCall, Random

# ╔═╡ 3a843146-c28f-11ee-2137-8fc75a5bb852
md"""
#### Author: Curro Campuzano Jiménez (@currocam)
"""

# ╔═╡ 393d51b4-21e5-434d-9d5e-590b7f761d23
md"""
# Week 4
## Question 1

Revisit the marriage, age, and happiness collider bias example from Chapter 6. Run models m6.9 and m6.10 again (pages 178–179). Compare these
two models using both PSIS and WAIC. Which model is expected to make
better predictions, according to these criteria, and which model yields the
correct causal inference?

##### Data simulation
First, we have to simulate the data. Here is the design (from the book).

1. Each year, 20 people are born with uniformly distributed happiness values.
2. Each year, each person ages one year. Happiness does not change.
3. At age 18, individuals can become married. The odds of marriage each year are proportional to an individual’s happiness.
4. Once married, an individual remains married.
5. After age 65, individuals leave the sample. (They move to Spain.)
"""

# ╔═╡ b0ee36ed-a97d-4a4d-87de-c808cfd41244
function happiness_simulation()
	N_years=1000; max_age=65; N_births=20; aom=18
	ages = Int[]; happiness = Float64[]; marital_status = Bool[];
	for year in 1:N_years
		ages .+=1
		prepend!(ages, ones(N_births))
		prepend!(happiness, range(-2, 2, N_births))
		prepend!(marital_status, zeros(N_births))
		retired = findall(a -> a >max_age, ages)
		deleteat!(ages, retired)
		deleteat!(happiness, retired)
		deleteat!(marital_status, retired)
		for ((i, status), age) in zip(enumerate(marital_status), ages)
			if !status && age > aom
				marital_status[i] = happiness[i] -4 |> BernoulliLogit |> rand
			end
		end
	end
	DataFrame([:age => ages, :married => marital_status, :happiness => happiness])
end


# ╔═╡ 0ae3983e-12f9-403d-afb1-0201c3890dd5
data = filter(:age => >(17), happiness_simulation())

# ╔═╡ 3860469f-a3a4-4709-8ee3-38c96ec8b241
md"""
##### Pointwise log-likelihood
Now, we have to fit both models and extract the pointwise log-likelihood.
"""

# ╔═╡ 0e1eeaf9-afd6-4fc6-baf2-93beca18c4e4
function m6_9(data)
	stan_model = "
data {
	int <lower=1> N;
	vector[N] happiness;
	vector[N] A;
	array[N] int married;
}
parameters {
  vector[2] alphas;
  real beta;
  real sigma;
}
model {
	vector[N] mu;
	sigma ~ exponential(1);
	alphas ~ normal(0, 1);
	beta ~ normal(0, 2);  
	for (i in 1:N) 
		mu[i] = alphas[married[i]] + beta * A[i];
	happiness ~ normal(mu, sigma);
}
generated quantities {
  vector[N] log_lik;
  for (n in 1:N) 
	log_lik[n] = normal_lpdf(happiness[n] | alphas[married[n]] + A[n] * beta, sigma);
}
";
	model = SampleModel("m6.9", stan_model)
  data_dict = Dict(
    "N" => size(data, 1),
    "happiness" => data.happiness,
    "A" => (data.age .- 18) ./ ( 65 - 18 ),
	"married" => Int.(data.married) .+ 1
  )
  rc = stan_sample(model; data=data_dict)
  success(rc) ? model : nothing
end

# ╔═╡ fd068386-c913-4c27-9422-10b62d190f55
# ╠═╡ show_logs = false
begin
	post1 = data |> m6_9 |> read_samples
	log_lik_1 = reduce(hcat, [post1[i] for i in 5:lastindex(post1)])
end

# ╔═╡ 508a3c79-ddeb-4059-9ded-dc159724dad2
md"""
And now, the second model where we exclude the collider. 
"""

# ╔═╡ a9cf4ae2-0597-4a52-8ff2-7872d3a90e7b
function m6_10(data)
	stan_model = "
data {
	int <lower=1> N;
	vector[N] happiness;
	vector[N] A;
}
parameters {
  real alpha;
  real beta;
  real sigma;
}
model {
  sigma ~ exponential(1);
  beta ~ normal(0, 2);
  alpha ~ normal(0, 1);
  happiness ~ normal(alpha + beta*A, sigma);
}
	
generated quantities {
  vector[N] log_lik;
  for (n in 1:N) {
    log_lik[n] = normal_lpdf(happiness[n] | alpha + beta*A[n], sigma);
  }
}
";
	model = SampleModel("m6.10", stan_model)
  data_dict = Dict(
    "N" => size(data, 1),	  
    "happiness" => data.happiness,
    "A" => (data.age .- 18) ./ ( 65 - 18 ),
  )
  rc = stan_sample(model; data=data_dict)
  if success(rc)
    return model
  end
  nothing
end

# ╔═╡ 0736aa8b-d00a-4d18-a0ba-1751b022039c
# ╠═╡ show_logs = false
begin
	post2 = data |> m6_10 |> read_samples
	log_lik_2 = reduce(hcat, [post2[i] for i in 4:lastindex(post2)])
end

# ╔═╡ b85dd9ac-5f46-4149-a42a-774917b80a52
md"""
We compared both models using WAIC and PSIS (PSIS-LOO CV). We used in both cases the R package loo. 
"""

# ╔═╡ 190ed2af-652b-47f9-a9d3-c12348303b14
# ╠═╡ show_logs = false
begin
	@rput log_lik_1
	@rput log_lik_2
	R"""
	library("loo")
	waic1 <- waic(log_lik_1)
	waic2 <- waic(log_lik_2)
	loo_compare(waic1, waic2)
	"""
end

# ╔═╡ 8ce50eee-694c-4984-9068-d03b4a61ff49
R"""
chain_id <- rep(1:4, each = 1000)
r_eff1 <- relative_eff(exp(log_lik_1), chain_id = chain_id, cores = 4) 
r_eff2 <- relative_eff(exp(log_lik_2), chain_id = chain_id, cores = 4) 
loo_1 <- loo(log_lik_1, r_eff = r_eff1, cores = 4)
loo_2 <- loo(log_lik_2, r_eff = r_eff2, cores = 4)
loo_compare(loo_1, loo_2)
"""

# ╔═╡ c1af2132-4a41-4bdf-8e84-17514451a43a
md"""
As expected, the first model (the one with the collider), predicts better than the causally correct one. 

## Exercise 2

Reconsider the urban fox analysis from last week’s homework. First, we refit the models again (no explanation included). 
"""

# ╔═╡ 11e04cb4-098a-4bc8-9267-2cb6d7faa97f
begin
	function Foxes()
	  url = "https://raw.githubusercontent.com/rmcelreath/rethinking/master/data/foxes.csv"
	  CSV.read(download(url), DataFrame)
	end
	foxes = Foxes()
	function model3_foxes(data)
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
	generated quantities {
	  vector[N] log_lik;
	  for (n in 1:N) {
	    log_lik[n] = normal_lpdf(weight[n] | alpha + beta_f * avgfood[n] + beta_g * groupsize[n], sigma);
	  }
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
	function model2_foxes(data)
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
	generated quantities {
	  vector[N] log_lik;
	  for (n in 1:N) {
	    log_lik[n] = normal_lpdf(weight[n] | alpha + beta*avgfood[n], sigma);
	  }
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
end

# ╔═╡ 3fb4c7f7-663b-4644-b87e-f35f95a92dcb
# ╠═╡ show_logs = false
begin
	post2_foxes = foxes |> model2_foxes |> read_samples
	log_lik_foxes_2 = reduce(hcat, [post2_foxes[i] for i in 4:lastindex(post2_foxes)])
	post3_foxes = foxes |> model3_foxes |> read_samples
	log_lik_foxes_3 = reduce(hcat, [post3_foxes[i] for i in 5:lastindex(post3_foxes)])
end

# ╔═╡ 41515c1e-3afb-490d-bdfa-cb4c18c36ae9
begin
	@rput log_lik_foxes_2
	@rput log_lik_foxes_3
	R"""
	waic2_foxes <- waic(log_lik_foxes_2)
	waic3_foxes <- waic(log_lik_foxes_3)
	loo_compare(waic2_foxes, waic3_foxes)
	"""
end

# ╔═╡ 0fdff5a9-bb06-44f0-939a-e09d7c387c2e
R"""
r_eff2_foxes <- relative_eff(exp(log_lik_foxes_2), chain_id = chain_id, cores = 4) 
r_eff3_foxes <- relative_eff(exp(log_lik_foxes_3), chain_id = chain_id, cores = 4) 
loo_2_foxes <- loo(log_lik_foxes_2, r_eff = r_eff2_foxes, cores = 4)
loo_3_foxes <- loo(log_lik_foxes_3, r_eff = r_eff3_foxes, cores = 4)
loo_compare(loo_2_foxes, loo_3_foxes)
"""

# ╔═╡ 56477469-5cfa-4a84-9941-9232cb00492c
md"""
 On the
basis of PSIS and WAIC scores, which combination of variables best predicts
body weight (W, weight)? 

WAIC and PSIS suggested that the combination of groupsize and average food. 

What causal interpretation can you assign each coefficient (parameter) from the best scoring model?

For that model, we have the parameters $\alpha$, $\beta_g$, $\beta_f$ and $\sigma$. We can interpretate $\beta_f$ to be the direct causal effect of food in weight.  
"""

# ╔═╡ 9da0df20-57e3-40c3-9682-d642e1e60cae
mean(post3_foxes.beta_f)

# ╔═╡ bf97a9d7-ffff-4389-808f-fe96959f54d8
md"""
## Exercise 3

The data in data(Dinosaurs) are body mass estimates at different estimated ages for six different dinosaur species. See ?Dinosaurs for more details. Choose one or more of these species (at least one, but as many as you like) and model its growth. To be precise: Make a predictive model of body mass using age as a predictor. Consider two or more model types for the function relating age to body mass and score each
using PSIS and WAIC.
"""

# ╔═╡ 8531c278-3d0b-44ec-b0bc-631d4a33013e
function Dinosaurs()
  url = "https://raw.githubusercontent.com/rmcelreath/rethinking/master/data/Dinosaurs.csv"
  CSV.read(download(url), DataFrame)
end

# ╔═╡ 808b8261-832b-499c-97b1-289d16ee9cb4
dinosaurs = Dinosaurs()

# ╔═╡ 36dc2593-e1ff-4355-b341-b927aaac6f60
begin
	using CategoricalArrays
	species = dinosaurs.species |> categorical  .|> levelcode
	age = (dinosaurs.age .- minimum(dinosaurs.age)) ./ ( maximum(dinosaurs.age) - minimum(dinosaurs.age) )
	mass = standardize(ZScoreTransform, dinosaurs.mass)
end

# ╔═╡ 7b442fc6-cc5e-4cad-b7cb-f4d32984ad26
md"""
We are going to compare two models. The first model things that both rate and intercept are shared between species (but uses a t-student to allow more variation). The other fits specific parameters for each specie. 
"""

# ╔═╡ 2db858a4-b842-48a1-9bc6-7f76481a72bb
	function simple_dinosaurs(species, age, mass, N, K)
		stan_model = "
	data {
	  int<lower=0> N;
	  int<lower=1> K;
	  array[N] int species;
	  vector[N] age;
	  vector[N] mass;
	}
	parameters {
 	  vector[K] alpha;
	  real beta;
	  real sigma;
	}
	model {
	  sigma ~ exponential(1);
	  alpha ~ normal(0, 2);
	  beta ~ normal(0, 2);
	  for (i in 1:N) 
		mass[i] ~ student_t(2, alpha[species[i]] + beta * age[i], sigma);
	}
	generated quantities {
	  vector[N] log_lik;
	  for (n in 1:N) {
	    log_lik[n] = student_t_lpdf(mass[n] | 2, alpha[species[n]] + beta * age[n], sigma);
	  }
	}
	";
		model = SampleModel("simple_dino", stan_model)
	  data_dict = Dict(
		  "age" => age, "N" => N, "mass" => mass, "species" => species, "K" => K
	  )
	  rc = stan_sample(model; data=data_dict)
	  success(rc) ? model : nothing
	end


# ╔═╡ 4d851f22-6341-42a3-ae62-5defed03bec6
# ╠═╡ show_logs = false
simple_dino = simple_dinosaurs(species, age, mass, length(species), maximum(species));

# ╔═╡ 07862ab0-4106-40a5-905a-67f2079c3f83
	function complex_dinosaurs(species, age, mass, N, K)
		stan_model = "
	data {
	  int<lower=0> N;
	  int<lower=1> K;
	  array[N] int species;
	  vector[N] age;
	  vector[N] mass;
	}
	parameters {
 	  vector[K] alpha;
	  vector[K] beta;
	  vector[K] sigma;
	}
	model {
	  sigma ~ exponential(1);
	  alpha ~ normal(0, 1);
	  beta ~ normal(0, 1);
	  for (i in 1:N) 
		mass[i] ~ normal(alpha[species[i]] + beta[species[i]] * age[i], sigma[species[i]]);
	}
	generated quantities {
	  vector[N] log_lik;
	  for (n in 1:N) {
	    log_lik[n] = normal_lpdf(mass[n] | alpha[species[n]] + beta[species[n]] * age[n], sigma[species[n]]);
	  }
	}
	";
		model = SampleModel("simple_dino", stan_model)
	  data_dict = Dict(
		  "age" => age, "N" => N, "mass" => mass, "species" => species, "K" => K
	  )
	  rc = stan_sample(model; data=data_dict)
	  success(rc) ? model : nothing
	end


# ╔═╡ 29f2075f-cb67-4be8-9fc2-a6ca50afa746
# ╠═╡ show_logs = false
complex_dino = complex_dinosaurs(species, age, mass, length(species), maximum(species));

# ╔═╡ 0ec1996a-c0bd-4651-90b3-133774f221d9
md"""
Which model do you think is best, on predictive grounds? On scientific
grounds? If your answers to these questions differ, why?
This is a challenging exercise, because the data are so scarce. But it is also a
realistic example, because people publish Nature papers with even less data.
So do your best, and I look forward to seeing your growth curves.
I expect the simple model work better, as we don't have enough data. However, on predictive grounds, the more complex model works better, as indicated by PSIS and WAIC. 
"""

# ╔═╡ ab922a5f-5817-4955-bdfe-cacdb52e7cd0
begin
	symbols = Symbol.("log_lik." .* string.(1:32))
	post_simple_dino = simple_dino |> read_samples
	log_lik_simple_dino = reduce(hcat, [post_simple_dino[sym] for sym in symbols])
	post_complex_dino = complex_dino |> read_samples
	log_lik_complex_dino = reduce(hcat, [post_complex_dino[sym] for sym in symbols])
end

# ╔═╡ 7c5ae67e-a157-4da7-8ee0-638a970b97e1
begin
	@rput log_lik_simple_dino
	@rput log_lik_complex_dino
	R"""
	r_eff_simple_dino <- relative_eff(exp(log_lik_simple_dino), chain_id = chain_id, cores = 4) 
	r_eff_complex_dino <- relative_eff(exp(log_lik_complex_dino), chain_id = chain_id, cores = 4) 
	loo_simple_dino <- loo(log_lik_simple_dino, r_eff = r_eff_simple_dino, cores = 4)
	loo_complex_dino <- loo(log_lik_complex_dino, r_eff = r_eff_complex_dino, cores = 4)
	loo_compare(loo_simple_dino, loo_complex_dino)
	"""
end

# ╔═╡ 1b6caefa-6346-4c2d-983c-bb7e53615a01
md"""
Let's visualize for a few dinosaurs. A simple line works works reasonably well, even if it doesn't have "biological" sense. 
"""

# ╔═╡ 53b9b4ef-64f2-4c84-ac3c-f55693077db8
begin
	x = range(0, 1.0, 50)
	y = [post_complex_dino[Symbol("alpha.3")][i] .+ x .* post_complex_dino[Symbol("beta.3")][i] for i in 1:50]
	plot(x, y, legend = false)
	scatter!(age[species .== 3], mass[species .== 3])
	xaxis!("Normalized age")
	yaxis!("Standarized mass")
end

# ╔═╡ 1dec8902-9c0b-4646-946d-15f0bbbd32bb
md"""
When we don't have enough data, is nice to see that the "uncertainty" is captured when extrapolating. 
"""

# ╔═╡ c5593f0b-65ee-424f-961a-3a0c8fd4b4f5
begin
	y2 = [post_complex_dino[Symbol("alpha.5")][i] .+ x .* post_complex_dino[Symbol("beta.5")][i] for i in 1:50]
	plot(x, y2, legend = false)
	scatter!(age[species .== 5], mass[species .== 5])
	xaxis!("Normalized age")
	yaxis!("Standarized mass")
end

# ╔═╡ Cell order:
# ╠═3a843146-c28f-11ee-2137-8fc75a5bb852
# ╠═ebf977b2-9891-4682-a552-0a1f94a00d37
# ╠═5d30627f-54a8-4c4f-a0d0-a90f4f73f28a
# ╠═393d51b4-21e5-434d-9d5e-590b7f761d23
# ╠═b0ee36ed-a97d-4a4d-87de-c808cfd41244
# ╠═0ae3983e-12f9-403d-afb1-0201c3890dd5
# ╠═3860469f-a3a4-4709-8ee3-38c96ec8b241
# ╠═0e1eeaf9-afd6-4fc6-baf2-93beca18c4e4
# ╠═fd068386-c913-4c27-9422-10b62d190f55
# ╠═508a3c79-ddeb-4059-9ded-dc159724dad2
# ╠═a9cf4ae2-0597-4a52-8ff2-7872d3a90e7b
# ╠═0736aa8b-d00a-4d18-a0ba-1751b022039c
# ╠═b85dd9ac-5f46-4149-a42a-774917b80a52
# ╠═190ed2af-652b-47f9-a9d3-c12348303b14
# ╠═8ce50eee-694c-4984-9068-d03b4a61ff49
# ╠═c1af2132-4a41-4bdf-8e84-17514451a43a
# ╠═11e04cb4-098a-4bc8-9267-2cb6d7faa97f
# ╠═3fb4c7f7-663b-4644-b87e-f35f95a92dcb
# ╠═41515c1e-3afb-490d-bdfa-cb4c18c36ae9
# ╠═0fdff5a9-bb06-44f0-939a-e09d7c387c2e
# ╠═56477469-5cfa-4a84-9941-9232cb00492c
# ╠═9da0df20-57e3-40c3-9682-d642e1e60cae
# ╠═bf97a9d7-ffff-4389-808f-fe96959f54d8
# ╠═8531c278-3d0b-44ec-b0bc-631d4a33013e
# ╠═808b8261-832b-499c-97b1-289d16ee9cb4
# ╠═7b442fc6-cc5e-4cad-b7cb-f4d32984ad26
# ╠═36dc2593-e1ff-4355-b341-b927aaac6f60
# ╠═2db858a4-b842-48a1-9bc6-7f76481a72bb
# ╠═4d851f22-6341-42a3-ae62-5defed03bec6
# ╠═07862ab0-4106-40a5-905a-67f2079c3f83
# ╠═29f2075f-cb67-4be8-9fc2-a6ca50afa746
# ╠═0ec1996a-c0bd-4651-90b3-133774f221d9
# ╠═ab922a5f-5817-4955-bdfe-cacdb52e7cd0
# ╠═7c5ae67e-a157-4da7-8ee0-638a970b97e1
# ╠═1b6caefa-6346-4c2d-983c-bb7e53615a01
# ╠═53b9b4ef-64f2-4c84-ac3c-f55693077db8
# ╠═1dec8902-9c0b-4646-946d-15f0bbbd32bb
# ╠═c5593f0b-65ee-424f-961a-3a0c8fd4b4f5
