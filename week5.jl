### A Pluto.jl notebook ###
# v0.19.37

using Markdown
using InteractiveUtils

# ╔═╡ f5bb2bc8-a6c5-4f92-8a1a-2f61aeda6d79
using Pkg

# ╔═╡ b1ca1713-c403-4193-9e47-f9ebaa94c7e7
# ╠═╡ show_logs = false
Pkg.activate(".")

# ╔═╡ cbb0cf2b-0004-4f6a-9bec-563773b40511
using Dagitty, CSV, DataFrames, StanSample, StatsPlots, CategoricalArrays, Distributions

# ╔═╡ 2dd0f4e0-c823-11ee-0044-95ce18b618ee
md"""
#### Author: Curro Campuzano Jiménez (@currocam)
"""

# ╔═╡ ec26760e-753e-49dd-8891-fe4fbaa24f68
md"""
## Week 5
### Exercise 1

The data in data(NWOGrants) are outcomes for scientific funding applications
for the Netherlands Organization for Scientific Research (NWO) from 2010–2012
(see van der Lee and Ellemers doi:10.1073/pnas.1510159112). These data have a
structure similar to the UCBAdmit data discussed in Chapter 11 and in lecture.
There are applications and each has an associated gender (of the lead researcher).
But instead of departments, there are disciplines. Draw a DAG for this sample. Then
use the backdoor criterion and a binomial GLM to estimate the TOTAL causal effect
of gender on grant awards.

"""

# ╔═╡ 04848740-af79-4276-9e01-ba8f152c2258
function NWOGrants()
	url = "https://raw.githubusercontent.com/rmcelreath/rethinking/master/data/NWOGrants.csv"
	CSV.read(download(url), DataFrame)
end

# ╔═╡ 01649a63-ccac-40d2-ae03-08b91833baf9
data = NWOGrants()

# ╔═╡ e5ea1b8c-e722-4aa7-bc12-7a1cbd3a7dc1
g = DAG(:discipline => :award, :gender => :award, :gender => :discipline)

# ╔═╡ c3b1faae-478c-49bc-b712-d1787e940a2b
drawdag(g)

# ╔═╡ c4b66abb-07f8-4608-960e-99fd4dcb2d6e
stan_total_effect = "
data {
  int<lower=0> N;
  array[N] int<lower=1> gender;
  array[N] int<lower=1> applications;
  array[N] int<lower=0> awards;
}
parameters {
  array[2] real alpha;
}
model {
	alpha ~ normal(-1, 1);
	awards ~ binomial(applications, inv_logit(alpha[gender]));
}
";

# ╔═╡ b031924d-3535-4616-b321-04929cae96e0
function fit_total_gender(data)
  model = SampleModel("total_effect_gender", stan_total_effect)
  data_dict = Dict(
    "N" => size(data, 1),
    "gender" => ifelse.(data.gender .== "m", 1, 2),
    "awards" => data.awards,
    "applications" => data.applications,
  )
  rc = stan_sample(model; data=data_dict)
  if success(rc)
    return model
  end
  nothing
end

# ╔═╡ 66bf61d3-2552-458e-a1dd-3235d8808d80
 inv_logit = x->exp(x)/(1+exp(x))

# ╔═╡ f5daf0b3-8619-4340-bbcb-fc516f34e91b
# ╠═╡ show_logs = false
m1 = fit_total_gender(data);

# ╔═╡ eea10fa6-83ed-42e0-bc21-bed6af25e728
post1 = read_samples(m1)

# ╔═╡ 6062001e-6ef1-44dd-b999-4c6565eed542
density(
	inv_logit.(post1[1]) .- inv_logit.(post1[2]) ,
	labels=false,
	xaxis="Gender contrast (probability)",
	yaxis="Density", colour=:red
)

# ╔═╡ ca8bd31c-86a8-4077-aa01-ffdc56b1e382
md"""
## Exercise 2
Now estimate the DIRECT causal effect of gender on grant awards. Use the same
DAG as above to justify one or more binomial models. Compute the average direct
causal effect of gender, weighting each discipline in proportion to the number of
applications in the sample. Refer to the marginal effect example in Lecture 9 for
help.

First, we can compute the posterior per discipline:
"""

# ╔═╡ 3d31be76-bd28-4ec6-95c3-31abf4ed1a52
stan_direct_effect = "
data {
  int<lower=0> N;
  int<lower=1> K;
  array[N] int<lower=1> gender;
  array[N] int<lower=1> applications;
  array[N] int<lower=1> discipline;
  array[N] int<lower=0> awards;
}
parameters {        
  array[2, K] real alpha;
}
model {
	alpha[1] ~ normal(-1, 1);
	alpha[2] ~ normal(-1, 1);
	for (n in 1:N) {
		awards[n] ~ binomial(applications[n], inv_logit(alpha[gender[n], discipline[n]]));
	}
}
";

# ╔═╡ f6c50524-19a6-410e-be33-5a67b841d4fa
function fit_direct_gender(data)
  discipline = data.discipline |> categorical .|> levelcode
  model = SampleModel("direct_effect_gender", stan_direct_effect)
  data_dict = Dict(
    "N" => size(data, 1),
	"K" => maximum(discipline),
    "gender" => ifelse.(data.gender .== "m", 1, 2),
    "awards" => data.awards,
	"discipline" => discipline,
    "applications" => data.applications,
  )
  rc = stan_sample(model; data=data_dict)
  if success(rc)
    return model
  end
  nothing
end

# ╔═╡ 5c5ff246-011c-4e1e-b881-f7525a41c6bc
# ╠═╡ show_logs = false
m2 = fit_direct_gender(data);

# ╔═╡ 0c26e818-943a-4550-ac19-9cfc51cdf9a5
post2 = read_samples(m2)

# ╔═╡ 95ba97c7-117d-4e5d-b500-1dce9452d04d
begin
	alphas = matrix(post2)
	diffs = zeros((4000, 9))
	for i in 1:9
		diffs[:,i] .= inv_logit.(alphas[:, i*2-1]) - inv_logit.(alphas[:, i*2])
	end
	density(diffs, label = reshape(data.discipline |> categorical |> levels, (1, 9)))
	xaxis!("Gender contrast")
	yaxis!("Density")
end


# ╔═╡ 4edbd785-da4c-4737-8528-8b02005af5fa
md"""
And, then, the weighted one (we draw a number of samples proportional to the total number of applications).
"""

# ╔═╡ 89b98ce4-d52c-4663-b157-5992cc962ddc
counts = sort(combine(groupby(data, :discipline), :applications => sum), :discipline)

# ╔═╡ aa439d1d-559f-42f2-94ae-f3af1756d603
total_apps = sum(counts.applications_sum)

# ╔═╡ e774f034-9746-40e9-af88-3adcf50e0a73
begin
	weighted_contrast = Float64[]
	sizehint!(weighted_contrast, total_apps*1000)
	for i in 1:9
		n_draws = counts.applications_sum[i] * 1000
		append!(weighted_contrast, sample(alphas[:, i*2-1], n_draws) - sample(alphas[:, i*2], n_draws))
	end	
	weighted_contrast
end

# ╔═╡ 93b156f5-592f-4c61-a6e8-ca039701064e
density(weighted_contrast, label=false, xaxis = "Weighted Gender Contrast", yaxis = "Density")

# ╔═╡ Cell order:
# ╟─2dd0f4e0-c823-11ee-0044-95ce18b618ee
# ╟─f5bb2bc8-a6c5-4f92-8a1a-2f61aeda6d79
# ╟─b1ca1713-c403-4193-9e47-f9ebaa94c7e7
# ╟─ec26760e-753e-49dd-8891-fe4fbaa24f68
# ╠═cbb0cf2b-0004-4f6a-9bec-563773b40511
# ╠═04848740-af79-4276-9e01-ba8f152c2258
# ╠═01649a63-ccac-40d2-ae03-08b91833baf9
# ╠═e5ea1b8c-e722-4aa7-bc12-7a1cbd3a7dc1
# ╠═c3b1faae-478c-49bc-b712-d1787e940a2b
# ╠═c4b66abb-07f8-4608-960e-99fd4dcb2d6e
# ╠═b031924d-3535-4616-b321-04929cae96e0
# ╠═66bf61d3-2552-458e-a1dd-3235d8808d80
# ╠═f5daf0b3-8619-4340-bbcb-fc516f34e91b
# ╠═eea10fa6-83ed-42e0-bc21-bed6af25e728
# ╠═6062001e-6ef1-44dd-b999-4c6565eed542
# ╟─ca8bd31c-86a8-4077-aa01-ffdc56b1e382
# ╠═3d31be76-bd28-4ec6-95c3-31abf4ed1a52
# ╠═f6c50524-19a6-410e-be33-5a67b841d4fa
# ╠═5c5ff246-011c-4e1e-b881-f7525a41c6bc
# ╠═0c26e818-943a-4550-ac19-9cfc51cdf9a5
# ╠═95ba97c7-117d-4e5d-b500-1dce9452d04d
# ╟─4edbd785-da4c-4737-8528-8b02005af5fa
# ╠═89b98ce4-d52c-4663-b157-5992cc962ddc
# ╠═aa439d1d-559f-42f2-94ae-f3af1756d603
# ╠═e774f034-9746-40e9-af88-3adcf50e0a73
# ╠═93b156f5-592f-4c61-a6e8-ca039701064e
