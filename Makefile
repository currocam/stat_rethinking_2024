.PHONY: run
run:
	julia -e 'import Pluto; Pluto.run()'

.PHONY: deps
deps:
	julia -e 'import Pkg; Pkg.add("Pluto")'