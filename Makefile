.PHONY: run
run:
	julia --project=. -p auto -e 'import Pluto; Pluto.run()'