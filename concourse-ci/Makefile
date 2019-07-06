.PHONY: build_docker_image
build_docker_image: Dockerfile
	docker build -t ic --file Dockerfile ../

.PHONY: fly_execute
fly_execute:
	fly -t local execute \
		--include_ignored \
		--input IC=../ \
		--config run.yml

.PHONY: fly_set_pipeline
fly_set_pipeline:
	fly -t local set-pipeline \
    --pipeline IC \
    --config pipeline.yml
