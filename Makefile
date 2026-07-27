meep-docker-build:
	docker build -t meep-docker .

meep-docker-run:
	docker run --mount type=bind,source=./,target=/ms-opt --shm-size="8g" -it meep-docker