ABSOLUTE_PATH := $(shell pwd)
BASE_IMAGE_NAME := learning_programming_list
TRAINING_PATTERN := training_pattern
TRAINING_PROJECT := iris_project
IMAGE_VERSION := 0.0.1

DOCKERFILE := Dockerfile

.PHONY: dev
dev:
	pip install -r requirements.txt

.PHONY: d_build
d_build:
	docker build \
		-t ${BASE_IMAGE_NAME}:$(TRAINING_PATTERN)_$(TRAINING_PROJECT)_$(IMAGE_VERSION) \
		-f $(DOCKERFILE) .

.PHONY: train
train:
	mlflow run .

.PHONY: ui
ui:
	mlflow ui