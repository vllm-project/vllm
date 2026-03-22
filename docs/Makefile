# Minimal zensical makefile

ZENSICAL_CMD := zensical
ZENSICAL_CONF := ../zensical.toml

.PHONY: help install gen serve build clean

help:
	@echo "Available targets:"
	@echo "  install  Install dependencies globally"
	@echo "  gen      Generate docs content (files + API + nav)"
	@echo "  serve    Serve docs locally"
	@echo "  build    Build static site"
	@echo "  clean    Remove build artifacts"

install:
	pip install -r "../requirements/docs.txt"

gen:
	cd .. && python docs/scripts/zensical_gen_files.py

serve: gen
	cd .. && $(ZENSICAL_CMD) serve

build: gen
	cd .. && $(ZENSICAL_CMD) build

clean:
	rm -rf site/ .cache/ api/vllm/
	rm -rf examples/
	rm -rf ../docs/generated/
	cd .. && python3 docs/scripts/zensical_gen_files.py --clean
