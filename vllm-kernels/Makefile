.PHONY: cmake
cmake:
	mkdir -p build
	(cd build && cmake -DVLLM_PYTHON_EXECUTABLE=$(shell which python) ../)

.PHONY: clean
clean:
	rm -rf build
