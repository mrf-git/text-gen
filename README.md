text-gen
========

Hosts a server to take a text prompt and generate a text content.

## Local setup/build instructions

**Prerequisites**
* Docker
* GPU hardware
* Python
* VSCode


To build `llama-cpp-python` on Windows, you need to first set the environment variables:

```
set CMAKE_GENERATOR=MinGW Makefiles
set CMAKE_ARGS=-DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ -DCMAKE_ASM_COMPILER=gcc -DCMAKE_MAKE_PROGRAM=mingw32-make

```

before running `poetry install`. CMake should first be installed and the following packages from MSYS2:
`mingw-w64-x86_64-gcc` and `mingw-w64-x86_64-pkg-config` and `make` or `mingw-w64-i686-make`.


First create a directory under `base_image/` called `models/`.

Next create a directory under `models/` called `Llama-2-7B-Chat-GGUF/` and download the
[TheBloke/Llama-2-7B-Chat-GGUF](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/tree/main) model files.


Build the base_image container with the VSCode task, and then build and run the main docker image using the VSCode task.

