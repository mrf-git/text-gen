text-gen
========

Hosts a server to take a text prompt and generate a text content.

## Local setup/build instructions

**Prerequisites**
* Docker
* GPU hardware
* Python
* VSCode

First create a directory under `base_image/` called `models/`.

Next create a directory under `models/` called `TinyLlama-1.1B-Chat-v1.0/` and download the
[TinyLlama/TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/tree/main) model files.

```
cd base_image/models
git lfs install
git clone --single-branch --depth=1 -b main https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

Build the base_image container with the VSCode task, and then build and run the main docker image using the VSCode task.

