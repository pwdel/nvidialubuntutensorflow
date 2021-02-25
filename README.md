# Installing Tensorflow on x64 Lubuntu

## Objective

To get an idea of what it's like to install Tensorflow directly on a local machine, to better understand some of the challenges and processes vs. using Tensorflow in a cloud environment. Basically, to get a better feeling of being closer to the bare metal.

## Problems with Lubuntu, Replacement with Ubuntu

So, the first problem that I had was that Lubuntu permanently crashed on an HP Omen Desktop with an NVidia GeForce GTX 1700 that I had been using, so I just went ahead and installed Ubuntu instead. I had thought that reducing the operating system memory load from 3GB down to sub 1GB would be a nice way to free up some performance, but it seems like the hassle of Lubuntu isn't worth it and is getting in the way.

[Tensorflow does have some installing GPU documentation](https://www.tensorflow.org/install/gpu), but Ubuntu starts right off during its installation phase and asks whether you would like to install any third party drivers. I went ahead and did that during Ubuntu installation.

The version of Ubuntu is 20.04 LTS.

## Download the GeForce GTX Nvidia Driver

This was done during Ubuntu installation, but otherwise the documentation and driver for that can be found [here](https://www.nvidia.com/download/driverResults.aspx/104284/en-us).

## Installing Tensorflow

Rather than installing [Tensorflow directly on my computer](https://www.tensorflow.org/install/gpu), I thought it would make more sense to work with a [Docker Container](https://www.tensorflow.org/install/docker), which uses the underlying system resources of the GPU below.

1. Downloading the [docker image](https://hub.docker.com/r/tensorflow/tensorflow/) is done through 'docker pull tensorflow/tensorflow'
2. However, we need to use the optional feature with the -gpu tag, which is based upon NVIDIA CUDA.
3. We need [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) to run the -gpu tag options.
4. The installation guide for [nvidia-docker is here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker).

### Setting Up the NVIDIA Container toolkit

1. We setup the GPG Key.
2. If desired, install the experimental branch. I did not do that.
3. 'sudo apt-get update' and then 'sudo apt-get install -y nvidia-docker2'
4. Restart docker. `sudo systemctl restart docker`
5. Test by running a base CUDA container. `sudo docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi`

I got a success message as shown below:

Thu Feb 25 21:40:53 2021       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 460.39       Driver Version: 460.39       CUDA Version: 11.2     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  GeForce GTX 1070    Off  | 00000000:01:00.0  On |                  N/A |
| 28%   32C    P8     7W / 151W |    236MiB /  8117MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
+-----------------------------------------------------------------------------+
