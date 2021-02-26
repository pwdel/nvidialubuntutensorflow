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

### Installing Tensorflow with GPU Tag

We run: `sudo docker pull tensorflow/tensorflow:latest-gpu`

We could also install a container that includes tensorflow and a jupyter notebook.

`sudo docker pull tensorflow/tensorflow:latest-gpu-jupyter`

## Running Docker with Tensorflow

We can run Tensorflow containers with the following command format:

```
sudo docker run [-it] [--rm] [-p hostPort:containerPort] tensorflow/tensorflow[:tag] [command]
```
To verify that our CPU version can work, we can run:
```
sudo docker run -it tensorflow/tensorflow bash
```
Upon doing this, we get a nice readout:

```
________                               _______________                
___  __/__________________________________  ____/__  /________      __
__  /  _  _ \_  __ \_  ___/  __ \_  ___/_  /_   __  /_  __ \_ | /| / /
_  /   /  __/  / / /(__  )/ /_/ /  /   _  __/   _  / / /_/ /_ |/ |/ /
/_/    \___//_/ /_//____/ \____//_/    /_/      /_/  \____/____/|__/


WARNING: You are running this container as root, which can cause new files in
mounted volumes to be created as the root user on your host machine.

To avoid this, run the container by specifying your user's userid:

$ docker run -u $(id -u):$(id -g) args...

root@64c3bca68909:/#
```
Within this container session we can start python and import tensorflow.

## Running Python Tensorflow with CPU Version

If we run our [CPU version with a basic beginner kickstart program](https://www.tensorflow.org/tutorials/quickstart/beginner), we see some errors showing that a GPU exists, but it's not using the GPU.

We run it with:

```
sudo docker run -it tensorflow/tensorflow bash
```

then...

```
root@64c3bca68909:/# python
Python 3.6.9 (default, Oct  8 2020, 12:12:24)
[GCC 8.4.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import tensorflow as tf
2021-02-25 21:57:32.615412: W tensorflow/stream_executor/platform/default/dso_loader.cc:60] Could not load dynamic library 'libcudart.so.11.0'; dlerror: libcudart.so.11.0: cannot open shared object file: No such file or directory
2021-02-25 21:57:32.615433: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
>>> mnist = tf.keras.datasets.mnist
>>>
>>> (x_train, y_train), (x_test, y_test) = mnist.load_data()
Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
11493376/11490434 [==============================] - 1s 0us/step
>>> x_train, x_test = x_train / 255.0, x_test / 255.0
>>> model = tf.keras.models.Sequential([
...   tf.keras.layers.Flatten(input_shape=(28, 28)),
...   tf.keras.layers.Dense(128, activation='relu'),
...   tf.keras.layers.Dropout(0.2),
...   tf.keras.layers.Dense(10)
... ])
2021-02-25 21:58:35.933087: I tensorflow/compiler/jit/xla_cpu_device.cc:41] Not creating XLA devices, tf_xla_enable_xla_devices not set
2021-02-25 21:58:35.933237: W tensorflow/stream_executor/platform/default/dso_loader.cc:60] Could not load dynamic library 'libcuda.so.1'; dlerror: libcuda.so.1: cannot open shared object file: No such file or directory
2021-02-25 21:58:35.933252: W tensorflow/stream_executor/cuda/cuda_driver.cc:326] failed call to cuInit: UNKNOWN ERROR (303)
2021-02-25 21:58:35.933277: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:163] no NVIDIA GPU device is present: /dev/nvidia0 does not exist
2021-02-25 21:58:35.933459: I tensorflow/core/platform/cpu_feature_guard.cc:142] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 FMA
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2021-02-25 21:58:35.933692: I tensorflow/compiler/jit/xla_gpu_device.cc:99] Not creating XLA devices, tf_xla_enable_xla_devices not set
```
## Running Python Tensorflow with GPU Version

We run this with:

```
sudo docker run -it tensorflow/tensorflow:latest-gpu bash
```
Once in the container, when we import Tensorflow we don't see an error:

```
root@56c1bc02cd22:/# python
Python 3.6.9 (default, Oct  8 2020, 12:12:24)
[GCC 8.4.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import tensorflow as tf
2021-02-25 22:06:38.769622: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcudart.so.11.0
>>>
```
When we attempt to run the model:

```
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])
```
We get the error:

```
This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 FMA
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2021-02-25 22:07:26.673760: I tensorflow/compiler/jit/xla_gpu_device.cc:99] Not creating XLA devices, tf_xla_enable_xla_devices not set
```
Looking online, "Not creating XLA devices" appears to be an error that we can ignore.

To check how many GPUs are detected, we can run:

```
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
```
When we run this we get:

```
Num GPUs Available:  0
```
this is because we have to run with a special tag, as follows:

```
sudo docker run -it --gpus all tensorflow/tensorflow:latest-gpu bash
```
Once we run that, open python, re-import tensorflow and then do the above print command we get:

```
Num GPUs Available:  1
```
So going back and running:

```
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])

```
We get:
```
I tensorflow/core/common_runtime/gpu/gpu_device.cc:1720] Found device 0 with properties:
pciBusID: 0000:01:00.0 name: GeForce GTX 1070 computeCapability: 6.1
coreClock: 1.683GHz coreCount: 15 deviceMemorySize: 7.93GiB deviceMemoryBandwidth: 238.66GiB/s
...
I tensorflow/core/common_runtime/gpu/gpu_device.cc:1406] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 7222 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1070, pci bus id: 0000:01:00.0, compute capability: 6.1
```
Plus a bunch of other lines, which seem like we can ignore them, just operational lines.

We then can output predictions:

```
predictions = model(x_train[:1]).numpy()
predictions
```
Which yields:

```
array([[ 0.01331493,  0.572829  ,  0.264568  , -0.37259772,  0.11883231,
         0.8207073 , -0.7673784 , -0.38996065, -0.3025323 , -0.18874598]],
      dtype=float32)
```
Then tf.nn.softmax converts these logits to probabilities for each class:

```
tf.nn.softmax(predictions).numpy()
```
We then calculate a loss function:
```
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
```
This loss is equal to the negative log probability of the true class: It is zero if the model is sure of the correct class.

This untrained model gives probabilities close to random (1/10 for each class), so the initial loss should be close to -tf.math.log(1/10) ~= 2.3.

```
loss_fn(y_train[:1], predictions).numpy()
```

Which yields: 1.5666813

We then compile our model:

```
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])
```
And fit it...
```
model.fit(x_train, y_train, epochs=5)
```
Which yields:

```
Epoch 1/5
   1/1875 [..............................] - ETA: 7:28 - loss: 2.4775 - accuracy: 0.093  54/1875 [..............................] - ETA: 1s - loss: 1.7482 - accuracy: 0.4337 1875/1875 [==============================] - 2s 859us/step - loss: 0.4828 - accuracy: 0.8602
Epoch 2/5
1875/1875 [==============================] - 2s 874us/step - loss: 0.1535 - accuracy: 0.9557
Epoch 3/5
1875/1875 [==============================] - 2s 863us/step - loss: 0.1106 - accuracy: 0.9673
Epoch 4/5
1875/1875 [==============================] - 2s 866us/step - loss: 0.0853 - accuracy: 0.9741
Epoch 5/5
1875/1875 [==============================] - 2s 867us/step - loss: 0.0758 - accuracy: 0.9753
```
We then check the model validity:

```
model.evaluate(x_test,  y_test, verbose=2)
```
Which yields:

```
313/313 - 0s - loss: 0.0748 - accuracy: 0.9758

[0.07476752996444702, 0.9757999777793884]
```
So we now have an image classifier trained to 98% accuracy on this dataset.

If you want your model to return a probability, you can wrap the trained model, and attach the softmax to it:

```
probability_model = tf.keras.Sequential([
  model,
  tf.keras.layers.Softmax()
])
```
We can print that out with:

```
probability_model(x_test[:5])

<tf.Tensor: shape=(5, 10), dtype=float32, numpy=
array([[1.12014416e-08, 2.06343298e-09, 9.44028670e-06, 3.84519371e-05,
        5.40175960e-11, 1.57378608e-08, 4.46096556e-13, 9.99943733e-01,
        1.66466648e-07, 8.21956746e-06],
       [3.13455786e-07, 3.47839923e-05, 9.99942183e-01, 6.49015919e-06,
        4.46333319e-13, 7.55193980e-07, 9.94391257e-07, 3.53041001e-13,
        1.45972899e-05, 6.18286715e-12],
       [1.19702094e-07, 9.97879744e-01, 1.40806398e-04, 1.30377666e-05,
        1.08557593e-04, 1.13779515e-06, 9.81380726e-05, 9.66617488e-04,
        7.91212486e-04, 6.97812425e-07],
       [9.99660134e-01, 3.72114528e-07, 4.11666661e-05, 3.10024620e-06,
        4.62686138e-08, 1.64676476e-05, 2.76185077e-04, 4.67025558e-07,
        1.02093294e-08, 2.19160893e-06],
       [8.10114898e-07, 6.72608774e-11, 9.34032357e-07, 1.13792233e-07,
        9.98626709e-01, 7.41321386e-08, 1.15308185e-05, 7.61476986e-05,
        2.74434058e-07, 1.28322886e-03]], dtype=float32)>

```
## Running the Above in a Jupyter Notebook

Install Jupyter Notebooks version:

```
sudo docker pull tensorflow/tensorflow:latest-gpu-jupyter
```
Run Notebook on port 8888:8888, then open browser:
```
sudo docker run -it --rm --gpus all -v $(realpath ~/notebooks):/tf/notebooks -p 8888:8888 tensorflow/tensorflow:latest-jupyter
```
After running this, we get a success message, saying:

To access the notebook, open this file in a browser:
        file:///root/.local/share/jupyter/runtime/nbserver-1-open.html
    Or copy and paste one of these URLs:

(two URLs with tokens are displayed.)

I copied one of the URLs (physical copy and paste) into a browser and it ran the Juputer Notebook.

Interestingly, within the Jupyter notebook there were ready mde tutorials to run.

At this time, the system memory was running at about 4GB.  Running through the notebooks, system memory got to be no higher than around 5.5GB.

## Conclusion

So in conclusion, we were able to successfully set up TensorFlow using GPUs on a local machine with a GPU, through docker, and run a basic example.

While the CPU version output errors, the GPU version worked seamlessly and all of the commands shown worked more or less instantaneously.

At the end of this exercise, system memory was cached at about 11GB, with 6GB in use out of a total 16GB. So, this shows that having more memory will indeed be helpful for this application, particularly as applications get more advanced and we run more over time.

### Comparison to Free Online Tools

Since I already **have** this equipment, it's essentially something that I can use at will. Other products and services that I may choose to use at will which might be free might be superior, so it would be less logical for me to use my own resource if that were the case.

The usage of Tensorflow in combination with a Jupyter notebook on a local machine is highly useful for speed. There may not be a huge advantage over using for example a Google CoLab notebook, but the ability to run a simple Dockerized Tensorflow model on a local machine before deploying to the cloud may have some benefits from the standpoint of being able to quickly iterate and train locally before deploying remotely, much like in web and webapp development. CoLab kind of ties in almost unavoidably with Google Drive, which means you have to troubleshoot all sorts of ways of working with the Google Drive API to get access to data. Contrast this with just using our own Docker container, where we are essentially using normal containerized deployment methods, and not needing to re-do the code to work directly with a linux/container environment rather than a Google Drive environment.

#### Qualitative Comparison to CoLab

Iterative design speed may come into particular consideration if integrating some kind of machine learning microservice along with a web app in a multi-container dockerized format of some kind.

Of course the limitations of this system include:

* The fact that the Jupyter notebook doesn't automatically save as a CoLab notebook does, unless we set it up that way.
* We are only limited to the number of GPUs that we have on our system, whereas with Cloud we could access as many GPUs as we may need to quickly train a model.

#### Speed Comparison

##### Kaggle Kernals

[This article](https://medium.com/@saj1919/is-free-kaggle-k80-gpu-is-better-than-gtx-1070-maxq-8f9cecc4dc1b) does a nice detailed comparison of various NVIDIA families of hardware, including the Tesla family of products and including the GeForce GTX 1070.

Using Keras examples, the author compared a Kaggle K80 GPU to the GeForce GTX 1700 and found that the GTX17000 had an almost 2 to 3 times speedup.

##### Google CoLab

Google CoLab is also said to use the same Tesla K80 GPU as the Kaggle K80.

That being said, these cloud services are always going to have updates, so it's going to be important to know how to double check them.

For Google Colab, you can go under, "Runtime" and then click, "Change Runtime type" and select GPU.

We can then enter in the command: `!nvidia-smi` to look at which GPU is being used.

```
Fri Feb 26 13:12:44 2021       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 460.39       Driver Version: 460.32.03    CUDA Version: 11.2     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  Tesla T4            Off  | 00000000:00:04.0 Off |                    0 |
| N/A   53C    P8     9W /  70W |      0MiB / 15109MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+

```

As we can see above, the default is currently at the date of authoring, a Tesla T4.

This article lists out different benchmarks [comparing the Tesla T4 to the GTX 1700](https://askgeek.io/en/gpus/vs/NVIDIA_Tesla-T4-vs-NVIDIA_GeForce-GTX-1070-Desktop).

Which shows that the T4 is faster by one metric, but the GTX1700 is fater by most metrics.
