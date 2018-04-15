# Production Ready Docker Container for Tensorflow Serving

Note: The latest image tagged with :optimized is a highly optimized version consuming only 300 MBs and optimized to use AVX and SSE instruction sets. 

TensorFlow Serving is a flexible, high-performance serving system for machine learning models, designed for production environments. It handles the complete life-cycle of ML models in production. This docker image should significantly reduce the time required to take Tensorflow models from research bench to production.

![alt text](https://tensorflow.github.io/serving/images/tf_diagram.svg "Tesorflow Serving Lifecycle")

**New to Docker?** 
Check out this video to get a quick overview: https://www.youtube.com/watch?v=Vyp5_F42NGs 

Created by Amiya Patanaik
last updated: 4-Aug-2017

To load the image into Docker:
```sh
$ docker pull amiyapatanaik/tfserve:optimized
```
to check if the image loaded successfully:
```sh
$ docker images
REPOSITORY             TAG                 IMAGE ID            CREATED             SIZE
amiyapatanaik/tfserve   optimized           5c1882a5e601        19 minutes ago      305MB

**Run an instance of the image (default settings)**: 
Spin up a new container:
```sh
using -d switch to run it as a daemon
$ docker run --name tfserve_deploy -p 9000:9000 -d amiyapatanaik/tfserve:optimized
To check if the container is running
$ docker ps
```
Tensorflow Serving is now running on port 9000 and is serving a MNIST model that can classify handwritten digits. To know more about MNIST: https://en.wikipedia.org/wiki/MNIST_database

**Serving your own models**
The image has a models folder from which both the model configurations and actual models will be loaded. The folder structure is:
```
/ (root)
.
|-- models (dir)
|   |-- mnist (model name, dir)
|   |   `-- 1 (version number, dir)
|   |       |-- variables 
|   |       |   |-- variables.data-00000-of-00001
|   |       |   `-- variables.index
|   |       `-- saved_model.pb
|   `-- config.ini (model configuration file)
```
The model config file has the necessary settings:
```
model_config_list: {
config: {
name: "mnist",
base_path: "/models/mnist",
model_platform: "tensorflow"
}
}
```
You may add multiple models to the config list.
To serve your own models create a models folder locally with all the necessary models and configuration file in it. Then mount the folder to the Docker container and it will load your own models instead of running the default ones. 
```sh
first stop and remove old container
$ docker stop tfserve_deploy:optimized
$ docker rm tfserve_deploy:optimized
spin up a new container to serve your custom models
$ docker run --name tfserve_deploy -p 9000:9000 -d -v /pathtolocal/models:/models:ro amiyapatanaik/tfserve:optimized
```

***Train and Save TF model using Keras***
Sample code to train a MNIST model is included which is directly taken from Keras repo. You must have keras and tensorflow installed on your system. 
```sh
$ python train_mnist.py
```
The model runs for 12 epochs and attains an accuracy of 99.25%. The trained model is saved in mnist.hdf5. 

***Convert Keras model to Tensorflow Serving***
This model must be converted to tensorflow compatible format. Use the export_model.py script:
```sh
-n is name of the model, -p is path to the keras model and -v is version number
$ python export_model.py -n mnist -p mnist.hdf5 -v 1
```

***Sample python client***
Once the server is running, it is possible to query it using the gRPC protocol. A complex example comes from official tf git repo, included here for reference:
```sh
$ python mnist_client.py --num_tests=1000 --server=localhost:9000
```

A much simpler code is provided here. Run it as:

```sh
$ python sample_client.py
```
