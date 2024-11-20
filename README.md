# MP<sup>2</sup>


## Updates
11/20/2024: Tensorflow version uploaded


## ABSTRACT
As the size of deep neural network (DNN) models and datasets increases, distributed training becomes popular to
 reduce the training time. However, a severe communication bottleneck in distributed training limits its scal
ability. Many methods aim to address this communication bottleneck by reducing communication traffic, such as
 gradient sparsification and quantization. However, these methods either are at the expense of losing model
 accuracy or introducing lots of computing overhead. We have observed that the data distribution between layers
 of neural network models is similar. Thus, we propose a model parameter prediction method (MP<sup>2</sup>) to accelerate
 distributed DNN training under parameter server (PS) framework, where workers push only a subset of model
 parameters to the PS, and residual model parameters are locally predicted by an already-trained deep neural
 network model on the PS. We address several key challenges in this approach. First, we build a hierarchical
 parameters dataset by randomly sampling a subset of model from normal distributed trainings. Second, we
 design a neural network model with the structure of “convolution + channel attention + Max pooling” for
 predicting model parameters by using a prediction result-based evaluation method. For VGGNet, ResNet, and
 AlexNet models on CIFAR10 and CIFAR100 datasets, compared with Baseline, Top-k, deep gradient compression
 (DGC), and weight nowcaster network (WNN),MP<sup>2</sup> can reduce traffic by up to 88.98%; and accelerates the
 training by up to 47.32% while not losing the model accuracy. MP<sup>2</sup>has shown good generalization.


## The system framework of MP<sup>2</sup>
![image](https://github.com/user-attachments/assets/2fec5456-cab2-42c8-afc9-cfcc1e3b08a8)


## The overall structure of prediction model
![image](https://github.com/user-attachments/assets/590c7a65-9edb-40a1-933c-4f7362ad7b52)



## Dependency
| Library | Known Working
| tensorflow | 1.14.0


## Experiments
Please first download the required datasets: mnist_data, cifar-10-python, and cifar-100-python.
If needed, train the prediction models for the datasets using the file mp2-train-stage.py.

Training process:
Run mp2-ps.py and mp2-worker.py.

## Contact
If you have any questions, please contact us via email: [2112230039@e.gzhu.edu.cn]

Thank you for your interest!
