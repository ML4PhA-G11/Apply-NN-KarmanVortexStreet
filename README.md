# Karman Vortex Street
This repo is a branched off version of the original repo, focusing on training a NN, letting it predict values from a Karm Vortex Street, evaluating it and producing an animation. To reproduce the results, use the following steps:
1. train the network by running ```python run-all-tensorflow.py``` this should result in a keras model being saved.
2. run the Karman simulator by running ```python scripts-simulator-lbm/lbm_karman-ng.py --save-every 2000```. This should save data in an output folder. The most important data are the fpost and fpre data files as these are required for step 3.
3. Finally run  ```python apply-nn.py --animate --anim-steps 30000 --update-steps 50``` This will result in both an animation and some plots and csv. The animation is meant to show how the NN predicts the Karman Vortex Street situation and will only be visible after the code has finished running. Its location is in eval_results/nn_velocity_field.gif. 
The code takes a while to run. To shorten it, you can lower --anim-steps so that the animation includes less timesteps. Or you can increase --update-steps so that there will be 'less frames' in the gif. The code can be used without creating an animation by simply running: ```python apply-nn.py```. This will make it so that only the csv and plots are made.

# Towards learning Lattice Boltzmann collision operators

In this repository we provide a set of jupyter notebooks which allows to reproduce the results presented in [arxiv.2212.06124](https://arxiv.org/abs/2212.06124):

1.  [create_trainset.ipynb](create_trainset.ipynb) allows to generate a training dataset (see Algorithm 1 in [arxiv.2212.06124](https://arxiv.org/abs/2212.06124) ). The training set consists of pre and post collision distribution functions generated using the LBGK collisional operator.
2.  [train_network.ipynb](train_network.ipynb) makes use of the training data to train a neural network mapping 9 pre-collisional distribution functions (input) to 9 post-collisional distribution functions (output). Conservation laws and symmetries are embedded in the network architecture ( 
using the dataset generated  
3.  [lbml_simulation.ipynb](lbml_simulation.ipynb) Implements a LBM simulation of a Taylor-Green vortex flow, where the BGK operator is replaced by a Neural Network

![image](https://github.com/agabbana/learning_lbm_collision_operator/assets/90458863/7c9b7e56-819b-4bf6-adb2-1e32184d2711)


### How to cite this work

```
@article{toward-learning-lattice-boltzmann-collision-operators,
  title={Towards learning Lattice Boltzmann collision operators},
  author={Corbetta, Alessandro and Gabbana, Alessandro and Gyrya, Vitaliy and Livescu, Daniel and Prins, Joost and Toschi, Federico},
  journal={The European Physical Journal E},
  volume={46},
  number={3},
  pages={10},
  year={2023},
  publisher={Springer},
  doi = {10.1140/epje/s10189-023-00267-w},
}
```
