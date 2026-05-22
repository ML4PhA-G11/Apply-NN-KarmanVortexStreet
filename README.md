# Karman Vortex Street
This repo is a branched off version of the original repo, focusing on training a NN, letting it predict values from a Karm Vortex Street, evaluating it and producing an animation. To reproduce the results, use the following steps:
1. train the network by running ```python run-all-tensorflow.py``` this should result in a keras model being saved.
2. run the Karman simulator by running ```python scripts-simulator-lbm/lbm_karman-ng.py --save-every 2000```. This should save data in an output folder. The most important data are the fpost and fpre data files as these are required for step 3.
3. Finally run  ```python apply-nn.py --animate --anim-steps 30000 --update-steps 50``` This will result in both an animation and some plots and csv. The animation is meant to show how the NN predicts the Karman Vortex Street situation and will only be visible after the code has finished running. Its location is in eval_results/nn_velocity_field.gif. 
The code takes a while to run. To shorten it, you can lower --anim-steps so that the animation includes less timesteps. Or you can increase --update-steps so that there will be 'less frames' in the gif. The code can be used without creating an animation by simply running: ```python apply-nn.py```. This will make it so that only the csv and plots are made.

