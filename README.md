# Dog_TD3
 Twin Delayed DDPG train a 4 legged dog to walk/run

# Dependencies
 - Python 3.8
 	- Tensorflow 2.0
 	- numpy
 	- pynput
 	- gym
 - Vortex Studio 2020b (2020.5.0.133) 

# How to use
 1. Open Vortex Studio 2020b
 2. Load Vortex Resources/Setup.vxc
 3. Access the properties of the Setup (click on it on the Explorer)
 4. In "Parameters > Python 3 > Interpreter Directory", enter the path to your Python 3.8 Interpreter.
 5. Save and Close
 6. Using THE SAME INTERPRETER, run :
	- train_td3.py to train the model.
	- run_td3.py to run the model after training.