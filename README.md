# Implement-Neural-Network-with-PyTorch
## PyTorch 
PyTorch is an open source machine learning (ML) framework based on the Python programming language and the Torch library. Torch is an open source ML library used for creating deep neural networks and is written in the Lua scripting language. It's one of the preferred platforms for deep learning research. The framework is built to speed up the process between research prototyping and deployment. <br/>

The PyTorch framework supports over 200 different mathematical operations. PyTorch's popularity continues to rise, as it simplifies the creation of artificial neural network models. PyTorch is mainly used by data scientists for research and artificial intelligence (AI) applications. <br/>

PyTorch provides the following key features:

-Tensor computation. Similar to NumPy array -- an open source library of Python that adds support for large, multidimensional arrays -- tensors are generic n-dimensional arrays used for arbitrary numeric computation and are accelerated by graphics processing units. These multidimensional structures can be operated on and manipulated with application program interfaces (APIs). <br/>
-TorchScript. This is the production environment of PyTorch that enables users to seamlessly transition between modes. TorchScript optimizes functionality, speed, ease of use and flexibility. <br/>
-Dynamic graph computation. This feature lets users change network behavior on the fly, rather than waiting for all the code to be executed. <br/>
-Automatic differentiation. This technique is used for creating and training neural networks. It numerically computes the derivative of a function by making backward passes in neural networks. <br/>
## PROBLEM 1: Pattern Classification with Multi Layer Perceptron
A multi-layered network of 4 classes, each of which can separate black and white patterns with a resolution of 10x5, will be designed.What is expected from the designed network is to be able to recognize clean and broken patterns and specify which class they belong to. <br/>
While creating our dataset, we created 4 10x5 matrices that look like 0,2,4,7 as digital numbers. <br/>
![image](https://user-images.githubusercontent.com/78887209/157834792-c2b9fe73-aef7-4d8d-8f9e-bb7f072713b8.png) <br/>
We defined two functions to add noise and provide distortion that makes a black point white and a white point black. For each data, we produced 6 noise added data and 6 distorted data.In the first place, we made a 2% deterioration in the data.After this stage, we have 13 data for each class, 6 of which are corrupted, 6 are changed and 1 is the data itself.
Below is an example with one gray scale noise added and one negative distortion: <br/>
![image](https://user-images.githubusercontent.com/78887209/215317882-6b666e7d-df94-4e9e-9d00-78971df29527.png) ![image](https://user-images.githubusercontent.com/78887209/215317903-11dd1058-f5b4-4fd3-828f-0dd7adaae2c6.png) <br/>
In the training set, 9 data were taken from each class as input to the model. The remaining 16 data were used in the test set to measure the performance of the model. <br/>
### Structure and Characteristics of Neural Network <br/>
The structure of the network is designed as follows:
![image](https://user-images.githubusercontent.com/78887209/215318248-223ff717-0c9d-4ae6-90f4-57c0b2055883.png) <br/>
Since the matrices representing the data set consist of numbers between 0 and 1, including 0 and 1, the "sigmoid" function, which produces the activation function between 0 and 1, was chosen. The cross entropy loss was used as the error function, and if the cross entropy loss was less than one in ten thousand, the training stop condition was set to stop the training. Learning rate was determined as 0.01 and momentum term coefficient alpha was determined as 0.9. For additional information on the momentum term, see the link in the 'further readings' section. Stochastic Gradient Descent was used as the back propagation algorithm. In each epoch, the weights are updated with the help of the stochastic gradient descent algorithm. For detailed information about the gradient descent algorithm, a related article is given in the 'further readings' section. After the training process, the model achieved 100% accuracy in classifying the data in the test set.
### Further Readings <br/>
https://medium.com/edureka/pytorch-tutorial-9971d66f6893 <br/>
https://towardsdatascience.com/gradient-descent-algorithm-a-deep-dive-cf04e8115f21 <br/>
https://machinelearningmastery.com/gradient-descent-with-momentum-from-scratch/#:~:text=Momentum%20is%20an%20extension%20to,spots%20of%20the%20search%20space. <br/>
## PROBLEM 2: IRIS Classification with Multi Layer Perceptron
In Problem 2, the IRIS dataset classification problem, which is a frequently used dataset in machine learning problems, is discussed. Edgar Anderson collected data in 1935 to distinguish three genera of iris flowers growing on the Gaspé peninsula, and this dataset was first used by Sir Ronald Aylmer Fisher (1936) to test the success of the "Linear Discriminant Model" he developed. Later, this dataset has been frequently used in reporting the performance of classification techniques. To access the dataset and see the articles using the dataset, you can check this link: 
http://archive.ics.uci.edu/ml/datasets/Iris <br/>
To solve the problem, the deep learning network shown in the picture below is designed:
![image](https://user-images.githubusercontent.com/78887209/215330335-280336dc-690a-445a-9670-763f9f3b71ba.png) <br/>
The learning rate value was determined as 0.01, the coefficient of momentum term alpha was determined as 0.9. The dimensions of the initial weight matrices were determined randomly, in accordance with the hidden layer and input dimensions, and the data were between 0 and 1. First of all, vgk1 is obtained by passing the initial weights through the neurons, and the first hidden layer output is obtained by applying the activation function. After adding bayes to the first hidden layer output, vgk2 is obtained by passing it through the second hidden layer. The second hidden layer output is obtained by applying the activation function to vgk2. Bayesian term is added to the second hidden layer output and passed through the output layer and vo's are obtained. Again, the outputs are obtained by applying the activation function to the vos. The Adam optimizer algorithm is used in the back propagation algorithm of the multilayer perceptron. The article containing detailed information about the Adam algorithm is given in the further readings section. During the training process, the error function was chosen as the cross entropy loss. The stopping condition was also determined as stop training when the cross entropy loss is less than 0.0001. After the training process, the correct estimation rate in the test set was 100%. 
## PROBLEM 3: System Recognition with Multi Layer Perceptron
System recognition is the finding of a dynamic system model with input-output measurements taken from the real system. Its purpose is to establish a reliable mathematical model that can be used in further studies on the system from the data input-output relationship to a particular system. A dataset specific to the system recognition problem can be found by using the link below: <br/>
 https://archive-beta.ics.uci.edu/ <br/> 
 As an example dataset, the dataset containing the "chickenpox" cases that took place in Budapest in a certain time period was selected from the relevant link. 
If a system recognition problem is to be solved using a multi-layered network, first of all, it should be decided how many data in the problem are affected by the data in the past. This decision has been chosen as 5 specifically for this problem. <br/> 
Inputs	Y[i-5]	Y[i-4]	Y[i-3] 	Y[i-2]	Y[i-1]	N[i] <br/>
Output 	Y[i] <br/> 
N[i] is the noise term that will be added to the data at each step. The noise term was randomly chosen between 0.01 and 0.04 at each step. The article containing detailed information about the noise term is presented in the 'further readings' section. Since the classification problems were solved in the previous two problems, various changes should be made in the network while solving the system recognition problem with the multi-layer perceptron. These changes are seen below: <br/>
- When solving the classification problem with the multi-layer network, we mixed the data before presenting it to the network. However, this should not be done when solving the system recognition problem, because dynamic systems are time dependent, and mixing the data leads to the distortion of the time flow in the data. This makes it impossible for the model to find the time-dependent relationship between the data. <br/> 
- While there are data known to belong to certain classes in classification problems, there is no such situation in system recognition problems. For this reason, since the relationship between the data is tried to be found depending on time, more data is needed in system recognition problems for an approximate success rate. <br/> 
- In the first two problems, the data was already scaled. But the dataset chosen for this problem needed scaling. Therefore, before the data is presented to the network, it is scaled between 0.1-0.9, unlike the previous problems. <br/> 
The network is designed as shown in the image below: <br/>
![image](https://user-images.githubusercontent.com/78887209/215333791-0df5db40-29a4-4a7c-a191-72c27cfac6b5.png) <br/>
500 pieces of data are reserved for use in the problem, and 400 of these data are reserved for training the network and the remaining 100 for testing the network. The error function was chosen as the mean squared error, and the criterion for stopping the training was set to stop training when the cross entropy loss is less than 0.001. The learning rate was determined as 0.01 and the momentum term coefficient as 0.9. Initial weights were randomly determined between -0.5 and 0.5. As a result of the examinations made on the test set, an accuracy of 100% was obtained.
### Further Readings <br/>
https://www.sciencedirect.com/topics/engineering/pattern-recognition-systems <br/>
https://timeseriesreasoning.com/contents/white-noise-model/ <br/>
https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/ <br/>
