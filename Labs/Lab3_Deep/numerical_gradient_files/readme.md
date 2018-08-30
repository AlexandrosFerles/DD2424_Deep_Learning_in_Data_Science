#### Numerical Gradients for Validation

This folder contains the numerically computed gradients for 2, 3 and 4 layers.   
You can use this files for validation of this work or for validation of your own work.  
The settings on which the gradients were trained are the following:

'2_layers.npz':     X_training_1[:, :4] and W1.shape=(50, 3072), W2.shape=(10, 50)  
'3_layers.npz':     X_training_1[:, :4] and W1.shape=(50, 3072), W2.shape=(20, 50), W3.shape=(10, 20)  
'2_layers_num.npz': X_training_1[:, :2] and W1.shape=(50, 3072), W2.shape=(30, 50), W3.shape=(10, 30)  
'2_layers.npz':     X_training_1[:, :4] and W1.shape=(50, 3072), W2.shape=(20, 50), W3.shape=(15, 20) and W4.shape=(10, 15)
