# Benedict-College-SURI-Program

**ABOUT THIS CODE**
My Topics include the Deep Q Learning algorithims and Carla's vehicle simulator provided by sentdex's Reinforcement Learning videos.
The coding algorithims were modified by me for practical use and full operability on Python's older IDE compiler v. 3.5.1. 

Link to sentdex's RL website: https://pythonprogramming.net/own-environment-q-learning-reinforcement-learning-python-tutorial/

**REVISIONS**
//6.4.20 Added code for Tensorflow graphical data for CarlaVSTM.py and Blobworld.py //

#Tensorboard Documentation
 
    mnist = tf.keras.datasets.mnist
	
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
	
	    model = tf.keras.models.Sequential([
	  tf.keras.layers.Flatten(input_shape=(28, 28)),
	  tf.keras.layers.Dense(128, activation='relu'),
	  tf.keras.layers.Dropout(0.2),
	  tf.keras.layers.Dense(10)
	])

  	predictions = model(x_train[:1]).numpy()
	    predictions
	
	    tf.nn.softmax(predictions).numpy()
	    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)	    loss_fn(y_train[:1], predictions).numpy()
	
	    model.compile(optimizer='adam',	              
      loss=loss_fn,	              
      metrics=['accuracy'])
	
	    model.fit(x_train, y_train, epochs=5)
      
 //6.1.20 Code for logging Blobworld scalars into Tensorboard //
 
	# Custom method for saving own metrics
	    # Creates writer, writes custom metrics and closes writer
	    def update_stats(self, **stats):
	      def _write_logs(self, logs, index):
	
	        with self.writer.as_default():
	
	            for name, value in logs.items():
	
    tf.summary.scalar(name, value, step=index)
	
	                self.step += 1
	
	                self.writer.flush()
                  
**NOTE**

// I did not have the necessary rendering software to compile 3D Graphs at the time. //
// System required me to download latest NVIDIA CUDA 3D Tensor rendering architecture was exclusive only to NVIDIA Graphics Cards //


Owned by BigPete7.

