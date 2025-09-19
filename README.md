# HunarIntern_task04
Cat vs Dog Image Classification 

This project demonstrates the use of Convolutional Neural Networks (CNNs) for classifying images of cats and dogs. The dataset used is the Cats vs Dogs dataset provided by TensorFlow Datasets (TFDS), which eliminates the need for manual downloads and folder organization. The dataset is automatically split into training and validation sets, with 80% used for training and 20% for validation.

The images are preprocessed by resizing them to a fixed dimension of 128×128 pixels and normalizing pixel values to the range [0,1]. This ensures consistency across all inputs and speeds up training. The dataset is then batched, shuffled, and prefetched using the TensorFlow data pipeline, allowing for efficient GPU utilization during model training.

The model is built using a sequential CNN architecture with three convolutional and max-pooling layers. These layers extract hierarchical features such as edges, textures, and shapes from the images. The extracted features are flattened and passed through a fully connected dense layer with ReLU activation, followed by a final output layer with a sigmoid activation function for binary classification.

The model is compiled with the Adam optimizer and binary crossentropy loss, which are well-suited for binary image classification problems. Training is performed for three epochs to keep it lightweight and fast while still demonstrating the effectiveness of CNNs. During training, accuracy is tracked for both training and validation sets, and the results are visualized with a line plot showing accuracy trends.

Finally, the trained model is tested on a sample image from the validation dataset. The model outputs a probability, which is interpreted as either “Cat” or “Dog.” The image is then displayed alongside the prediction result, making the output intuitive and easy to understand.

This project highlights the practical workflow of building, training, and testing a deep learning model using TensorFlow and TensorFlow Datasets. It also demonstrates the importance of preprocessing, efficient data pipelines, and visualization of model performance.

In the future, the model can be improved by training for more epochs, using data augmentation techniques to improve generalization, or experimenting with deeper architectures such as VGG16, ResNet, or transfer learning with pretrained models. Such enhancements could significantly improve classification accuracy on more complex image datasets.
