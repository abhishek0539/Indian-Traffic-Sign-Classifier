# Indian Traffic Sign Classification

This project aims to classify different Indian traffic signs using a CNN model trained on the **Indian Traffic Sign Dataset**.The dataset used is downloaded via the Kaggle API it includes various traffic signs used in India, and the model predicts the class of a given sign from an image. It uses Convolutional Neural Network (CNN) implemented with TensorFlow and Keras. 


## About Dataset
The dataset consists of two folders:
- `Images`: Contains images of various traffic signs.
- `Csv`: Contains CSV file to map the class indices to their respective traffic sign names


## Model Architecture
The model you've implemented uses 2D Convolutional Neural Networks (CNNs), as it is designed for image processing, which involves 2D convolutions due to the spatial nature (height and width) of the images.


**Breakdown of the Layers:**
**Conv2D Layer 1:**
Kernel size: 5x5 ;
Depth (filters): 32 ;
Followed by Max Pooling and Dropout

**Conv2D Layer 2:**
Kernel size: 5x5 ;
Depth (filters): 64 ;
Followed by Max Pooling and Dropout

**Conv2D Layer 3:**
Kernel size: 5x5 ;
Depth (filters): 128 ;
Followed by Max Pooling and Dropout

**Fully Connected Layers:**
Flatten Layer: Flattens the output from the convolutional layers to a 1D vector.<br/>
Fully Connected (Dense) Layer: 512 units ; Dropout applied

**Output Layer:**
Number of units: 59 (one per class)


## Training and Results
The model was trained for 50 epochs, with the following key metrics observed during training:
- **Accuracy**: Around 89.21%
- **Validation Accuracy**: 87.19%
- **Test Accuracy**: 87.45%


### Training and Validation Accuracy:
![download](https://github.com/user-attachments/assets/2a171ed3-f137-4828-9397-99e93323d74a)


### Training and Validation Loss:)
![download copy](https://github.com/user-attachments/assets/6a88a9bd-d583-4d16-beb1-eac291e1ac74)


### Example of Predicted Traffic Sign:
<img width="546" alt="Screenshot 2024-09-16 at 6 05 33 PM" src="https://github.com/user-attachments/assets/3d46d655-cd00-46f3-ad52-e025164109ff">
