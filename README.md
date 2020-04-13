# Single Image Super Resolution Using Weighted ResNet.
![prediction](https://user-images.githubusercontent.com/46989642/79095895-a41aaf00-7d96-11ea-859d-0a05cdd4c86e.png)

# Goal
The goal of this project is to convert low-resolution image files to high-resolution image files using deep learning techniques.

The super-resolution algorithm is expected to have a myriad of applications in various fields of industry. One example would be CCTV for crime prevention purposes. If we use low-resolution CCTV, the image of people's faces can't be distinguishable in some cases. This image is unhelpful. However, the situation will be different if low-resolution images are converted into high-resolution images using the software implemented in this project.

# What is 'Weighted ResNet'?

The model was inspired by the following two papers.
-
Deep Residual Learning for Image Recognition(Kaiming He Xiangyu Zhang Shaoqing Ren Jian Sun)

Densely Connected Convolutional Networks(Gao Huang, Zhuang Liu, Laurens van der Maaten)

----------------------------------------------------------------------------------------------------------------
![W_ResNet](https://user-images.githubusercontent.com/46989642/79096700-dc22f180-7d98-11ea-9e04-db9df4fbb299.png)

In Weighted ResNet, each layer is implemented as a residual layer to prevent over-fitting. In addition, the output of each residual block was added to the final layer by passing a specific weight to the final output. By doing so, it is expected that each feature of the image will be preserved and transmitted without being diluted, so that a more 'human-tic' image can be realized. Also, it is much simpler to implement than DenslyConnectedNet, making it easy for real-time image processing, and has the advantage of being able to stack deeper layers because overfitting does not occur easily.

# Experiement
A neuron network consisting of a total of 17 convolution layers was used to implement the model(Weighted ResNet).
Relu function was used as an activation function for each layer, and MSE was used as a loss function. Also, the last output adopts a sub-pixel model for up-scaling of resolution.

# Training images
![celeba](https://user-images.githubusercontent.com/46989642/79109246-f10c7e80-7db2-11ea-954b-afcd8582f42f.png)
http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
I obtain training images from celebA dataset.(human face image dataset)
I reduceed size of all training images to 44*44*3 for training.

# Output images
Size of output image(prediction) is 196*196*3.
The images are up-scaled by sub-pixel technique.



# Result
![result1](https://user-images.githubusercontent.com/46989642/79097973-f4e0d680-7d9b-11ea-815c-0965cf5c373a.png)
-----------------------------------------------------------------------------------------------------------------
![result2](https://user-images.githubusercontent.com/46989642/79098019-09bd6a00-7d9c-11ea-9289-7268034a3975.png)
-----------------------------------------------------------------------------------------------------------------
![result3](https://user-images.githubusercontent.com/46989642/79098190-628d0280-7d9c-11ea-97a2-e5c0d629a904.png)
-----------------------------------------------------------------------------------------------------------------
psnr (EDSR) = 34.994

psnr (WRESNET) = 32.125

If you compare the EDSR model, which is one of the best models, and WRESNET (Weighted Renset), you can see an interesting fact. PNSR has higher EDSR but WRESNET made much more 'human-tic' predictions. So I think EDSR is better for overall line, colour processing such as scenery and background but WRESNET is better for catching a mysterious 'features' in a human face.
