# Research CNN with Keras

## Data Preprocessing and Augmentation

It’s always good to normalize data. Our Datasets will have data in each pixel in between 0–255 so now we scale it to 0–1 using below code.

```python
from keras.preprocessing.image import ImageDataGenerator

rescaled = ImageDataGenerator(rescale=1. / 255.).flow_from_directory(train_path)
```

### Augmentation for unbalanced image data set

Balancing an imbalanced dataset with keras image generator

> This would not be a standard approach to deal with unbalanced data. Nor do I think it would be really justified - you  would be significantly changing the distributions of your classes, where the smaller class is now much less variable. The larger class would have rich variation, the smaller would be many similar images with small affine transforms. They would live on a much smaller region in image space than the majority class.
>
> The more standard approaches would be:
>
>1. The class_weights argument in model.fit, which you can use to make the model learn more from the minority class.
>
>2. reducing the size of the majority class.
>
>3. accepting the imbalance. Deep learning can cope with this, it just needs lots more data (the solution to everything, really).
>
>The first two options are really kind of hacks, which may harm your ability to cope with real world (imbalanced) data. Neither really solves the problem of low variability, which is inherent in having too little data. If application to a real world dataset after model training isn't a concern and you just want good results on the data you have, then these options are fine (and much easier than making generators for a single class).
>
>The third option is the right way to go if you have enough data (as an example, the recent paper from Google about detecting diabetic retinopathy achieved high accuracy in a dataset where positive cases were between 10% and 30%).
If you truly want to generate a variety of augmented images for one class over another, it would probably be easiest to do it in pre-processing. Take the images of the minority class and generate some augmented versions, and just call it all part of your data. Like I say, this is all pretty hacky.

## Bottleneck feature extraction

### Problem

Usage of generator. Problem is that extracting bottleneck features runs "out of memory".

## Reference

[Image Data Preprocessing and Augementation](https://software.intel.com/en-us/articles/hands-on-ai-part-14-image-data-preprocessing-and-augmentation)

[Tutorial on using Keras flow_from_directory and generators](https://medium.com/@vijayabhaskar96/tutorial-image-classification-with-keras-flow-from-directory-and-generators-95f75ebe5720)

[Sub directories for keras.flow_from_directory
](https://www.kaggle.com/ericbenhamou/sub-directories-for-keras-flow-from-directory)

[[Keras] A thing you should know about Keras if you plan to train a deep learning model on a large dataset](https://towardsdatascience.com/keras-a-thing-you-should-know-about-keras-if-you-plan-to-train-a-deep-learning-model-on-a-large-fdd63ce66bd2)

[A Keras multithreaded DataFrame generator for millions of image files](https://techblog.appnexus.com/a-keras-multithreaded-dataframe-generator-for-millions-of-image-files-84d3027f6f43)

[Writing Custom Keras Generators](https://medium.com/@ensembledme/writing-custom-keras-generators-fe815d992c5a)

[Using Keras ImageDataGenerator in a regression model](https://stackoverflow.com/questions/41749398/using-keras-imagedatagenerator-in-a-regression-model?noredirect=1#comment70692649_41749398)

[A detailed example of how to use data generators with Keras](https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly)

[Tutorial on Keras ImageDataGenerator with flow_from_dataframe](https://medium.com/@vijayabhaskar96/tutorial-on-keras-imagedatagenerator-with-flow-from-dataframe-8bd5776e45c1)

[balancing an imbalanced dataset with keras image generator](https://stackoverflow.com/questions/41648129/balancing-an-imbalanced-dataset-with-keras-image-generator)

[How can you train convolutional neural networks on highly unbalanced datasets?(https://www.quora.com/How-can-you-train-convolutional-neural-networks-on-highly-unbalanced-datasets)

[Using Transfer Learning and Bottlenecking to Capitalize on State of the Art DNNs](https://medium.com/@galen.ballew/transferlearning-b65772083b47)