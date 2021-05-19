# automated-glasses-recomendation-system
# download face shape dataset

https://www.kaggle.com/niten19/face-shape-dataset

# Description For CNN algorithm 

In this report, four distinct challenging scopes are addressed under the supervised machine learning paradigm. They comprise binary classification tasks for gender (A1) and smile detection (A2) along with multi-categorical classification tasks concerning eye-colour (B2) and face-shape (B1) recognition. Most notably, several methodologies are proposed to deal with these duties

|                                       |                   Test 2                    |                                                Test 4                                                |             Test 2              |                                       Test 3                                      |
| ------------------------------------- | :------------------------------------------: | :----------------------------------------------------------------------------------------------------: | :------------------------------: | :---------------------------------------------------------------------------------: |
| Dataset                               |                    Face Shape Set                     |                                                 Face Shape Set                                                  |           Face Shape Set            |                                     Face Shape Set                                      |
| Dataset division                      |                   70:15:15                   |                                                70:15:15                                                |             60:20:20             |                                      60:20:20                                       |
| Original examples                              |                 5.000 images                 |                                              5.000 images                                              |          5.000 images           |                                    5.000 images                                    |
| Size of each image                    |                  178x218x3                   |                                               178x218x3                                                |            500x500x3             |                                      500x500x3                                      |
| First operations                      |                     None                     | faces are extracted by means of face_recognition models from images previously converted in grayscale |               None               | Harmful images are removed with the pre-trained model_glasses specifically designed |
| Examples                              |                  Unchanged                   |                                              4990 images                                               |            Unchanged             |                                     8146 images                                     |
| New image size                        |                  Unchanged                   |                                                96x48x1                                                 |            Unchanged             |                                      Unchanged                                      |
| Pre-processing                        |       Images are rescaled and reshaped       |         HOG features extracted from face images are standardised before being reduced by PCA          | Images are rescaled and reshaped |                          Images are rescaled and reshaped                           |
| Data augmentation on training dataset | Images are randomly and horizontally flipped |                                                  None                                                  |               None               |                                        None                                         |
| Input example shape                   |                   96x96x3                    |                                                 360x1                                                  |            224x224x3             |                                      224x224x3                                      |
| Model                                 |                     CNN                      |                                                  CNN                                                   |               CNN               |                                        CNN                                        |
| Batch size                            |                      16                      |                                                  None                                                  |                16                |                                         16                                          |
| Epoch                                 |                      25                      |                                                  None                                                  |                10                |                                         10                                          |

## How to start


The packages required for the execution of the code along with the role of each file and the software used are described in the Sections below.

## Packages required

The following lists gather all the packages needed to run the project code.
Please note that the descriptions provided in this subsection are taken directly from the package source pages. For more details it is reccomended to directly reference to their official sites.

**Compulsory :**

- **Pandas** provides fast, flexible, and expressive data structures designed to make working with structured and time series data both easy and intuitive.

- **Numpy** is the fundamental package for array computing with Python.

- **Tensorflow** is an open source software library for high performance numerical computation. Its allows easy deployment of computation across a variety of platforms (CPUs, GPUs, TPUs). **Important**: Recently Keras has been completely wrapped within Tensorflow.

- **Pathlib** offers a set of classes to handle filesystem paths.

- **Shutil** provides a number of high-level operations on files and collections of files. In particular, functions are provided which support file copying and removal.

- **Os** provides a portable way of using operating system dependent functionality.

- **Matplotlib** is a comprehensive library for creating static, animated, and interactive visualizations in Python.

- **Sklearn** offers simple and efficient tools for predictive data analysis.

- **Skimage** is a collection of algorithms for image processing.

- **Random** implements pseudo-random number generators for various distributions.

- **Cv2** is an open-source library that includes several hundreds of computer vision algorithms.

- **Face_recognition** is useful to recognize and manipulate faces with the world’s simplest face recognition library. Built from dlib’s state-of-the-art deep learning library.

**Optional :**

- **Comet_ml** helps to manage and track machine learning experiments.

- **Vprof** is a Python package providing rich and interactive visualizations for various Python program characteristics such as running time and memory usage.

## Software used

> <img src="https://financesonline.com/uploads/2019/08/PyCharm_Logo1.png" width="200" alt="pycharm">

PyCharm is an integrated development environment (IDE) for Python programmers: it was chosen because it is one of the most advanced working environments and for its ease of use.

> <img src="https://cdn-images-1.medium.com/max/1024/1*-QTg-_71YF0SVshMEaKZ_g.png" width="140" alt="tesorflow">

TensorFlow is an end-to-end open source platform for machine learning. It has a comprehensive, flexible ecosystem of tools.

> <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/3/3c/Flask_logo.svg/1200px-Flask_logo.svg.png" width="140" alt="flask">

flask is a front end framework that is very well suited for handling pythons end points.

```
--->
