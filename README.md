# Enabling Viewpoint Learning through Dynamic Label Generation

Tensorflow implementation of *Enabling Viewpoint Learning through Dynamic Label Generation*, published at Eurographics 2021.

[M. Schelling](https://www.uni-ulm.de/en/in/mi/institute/staff/michael-schelling/), 
[P. Hermosilla](https://www.uni-ulm.de/in/mi/institut/mitarbeiter/pedro-hermosilla-casajus/), 
[P.-P. Vázquez](https://www.cs.upc.edu/~ppau/index.html)
and [T. Ropinski](https://www.uni-ulm.de/in/mi/institut/mitarbeiter/timo-ropinski/)

![Teaser](https://raw.githubusercontent.com/schellmi42/viewpoint_learning/main/images/Teaser3.png)

[Paper Pre-Print](https://arxiv.org/abs/2003.04651) 

[Project Page](https://www.uni-ulm.de/in/mi/mi-forschung/viscom/publikationen?category=publication&publication_id=183)

## Prerequisites

- Download and compile the [MCCNN](https://github.com/viscom-ulm/MCCNN) library and place it it the `MCCNN/` folder.
- Download the data and place it in the `viewpoint_learning/data/` folder

This Implementation is in TensorFlow 1 and was tested using TF 1.11 and Python 2.7.
For the viewpoint computation methods the OpenGL package for python is required. For training this is not necessary.
We recommend to run training in a `tf=1.11_gpu` docker container.

## Example Training
The root directory contains scripts to train viewpoint prediction networks using dynamic label generation with Multiple Labels (*ML*), Gaussian Labels (*GL*) and a two staged learning using both (*ML-GL*).

Reference implenentation are given for Single Label (*SL*), Spherical Regression (*SR*), Deep Label Distribution Learning (*DLDL*).

Note: By default this only trains the *airplane* category, to train other categories in parallel on multiple GPUs please uncomment the respective lines in the `script_*.sh` files.

## View Quality computation
The function to compute view qualites from meshes are in `viewpoint_learning/code/DataProcessing.py`.

A basic computation can be done via:
```python
python DataProcessing.py --generate_views --f DATA_DIR
```
 which computes Visibility Ratio, Viewpoint Entropy, Viewpoint Kullback-Leibler Distance and Viewpoint Mutual Information.
