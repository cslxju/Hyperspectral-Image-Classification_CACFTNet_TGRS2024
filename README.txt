
## Requirements

-PyTorch 1.6

-CUDA 10.1

-Python 3.7


## Dataset

We used three publicly available datasets, Indian Pines, Pavia University, and Houston2013. The data set can be accessed at the following link:
https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes

Or it can be found in the data folder of the zip package


##Network

Because the number of channels in different data sets is different, the corresponding network structure is also different.

Indian Pines and Houston2013 datasets:  vit_pytorch_indian_Houston.py

Pavia University datasets:  vit_pytorch_pavia.py


## ######################################## Using the code should cite the following paper ######################################## 
S. Cheng, R. Chan and A. Du, "CACFTNet: A Hybrid Cov-Attention and Cross-Layer Fusion Transformer Network for Hyperspectral Image Classification," in IEEE Transactions on Geoscience and Remote Sensing, vol. 62, pp. 1-17, 2024, doi: 10.1109/TGRS.2024.3374081.
@ARTICLE{10460571,
  author={Cheng, Shuli and Chan, Runze and Du, Anyu},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={CACFTNet: A Hybrid Cov-Attention and Cross-Layer Fusion Transformer Network for Hyperspectral Image Classification}, 
  year={2024},
  volume={62},
  number={},
  pages={1-17},
  keywords={Feature extraction;Convolutional neural networks;Transformers;Data mining;Convolution;Image classification;Task analysis;Covariance;cross-layer attention;feature fusion;hyperspectral (HS) image classification;transformer},
  doi={10.1109/TGRS.2024.3374081}}




