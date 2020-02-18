# CurriculumGAN

The code for our paper: Image Difficulty Curriculum for Generative Adversarial Networks (Petru Soviany, Claudiu Ardei, Radu Tudor Ionescu, Marius Leordeanu) - WACV 2020.

The full paper can be consulted here: https://arxiv.org/abs/1910.08967.

Our code is based on the SNGAN implementation available at: https://github.com/watsonyanghx/GAN_Lib_Tensorflow/tree/master/SNGAN. 
Here we will explain and provide code only for the Curriculum part, which you can easily add to any other model. 
In the future, we plan to provide our own clean implementation of the full network.

For estimating the difficulty of your images, please see http://image-difficulty.herokuapp.com/.

More details will follow.

## Usage

In order to run our Curriculum GAN, you need to clone https://github.com/watsonyanghx/GAN_Lib_Tensorflow/ and to access the SNGAN directory.

All changes to the original network are in the file gan_cifar_resnet.py. You need to copy the functions defined in our curriculum.py file and set the paramaters accordingly. To use curriculum learning you have to set CURRICULUM = True and then pick one of the three modes available (batching/weighting/sampling). The commands for training or testing the model are the same as in the original repository. 

A file with the difficulty scores on cifar10 has been included. In order to generate your own scores, see http://image-difficulty.herokuapp.com/.

## Citation

When using information from this page or from our paper, please cite us.

```
@article{Soviany2019ImageDC,
  title={Image Difficulty Curriculum for Generative Adversarial Networks (CuGAN)},
  author={Petru Soviany and Claudiu Ardei and Radu Tudor Ionescu and Marius Leordeanu},
  journal={ArXiv},
  year={2019},
  volume={abs/1910.08967}
}
```
BibTex will be updated when WACV2020 proceedings will be available.
