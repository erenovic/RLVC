# Recurrent Learned Video Compression (RLVC)
An unofficial implementation of Recurrent Learned Video Compression Architecture using PyTorch

### The model is a reimplementation of architecture designed by Yang et al. (2021) 
For further details about the model and training, please refer to the the official project page and [Github repository](https://github.com/RenYang-home/RLVC):
> Ren Yang, Fabian Mentzer, Luc Van Gool and Radu Timofte, "Learning for Video Compression with Recurrent Auto-Encoder and Recurrent Probability Model", IEEE Journal of Selected Topics in Signal Processing (J-STSP), 2021. [[Paper]](https://ieeexplore.ieee.org/abstract/document/9288876)

```
@article{yang2021learning,
  title={Learning for Video Compression with Recurrent Auto-Encoder and Recurrent Probability Model},
  author={Yang, Ren and Mentzer, Fabian and Van Gool, Luc and Timofte, Radu},
  journal={IEEE Journal of Selected Topics in Signal Processing},
  volume={15},
  number={2},
  pages={388-401},
  year={2021}
}
```

The training and the re-implementation has to be followed according to the specifications in the paper. The training code provided here is for the fine-tuning of the model at the end.

For the key frame compression, the learned image compression model by Balle et al. (2018) is used. The implementation is taken from the ```compressai``` library:

```
@inproceedings{minnenbt18,
  author    = {David Minnen and
               Johannes Ball{\'{e}} and
               George Toderici},
  editor    = {Samy Bengio and
               Hanna M. Wallach and
               Hugo Larochelle and
               Kristen Grauman and
               Nicol{\`{o}} Cesa{-}Bianchi and
               Roman Garnett},
  title     = {Joint Autoregressive and Hierarchical Priors for Learned Image Compression},
  booktitle = {Advances in Neural Information Processing Systems 31: Annual Conference
               on Neural Information Processing Systems 2018, NeurIPS 2018, 3-8 December
               2018, Montr{\'{e}}al, Canada},
  pages     = {10794--10803},
  year      = {2018},
}
```
```
@article{begaint2020compressai,
	title={CompressAI: a PyTorch library and evaluation platform for end-to-end compression research},
	author={B{\'e}gaint, Jean and Racap{\'e}, Fabien and Feltman, Simon and Pushparaja, Akshay},
	year={2020},
	journal={arXiv preprint arXiv:2011.03029},
}
```


