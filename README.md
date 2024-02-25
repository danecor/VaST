# Variational State Tabulation

An implementation of Variational State Tabulation, using Python, TensorFlow and Cython. Based on the paper here: https://arxiv.org/abs/1802.04325.

### Prerequisites

You should have a working installation of TensorFlow (https://www.tensorflow.org/install/). The following should include all required python modules:

```
pip install -r requirements.txt
```

### Installation

You will need to install the Cython submodule and cythonize several modules:

```
git submodule init
git submodule update
chmod +x table/cypridict/install.sh
./table/cypridict/install.sh
cython table/hamming.pyx
cython table/ctable.pyx
```

## Testing

Run

```
pytest
```

to run all unit tests (on the priority queue, the prioritized sweeping algorithm and the replay memory).

### Example Run

Use the command

```
CUDA_VISIBLE_DEVICES=0 python run.py doom tmaze --num_steps=500000 --burnin=10000 --epsilon_period=40000
```

to run the experiment shown in Fig. 6 of the paper. By default, 125 minibatches are loaded at once onto the GPU and a separate thread is used to queue the minibatches for training the network. I recommend running only one job on each GPU (here, on Device 0) to avoid possible concurrency issues.

All of the output data will be written to tensorboard, which you can view with

```
tensorboard --logdir=doom/data/
```

## Author

* **Dane Corneil** - [EPFL]

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* The TensorFlow code was originally based in part on Jan Hendrik Metzen's [VAE implementation](https://jmetzen.github.io/2015-11-27/vae.html)
* The Atari environment wrapper is based on the implementation in [Atari-DeepRL](https://github.com/vvanirudh/Atari-DeepRL) by Nathan Sprague
* The Cython priority queue module is forked from Nan Jiang's [Cyheap tutorial](https://github.com/ncloudioj/cyheap)
