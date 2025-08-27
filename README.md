<div align="center"> 

# MeshGraphNets informed Locally by Thermodynamics for the simulation of Flows around Arbitrarly Shaped Objects

*Carlos Bermejo-Barbanoj, Alberto Badías, David González, and Elías Cueto*

[![Project page](https://img.shields.io/badge/-Project%20page-blue)](https://amb.unizar.es/people/carlos-bermejo-barbanoj/)

</div>

## Abstract

We present a thermodynamics-informed graph neural network framework for learning the time
evolution of complex physical systems, incorporating thermodynamic structure via a nodal portmetriplectic
formulation. Built upon the MeshGraphNet architecture, our method replaces the
standard decoder with multiple specialized decoders that predict local energy and entropy gradients,
along with Poisson and dissipative operators. These components are assembled at each graph node
according to the GENERIC formalism, enforcing the first and second laws of thermodynamics. The
framework is evaluated on two examples involving incompressible fluid flow past obstacles: one
with varying cylindrical obstacles and another with obstacles of different types, not seen during
training. The proposed model shows accurate long-term predictions, robust generalization to unseen
geometries, and substantial speedups compared to traditional numerical solvers.

<div align="center">


<img src="data/resources/Ux_error.gif" width="70%">
<img src="data/resources/Uy_error.gif" width="70%">
<img src="data/resources/P_error.gif" width="70%">

</div>

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/cberbarbanoj/Local-Thermodynamics-Informed-MGN.git
   cd Local-Thermodynamics-Informed-MGN
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Initialize Weights & Biases:
   ```bash
   wandb init
   ```


## Usage

Run the training script for any of the examples, the cylinder dataset can be downloaded from https://github.com/google-deepmind/deepmind-research/tree/master/meshgraphnets, if you need more details send a email to cbbarbanoj@unizar.es:
```bash
python main_train.py --dset_name dataset_CYLINDER.json
```

Replace or modify `data/jsonFiles/dataset_CYLINDER.json` with the appropriate configuration for your experiment.

### Test pretrained nets
```bash
python main_inference.py --dset_name dataset_CYLINDER.json --pretrain_weights trained_TIMGN_CYLINDER.ckpt
```


## License

This repository is licensed under the GNU License. See `LICENSE` for details.

---

For any questions or feedback, please contact **Carlos Bermejo Barbanoj** at cbbarbanoj@unizar.es
