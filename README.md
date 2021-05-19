## Large-scale Multi-agent Safe Control with Neural Contraction Metrics and Barrier Certificates

### TODO
- [x] CBF training with cars and drones
- [ ] CBF training with for two-wheeled vehicle
- [ ] CCM training


### Installation
Create a virtual environment with Anaconda:
```bash
conda create -n macbf python=3.6
```
Activate the virtual environment:
```bash
source activate macbf
```
Clone this repository:
```bash
git clone https://github.com/Zengyi-Qin/macbf.git
```
Enter the main folder and install the dependencies:
```bash
pip install -r requirements.txt
```

### Cars
In `cars`, we provide a multi-agent collision avoidance example with the double integrator dynamics. First enter the directory:
```bash
cd cars
```
To evaluate the pretrained neural network CBF and controller, run:
```bash
python evaluate.py --num_agents 32 --model_path models/model_save --vis 1
```
`--num_agents` specifies the number of agents in the environment. `--model_path` points to the prefix of the pretrained neural network weights. The visualization is disabled by default and will be enabled when `--vis` is set to 1.

To train the neural network CBF and controller from scratch, run:
```bash
python train.py --num_agents 32
```
We can add another argument `--model_path` and point to a pretrained model we want to use.


### Drones
In `drones`, we consider the drone dynamics with 8-dimsional state space. Details of the dynamics can be found in Appendix C of our paper. To experiment with this example, first enter the directory:
```bash
cd drones
```
To evaluate the pretrained neural network CBF and controller, run:
```bash
python evaluate.py --num_agents 32 --model_path models/model_save --vis 1
```
To train the neural network CBF and controller from scratch, run:
```bash
python train.py --num_agents 32
```
The arguments are the same as the cars example.
