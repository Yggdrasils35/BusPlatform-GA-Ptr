# BusPlatform-GA-Ptr
The project proposes an algorithm combined with Genetic Algorithm and Pointer Network to solve multi-object CVRP.

## Usage
To train your own Pointer Network, run

`
python train.py
`

To test your network performance, run

`
python test.py
`

The GA part we change the usual mutation process, instead we record the two gene which is exchanged in per chromosome
during per iteration.

And after all chromosomes have been mutated, we throw all the genes into the network for batch-processing for faster
processing speed.

To use this GA-Ptr algorithm, you can change the data file path and just run the GA-Ptr.py script.

Also we include the benchmark Genetic Algorithm for comparison. All the codes are in the folder GA_baseline.

## Requirements
- Python >= 3.6
- Pytorch==1.9.0

## Reference
- Pointer Network: https://github.com/shirgur/PointerNet
- Genetic Algorithm: `./GA_baseline/`
- L2i: The L2i method is implemented in tensorflow 1.x, so Colab may be a better choice to run code.