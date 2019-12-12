# Detecting Adversarial Attacks via KL Divergence Metrics

Skeleton Script for testing and evaluating proposed KL Divergence metrics for detecting basic adversarial attacks on the MNIST dataset. An overview behind the theory and motivation for this project can be found at the link below:

[Adversarial Attacks on Generative Models: Exploring KL Divergence as a Detection Method](https://www.youtube.com/watch?v=CULalSKiAxk&)

# Setup
To test the code on your own machine, first clone this repo into a specific directory of your choice. This repo relies on the [Cleverhans](https://github.com/tensorflow/cleverhans) repo for generating white noise and fgm attacks, so you will need to also clone that repo into the root folder of this repo. It should also be noted that, while not the same, the source code for the pixel attack was originally drawn from [sarathknv's impelementation](https://github.com/sarathknv/adversarial-examples-pytorch/tree/master/one_pixel_attack).

Your python interpreter needs to have specific packages installed. Torch, scipy, numpy, and matplotlib are all required dependencies, and can be downloaded by running (within your active python environment):

`pip install torch scipy numpy matplotlib`

The `main.py` script requires your own generative model to be implemented and inserted within the script for it to run. The expectations are documented within the script but are shown here in more detail for reference:

```
model (nn.Module): The model to be used for testing. It is expected to have the following attributes:
        .name (str): Name of the model
        .encoder (nn.Module): Neural network with callable, implemented forward function -- must expect inputs of size 784 and output the latent variable means and variances (of total size 2*z_dim)
        .classifier (nn.Module): Neural network with callable, implemented forward function -- must expect inputs of size 784 and output the class logits of y
        .z_prior (torch.nn.Parameter): length-2 tuple that holds the z_prior mean(s) in [0] and z_prior variances in [1]
```

Once implemented, the script can now be run. First, the delta and psi values must be calculated, and can be evaluated by running:

`python main.py --eval_deltas --eval_psis`

Then, specific adversarial tests can be run by specifying attacks as well as flags for adjusting the outputs:

`python main.py --attack [attack] --eps [eps] ...`

Brief documentation for all flags available to be set can be seen by running the -h or --help command:

`python main.py -h`