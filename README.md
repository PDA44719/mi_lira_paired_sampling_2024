## Likelihood Ratio Attack with Paired Sampling

This directory contains code to reproduce the online LiRA attack used in the paper:

**"Paired Sampling Technique to Enhance state-of-the-art Membership Inference Attacks"** <br>

This code was built on top of the repository created for the paper *"Membership Inference Attacks From First Principles"* by Nicholas Carlini et al [1].

Below, you can find the link to their paper, and the original repository:
- https://arxiv.org/abs/2112.03570
- https://github.com/tensorflow/privacy/tree/master/research/mi_lira_2021

### Environment setup

For the environment set up, the same installation steps that were detailed in the original repository are needed. Firstly, the following dependecies must be installed:

`pip install scipy, sklearn, numpy, matplotlib`

and also JAX + ObJAX will be required. You can follow the build instructions:
https://github.com/google/objax
https://objax.readthedocs.io/en/latest/installation_setup.html

### Reproducing the attacks

#### 1. Training

The first step is to train shadow models. For the experiments, 16 models were trained. The following bash script can be used for training:

> bash scripts/train_standard.sh *seed*
> bash scripts/train_paired.sh *seed* *target_record*

In both cases, the seed value must be specified. Additionally, for *train_paired.py*, the target_record that will be attacked also needs to be specified.

It must be noted that, while for the standard attack 16 models need to be trained, only 8 models need to be trained for the paired sampling version (as paired sampling is only applied in odd-numbered experiment ids).

The scripts can be adapted to use multiple GPUs.

These scripts will train the shadow models to 88-90% accuracy each, and will output a series of files under the directory exp/cifar10 with structure:

```
exp/cifar10/
- standard/
-- experiment_N_of_16/
--- hparams.json
--- keep.npy
--- ckpt/
---- 0000000035.npz
--- tb/
- paired/
-- record_N1/
--- experiment_N_of_16/
---- hparams.json
---- keep.npy
---- ckpt/
----- 0000000035.npz
---- tb/
```

where N1 is the target record that was attacks (an integer).

#### 2. Copying the even-numbered experiments
This step ust be completed after all the experiments (standard and paired) have been completed.

As was mentioned in step 1, only 8 models are needed to be trained for paired sampling. This is the case because, as only odd-numbered experiments implement paired sampling, the models trained for even-numbered experiments will be the same as the ones trained with *bash scripts/train_standard.sh*.

Therefore, you now need to copy the even-numbered experiments from the standard directory to the paired one. This can be done with the following script:

> bash scripts/copying_experiments *exp_directory* *target_record*

where *exp_directory* is the directory where the experiments are saved (*exp/cifar10* in the default case) and *target_record* is the record (i.e., integer value) that was attacked using paired sampling.

#### 3. Inference and Scoring

Once the copying step has been completed, now inference will be performed in order to compute the output features for each training example for each model in the dataset. Those output features will then be used to generate the logit-scaled membership inference score.

> bash scripts/inference_and_scoring.sh *exp_dir* *target_record*

Again, the experiment directory and target record needs to be specified. This script will add to the experiment directory a new set of files

```
exp/cifar10/
- standard/
-- experiment_N_of_16/
--- logits/
---- 0000000035.npy
--- scores/
---- 0000000035.npy
- paired/
-- record_N1/
--- experiment_N_of_16/
---- logits/
----- 0000000035.npy
---- scores/
----- 0000000035.npy
```

The score files have the shape (50000,) as there are 50000 records in the dataset.

### 4. Computing the results

So far, the steps above have described the process for attacking a **single** target record. Ideally, you should attack multiple records. If you repeat this process for various records, the final structure of the paired subdirectory should be the following:

```
exp/cifar10/
- paired/
-- record_N1/
--- [...]
-- record_N2/
--- [...]
-- record_N3/
--- [...]
```

As can be seen in the example structure above, three records were attacked. 

Once you have that structure, you can now compute the results for all the records that were attacked.

> python3 compute_results.py --exp_dir=exp/cifar10/standard --target_records=N1,N2,N3

> python3 compute_results.py --exp_dir=exp/cifar10/paired --target_records=N1,N2,N3 --paired_sampling=True

where target_records should be a comma-separated list of the records attacked.

**compute_results.py** will return two lists. The first containing the accuracy results for the attack for each of the target_records and a second list containing the AUC scores also for each of the target records, in the order specified in *--target_records*. For example, for three attacks, you may obtain:
```
Accuracy -> [0.578, 0.6012, 0.6201]
AUC -> [0.53125, 0.39062, 0.50519]
```

If you execute compute_results in the same order as detailed in this document, you would first obtain the accuracy and AUC results for the standard attack, and then the accuracy and AUC scores for the paired sampling variant.

### Citation

As previously mentioned, this code was built on top of the repository for the paper below:

[1] N. Carlini, S. Chien, M. Nasr, S. Song, A. Terzis, and F. Tramer, “Membership inference attacks from first principles,” in 2022 IEEE Symposium on Security and Privacy (SP). IEEE, 2022, pp. 1897–1914. pages 5, 6, 16, 20, 21

