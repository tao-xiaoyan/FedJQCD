# FedJQCD

FedJQCD is a research codebase for simulating federated learning with control variates, quantized client updates, and error feedback. The repository provides a standalone FedJQCD experiment runner and a separate runner for comparison baselines.

## Repository Structure

| Path | Contents |
| --- | --- |
| [FedJQCD.py](FedJQCD.py) | FedJQCD simulation, quantization, error feedback, and result plots |
| [system/main.py](system/main.py) | Entry point for FedAvg, FedProx, SCAFFOLD, and FedDyn |
| [system/flcore/](system/flcore/) | Baseline clients and servers, optimizers, and model definitions |
| [system/utils/](system/utils/) | Client-data loading and result utilities |
| [dataset/](dataset/) | Dataset preparation scripts and bundled client partitions |
| [env_cuda_latest.yaml](env_cuda_latest.yaml) | Conda environment specification |

## Requirements and Installation

The supplied Conda specification requests Python 3.11 and PyTorch 2.0.1. Install it from the repository root.

```sh
conda env create -f env_cuda_latest.yaml
conda activate fedjqcd
```

The environment's pip entries leave some package versions open, including torchvision. If dependency resolution fails, select a torchvision build compatible with your PyTorch version and platform. The FedJQCD runner imports SwanLab and initializes experiment tracking during a run. The baseline runner can disable tracking.

## Dataset Preparation

| Dataset | FedJQCD runner | Baseline runner |
| --- | --- | --- |
| MNIST | Downloads through torchvision into the runtime directory `data/` and partitions the training set during the run. | Reads the 20 bundled client partitions in [dataset/MNIST/](dataset/MNIST/). |
| CIFAR-10 | Downloads through torchvision into the runtime directory `data/` and partitions the training set during the run. Use `--dataset CIFAR10`. | Reads client partitions from `dataset/Cifar10/`, generated separately. Use `-data Cifar10` or `-data CIFAR10`. |
| Shakespeare | Reads the bundled partitions in [dataset/Shakespeare/](dataset/Shakespeare/) and uses the shared `LSTMNet` model. | Reads the same bundled partitions and uses `LSTMNet`. |

To prepare the 20 CIFAR-10 client partitions for the baseline runner, run the following from the repository root. The [generator](dataset/generate_Cifar10.py) downloads CIFAR-10 and creates a non-IID partition using its Dirichlet option.

```sh
cd dataset
python generate_Cifar10.py noniid balance dir
```

The baseline runner expects numbered `train/<client_id>.npz` and `test/<client_id>.npz` files. Match the requested client count to the available shards. The bundled Shakespeare partitions provide 118 clients. [generate_Shakespeare.py](dataset/generate_Shakespeare.py) can regenerate those files from the bundled LEAF-format JSON when run from `dataset/`. Regeneration writes to the existing shard paths. The shared loader reads pickled NumPy objects, so use trusted shards.

## Supported Algorithms and Configurations

- **FedJQCD.** The standalone entry point enables control variates. Use `--use-quant` for model-update quantization and add `--use-ef` for model-update error feedback. Use `--quant-control` to quantize control variables, with `--ef-control` for feedback on client control updates. Bit widths and group sizes are configurable. The CLI accepts the legacy `--early-stop-*` options without using them to stop training. It also accepts `--download-quant-bits`, `--download-group-size`, and `--download-scale-bits`, which the entry point currently does not pass to the experiment function.
- **Comparison baselines.** Select FedAvg, FedProx, SCAFFOLD, or FedDyn with `-algo`. The baseline runner supports CNN or ResNet18 for image datasets and LSTM for Shakespeare. Use `-nc` and `-jr` to set the client count and participation ratio.

## Quick Start

From the repository root, inspect the FedJQCD options or run a one-round Shakespeare example with 4-bit model-update quantization and error feedback. The example uses two bundled client shards and initializes SwanLab.

```sh
python FedJQCD.py --help
python FedJQCD.py --dataset Shakespeare --num-clients 2 --rounds 1 --local-epochs 1 --seeds 0 --use-quant --quant-bits 4 --use-ef  --quant-control-bits 4 --ef-control 
```

From the repository root, enter `system/` to run the baseline example. It uses CPU, bundled MNIST shards, and disabled SwanLab tracking.

```sh
cd system
python main.py --help
python main.py -algo FedAvg -data MNIST -m CNN -dev cpu -nc 2 -gr 1 -ls 1 --seed 0 --swanlab_mode disabled
```

## Experiments and Reproducibility

The quick-start settings are illustrative. For an experiment, record the repository revision, environment, dataset partition, model, seeds, client participation, local training settings, and communication settings. Set FedJQCD seeds with `--seeds`. The baseline runner uses `--seed` as the starting seed and `-t` to set the number of runs.

For MNIST and CIFAR-10, the two entry points use different data preparation and evaluation protocols. FedJQCD partitions the torchvision training set and evaluates on the torchvision test set. The baselines use the prepared client train and test shards. For Shakespeare, match the client IDs and LSTM dimensions across runners using `--feature-dim` and `-fd`. Communication volume and time are simulated from message-size estimates, bandwidth, and latency settings.

## Output and Logging

- **FedJQCD.** The runtime output directory defaults to `figures/` and can be set with `--save-dir`. The runner saves per-seed metric histories as `*_results.pt` and writes PNG plots. SwanLab also receives per-round metrics, including cumulative upload and total communication estimates.
- **Baselines.** When launched from `system/`, the runner writes HDF5 histories under the runtime directory `results/` at the repository root and saves server models under `system/models/`. SwanLab tracking defaults to online mode and can be disabled with `--swanlab_mode disabled`.

## Acknowledgements

This repository is adapted from [PFLlib](https://github.com/TsingZ0/PFLlib). The comparison framework and shared client-data and model components build on the upstream project. This README was rewritten for FedJQCD. The repository adds the standalone FedJQCD runner and adapts the baseline entry point and experiment logging for the comparisons described here. For the upstream framework, see the [PFLlib paper in JMLR](https://www.jmlr.org/papers/v26/23-1634.html).

## License

See the included [Apache License 2.0](LICENSE) for the code license. Dataset redistribution terms are separate from the code license.
