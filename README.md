# NAS-Bench-COMPRE: A Comprehensive Neural Architecture Search Benchmark with Decoupled Components

## Introduction

NAS-Bench-COMPRE is a benchmark for neural architecture search (NAS), which aims to provide a fair and comprehensive evaluation of **search spaces**, **search strategies**, **estimation strategies**. The NAS algorithms he currently supports include: DARTS, GMAENAS, NAG, OFA, AutoFormer, ViTSoup, TrainingFreeNAS.

This repostory contains six main files and directories:

- configs: The directory of the yaml configuration files that are used to configure the NAS algorithm to be run.
- lib: The directory of library files that define supernets, subnets, neural architecture generators, neural architecture performance predictors, training-free proxies, dataloaders, logging, etc.
- search_spaces: The directory of search spaces.
- search_strategies: The directory of search strategies.
- estimation_strategies: The directory of estimation strategies.
- api.py: It defines the class API to use this benchmark.

In addition, there are also some other directories that store model files, datasets, and output logs:

- datasets: The directory of datasets.
- model-ckpts:  The directory of model checkpoint. Download the necessary files of model checkpoints in Baidu Netdisk (link: https://pan.baidu.com/s/1PbxCrxqfDK-VG3d99ezJEA?pwd=ccip).
- output: The directory of output log files.

## Features

- **More comprehensive**: It can achieve the evaluation of **search spaces**, **search strategies**, **estimation strategies**.
- **Simpler**: It can decouple the three components of neural architecture search. We only need to focus on the implementation of the parts to be evaluated.
- **Fairer**: Decoupled components can ensure fairness.

## Getting Started

To use NAS-Bench-COMPRE, follow these steps:

1. Clone the repository: `git clone https://github.com/kunjing96/NAS-Bench-COMPRE`
2. Install the necessary dependencies: `pip install -r requirements.txt`
3. Add your search spaces, search strategies, or estimation strategies:
    - Create a your file for your search space, search strategy, or estimation strategy in corresponding directory
    - For your search space,
    ```
    @_register
    class YourSearchSpace(Base):

        def __init__(self, config):
            super(YourSearchSpace, self).__init__(config)
            # add your initialization

        def is_valid(self, arch):
            # add code to determine whether the architecture is valid, generally used for controlling model parameters, FLOPs, and latency.
            return True

        def encode(self, decoded_arch):
            arch = None
            # add code to encode architectures
            return arch

        def decode(self, arch):
            decoded_arch = None
            # add code to deconde architectures
            return decoded_arch

        def sample(self):
            arch = list()
            # sample an architecture
            return tuple(arch)
    ```
    - For your search strategy,
    ```
    @_register
    class YourSearchStrategy(Base):

        def __init__(self, config):
            super(YourSearchStrategy, self).__init__(config)
            # add your initialization

        def __call__(self):
            history = None # search history
            best = None # best architecture
            cost = None # time cost
            # define your search strategy
            return best, history, cost
    ```
    - For your estimation strategy,
    ```
    @_register
    class YourEstimationStrategy(Base):

        def __init__(self, config):
            super(YourEstimationStrategy, self).__init__(config)
            # add your initialization

        def __call__(self):
            val_res = None # validate result
            test_res = None # test result
            # define your estimation strategy
            return val_res, test_res
    ```
4. Modify the configuration file, and modify the path of your configuration file in api.py or your code
5. Run the script: `python api.py` or your code

Useing example:

```
    config = get_config(cfg_file=os.path.join('configs', <Filepath to Your Config>))
    if config.OUTPUT:
        Path(config.OUTPUT).mkdir(parents=True, exist_ok=True)

    prepare_logger(config=config)

    logging.info('Available search spaces:')
    logging.info(available_search_spaces())
    logging.info('Available search strategies:')
    logging.info(available_search_strategies())
    logging.info('Available estimation strategies:')
    logging.info(available_estimation_strategies())

    api = API(config=config)

    logging.info('='*5 + 'Search Start' + '='*5)
    best, history, time_cost = api.run()
    logging.info('*Best:')
    logging.info(best)
    logging.info('History:')
    logging.info(history)
    logging.info('Time cost:')
    logging.info(time_cost)
    torch.save(
        {
            'best': best,
            'history': history,
            'time_cost': time_cost
        },
        os.path.join(config.OUTPUT, 'checkpoint.pt')
    )
    logging.info('='*5 + 'Search End' + '='*5)
```

## Contributors

We welcome contributions from the community! If you find any issues or have any suggestions, please submit a pull request or issue report on GitHub.

<!-- ## Citation

If you use this benchmark in your research, please cite our paper:

{INSERT YOUR PAPER CITATION HERE} -->

<!-- ## License

This project is licensed under the {INSERT YOUR LICENSE HERE} license. -->
