import sys
import os
import logging
import argparse
import importlib

from models import aerca
from utils import utils
import warnings
warnings.filterwarnings("ignore")


def _lazy_import(module_path: str, attr: str):
    """按需 import 数据集 / 参数模块，避免某些数据集（如 lotka_volterra
    依赖 numba）的导入错误拖垮整个进程。"""
    mod = importlib.import_module(module_path)
    return getattr(mod, attr)


def main(argv, progress_callback=None, data_class=None):
    """
    Main function to run the AERCA model on a specified dataset.
    当从 Streamlit/FastAPI 调用时，返回结构化结果；直接运行时保持原有打印行为。

    Parameters
    ----------
    argv : list[str]
        命令行参数（可被 sys.argv hack 注入）。
    progress_callback : Optional[Callable[[dict], None]]
        训练进度回调，每个 epoch 结束触发一次。
    data_class : Optional[object]
        如果传入，则直接使用其 data_dict（避免重复生成数据）。
    """
    # Preliminary parsing: retrieve the dataset name.
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument(
        '--dataset_name',
        type=str,
        default='linear',
        help='Name of the dataset to run. Options: linear, lotka_volterra, lorenz96, msds, swat, nonlinear'
    )
    pre_args, remaining_args = pre_parser.parse_known_args(argv[1:])
    dataset_name = pre_args.dataset_name.lower()

    # 数据集懒加载映射：(dataset module path, class name, args module path, log file, use_slice)
    dataset_meta = {
        "linear":         ("datasets.linear",         "Linear",       "args.linear_args",         "linear.log",         True),
        "lotka_volterra": ("datasets.lotka_volterra", "LotkaVolterra","args.lotka_volterra_args", "lotka_volterra.log", True),
        "lorenz96":       ("datasets.lorenz96",       "Lorenz96",     "args.lorenz96_args",       "lorenz96.log",       True),
        "msds":           ("datasets.msds",           "MSDS",         "args.msds_args",           "msds.log",           False),
        "swat":           ("datasets.swat",           "SWaT",         "args.swat_args",           "swat.log",           False),
        "nonlinear":      ("datasets.nonlinear",      "Nonlinear",    "args.nonlinear_args",      "nonlinear.log",      True),
    }

    if dataset_name not in dataset_meta:
        print("Dataset '{}' not recognized. Available options are: {}".format(
            dataset_name, list(dataset_meta.keys())))
        sys.exit(1)

    ds_mod_path, ds_cls_name, args_mod_path, log_file, use_slice = dataset_meta[dataset_name]
    mapping = {
        "args": _lazy_import(args_mod_path, "create_arg_parser"),
        "dataset_class": _lazy_import(ds_mod_path, ds_cls_name),
        "log_file": log_file,
        "use_slice": use_slice,
    }

    # Set up logging
    logging_dir = os.path.join(os.getcwd(), "logs")
    if not os.path.exists(logging_dir):
        os.makedirs(logging_dir)
    log_file_path = os.path.join(logging_dir, mapping["log_file"])
    logging.basicConfig(
        filename=log_file_path,
        filemode='w',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Parse arguments
    parser = mapping["args"]()
    args, unknown = parser.parse_known_args(remaining_args)
    options = vars(args)

    # Set seed
    utils.set_seed(options['seed'])
    print('Set seed: {}'.format(options['seed']))

    # Dataset
    if data_class is None:
        data_class = mapping["dataset_class"](options)
        if options['preprocessing_data'] == 1:
            print('Preprocessing data: generating and saving new data...')
            data_class.generate_example()
            data_class.save_data()
        else:
            print('Loading existing data...')
            data_class.load_data()
    else:
        print('Using injected data_class (skip data generation).')

    # AERCA model
    aerca_model = aerca.AERCA(
        num_vars=options['num_vars'],
        hidden_layer_size=options['hidden_layer_size'],
        num_hidden_layers=options['num_hidden_layers'],
        device=options['device'],
        window_size=options['window_size'],
        stride=options['stride'],
        encoder_gamma=options['encoder_gamma'],
        decoder_gamma=options['decoder_gamma'],
        encoder_lambda=options['encoder_lambda'],
        decoder_lambda=options['decoder_lambda'],
        beta=options['beta'],
        lr=options['lr'],
        epochs=options['epochs'],
        recon_threshold=options['recon_threshold'],
        data_name=options['dataset_name'],
        causal_quantile=options['causal_quantile'],
        root_cause_threshold_encoder=options['root_cause_threshold_encoder'],
        root_cause_threshold_decoder=options['root_cause_threshold_decoder'],
        risk=options['risk'],
        initial_level=options['initial_level'],
        num_candidates=options['num_candidates']
    )

    print(f"模型运行在：{next(aerca_model.parameters()).device}")

    # Training
    if options['training_aerca']:
        if mapping["use_slice"]:
            training_data = data_class.data_dict['x_n_list'][:options['training_size']]
        else:
            training_data = data_class.data_dict['x_n_list']
        print('Start training AERCA model...')
        aerca_model._training(training_data, progress_callback=progress_callback)
        print('Done training')

    def _emit(phase, message=None):
        if progress_callback is not None:
            try:
                progress_callback({'phase': phase, 'message': message or phase})
            except Exception:  # noqa: BLE001
                pass

    # Causal discovery test
    _emit('causal_discovery_start', 'Causal discovery testing...')
    if mapping["use_slice"]:
        test_causal = data_class.data_dict['x_n_list'][options['training_size']:]
        print('Start testing AERCA model for causal discovery...')
        causal_results = aerca_model._testing_causal_discover(
            test_causal, data_class.data_dict['causal_struct'])
        print('Done testing for causal discovery')
    else:
        causal_results = None
    _emit('causal_discovery_done', 'Causal discovery complete.')

    # Root cause analysis
    if mapping["use_slice"]:
        test_x_ab = data_class.data_dict['x_ab_list'][options['training_size']:]
        test_label = data_class.data_dict['label_list'][options['training_size']:]
    else:
        test_x_ab = data_class.data_dict['x_ab_list']
        test_label = data_class.data_dict['label_list']

    _emit('root_cause_start', 'Root cause analysis...')
    print('Start testing AERCA model for root cause analysis...')
    root_cause_results = aerca_model._testing_root_cause(test_x_ab, test_label)
    print('Done testing for root cause analysis')
    _emit('root_cause_done', 'Root cause analysis complete.')

    print('done')

    # ====================== 返回结构化结果供 Streamlit / FastAPI 使用 ======================
    return {
        'root_cause_results': root_cause_results,
        'causal_results': causal_results,
        'test_x_ab': test_x_ab,
        'test_label': test_label,
        'test_causal': test_causal if mapping["use_slice"] else None,
        'use_slice': mapping["use_slice"],
        'training_size': options.get('training_size', 200),
        'num_vars': options.get('num_vars', 6),
        'data_class': data_class,
        'options': options,
        'aerca_model': aerca_model,
    }


if __name__ == '__main__':
    main(sys.argv)