import subprocess
import time
import hydra
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig
import pathlib
import os
import shutil
from datetime import datetime
from singlerun_metrics import generate_benchmark_results


def build_file_name(project_name, overrides, keys_to_include):
    parts = [project_name]
    for key in keys_to_include:
        if key in overrides:
            short_key = key.split(".")[-1]
            value = str(overrides[key])
            parts.append(f"{short_key}{value}")
    return "_".join(parts) + ".json.txt"


def check_log(exp_info):
    """
    Check whether the metrics log file already exists for this experiment.
    """
    exp_task = exp_info["exp_task"]
    overrides = exp_info["overrides"]
    exp_loggers = exp_info["exp_loggers"]
    dataset_type = exp_info["dataset_type"]

    if dataset_type:
        exp_task = f"{exp_task}_{dataset_type}"

    project_name_logger, keys_to_include_logger = (
        exp_loggers.project_name_logger,
        exp_loggers.params_logger,
    )
    project_name = overrides.get(project_name_logger, "default_project")
    file_name = build_file_name(project_name, overrides, keys_to_include_logger)
    save_path = os.path.join("metrics", exp_task, file_name)
    print("save_path", save_path)
    return os.path.exists(save_path)


def save_metrics(output_dir, exp_info):
    exp_task = exp_info["exp_task"]
    overrides = exp_info["overrides"]
    exp_loggers = exp_info["exp_loggers"]
    dataset_type = exp_info["dataset_type"]

    if dataset_type:
        exp_task = f"{exp_task}_{dataset_type}"

    log_path = os.path.join(output_dir, "logs.json.txt")
    metrics_dir = os.path.join("metrics", exp_task)
    os.makedirs(metrics_dir, exist_ok=True)

    project_name_logger, keys_to_include_logger = (
        exp_loggers.project_name_logger,
        exp_loggers.params_logger,
    )
    project_name = overrides.get(project_name_logger, "default_project")
    file_name = build_file_name(project_name, overrides, keys_to_include_logger)
    save_path = os.path.join(metrics_dir, file_name)

    try:
        shutil.copyfile(log_path, save_path)
        print(f"Copied log to {save_path}")
    except Exception as e:
        print(f"Failed to copy log: {e}")


def build_command(config_name, overrides, output_dir):
    cmd = ["python", "run/common/train.py", f"--config-name={config_name}"]
    for key, value in overrides.items():
        cmd.append(f"{key}={value}")

    now = datetime.now()
    time_str = now.strftime("%H.%M.%S")
    output_dir = output_dir + f"/{time_str}_{config_name}"
    cmd.append(f"hydra.run.dir='{output_dir}'")

    return " ".join(cmd), output_dir


def run_all_experiments(task, configs, logger, metrics, output_dir):
    """
    Run all experiments sequentially with logging and error handling.
    """
    exp_info = dict(exp_task=task, exp_loggers=logger)
    all_dataset_types = {
        exp_config["overrides"].get("task.dataset_type") for exp_config in configs
    }

    for exp_idx, exp_config in enumerate(configs):
        print(
            f"\n========== Running experiment {exp_idx + 1}/{len(configs)} =========="
        )

        # 1. extract a single exp setting
        config_name = exp_config["config_name"]
        overrides = exp_config["overrides"]
        dataset_type = overrides.get("task.dataset_type")
        info = {
            "idx": exp_idx,
            "config_name": config_name,
            "overrides": overrides,
            "dataset_type": dataset_type,
        }
        exp_info.update(info)
        print(f"Config: {config_name}")

        # 2. skip the experiment if its metrics log already exists
        if check_log(exp_info):
            print(f"⏩ Skipping experiment {exp_idx + 1}: log already exists.")
            continue

        # construct the exp command
        command, output_dir_ = build_command(config_name, overrides, output_dir)
        print(f"Executing command:\n{command}")

        # 3. run a single experiment
        try:
            subprocess.run(command, shell=True, check=True)
            save_metrics(output_dir_, exp_info)
        except subprocess.CalledProcessError as e:
            print(f"❌ Experiment {exp_idx + 1} failed with error: {e}")

        time.sleep(3)

    print("\n✅ All experiments completed.")

    # 4. save benchmark
    metrics_directory = "metrics/" + task
    metric_key = metrics.metric_key
    metrics = metrics.metrics
    for dataset_type in all_dataset_types:
        if dataset_type:
            metrics_directory_ = f"{metrics_directory}_{dataset_type}"
            print(f"📊 Generating benchmark for dataset_type: {dataset_type}...")
        else:
            metrics_directory_ = metrics_directory
            print(f"📊 Generating benchmark for default dataset_type...")

        generate_benchmark_results(
            metrics_dir=metrics_directory_, key=metric_key, metrics=metrics
        )


@hydra.main(
    # resolve to an absolute path so the config dir is found no matter where the script is invoked
    config_path=str(
        pathlib.Path(__file__).resolve().parent.parent.parent.joinpath("config/train")
    ),
    config_name="benchmark_lowdim",
    version_base=None,
)
def main(cfg: DictConfig):
    """
    Main function to load experiments from hydra config and execute them.
    """
    hydra_cfg = HydraConfig.get()
    output_dir = hydra_cfg.run.dir
    print(f"Hydra's output directory is: {output_dir}")

    run_all_experiments(
        task=cfg.task_name,
        configs=cfg.configs,
        logger=cfg.logger,
        metrics=cfg.metrics,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
