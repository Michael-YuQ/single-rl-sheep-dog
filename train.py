"""
Entry point for Hierarchical DreamerV3 shepherding experiment.
Patches the dreamerv3 make_env to add the 'shepherd' suite,
merges our shepherd.yaml config, then launches training.
"""

import sys
import pathlib

# Make dreamerv3 importable
_root = pathlib.Path(__file__).parent.parent / "minecraft" / "dreamerv3"
sys.path.insert(0, str(_root))
sys.path.insert(1, str(pathlib.Path(__file__).parent))

import importlib
import os
import ruamel.yaml as yaml
import elements
import embodied
import numpy as np
import portal
from functools import partial as bind

import dreamerv3.main as dv3main
from dreamerv3.agent import Agent

# ------------------------------------------------------------------
# Patch make_env to include shepherd suite
# ------------------------------------------------------------------
_original_make_env = dv3main.make_env


def _patched_make_env(config, index=0, **overrides):
    suite, task = config.task.split("_", 1)
    if suite == "shepherd":
        from envs.shepherd_dreamer import ShepherdEnv
        env = ShepherdEnv(task=task, n_sheep=10)
        return dv3main.wrap_env(env, config)
    if suite == "perimeter":
        from envs.shepherd_dreamer import PerimeterShepherdEnv
        env = PerimeterShepherdEnv(task=task, n_sheep=10)
        return dv3main.wrap_env(env, config)
    if suite == "push":
        from envs.shepherd_dreamer import PushShepherdEnv
        env = PushShepherdEnv(task=task, n_sheep=100)
        return dv3main.wrap_env(env, config)
    if suite == "threedog":
        from envs.shepherd_dreamer import ThreeDogShepherdEnv
        env = ThreeDogShepherdEnv(task=task, n_sheep=100)
        return dv3main.wrap_env(env, config)
    if suite == "shepherdmultidog":
        from envs.multidog_dreamer import MultiDogShepherdEnv
        env = MultiDogShepherdEnv(task=task, n_sheep=100, n_dogs=10)
        return dv3main.wrap_env(env, config)
    return _original_make_env(config, index, **overrides)


dv3main.make_env = _patched_make_env


# ------------------------------------------------------------------
# Load configs
# ------------------------------------------------------------------
def main(argv=None):
    folder = _root / "dreamerv3"

    # Load base configs
    configs = elements.Path(folder / "configs.yaml").read()
    configs = yaml.YAML(typ="safe").load(configs)

    # Load shepherd configs
    shepherd_cfg_path = pathlib.Path(__file__).parent / "configs" / "shepherd.yaml"
    shepherd_cfgs = yaml.YAML(typ="safe").load(shepherd_cfg_path.read_text())
    configs.update(shepherd_cfgs)
    multidog_cfg_path = pathlib.Path(__file__).parent / "configs" / "shepherd_multidog.yaml"
    multidog_cfgs = yaml.YAML(typ="safe").load(multidog_cfg_path.read_text())
    configs.update(multidog_cfgs)
    perimeter_cfg_path = pathlib.Path(__file__).parent / "configs" / "shepherd_perimeter.yaml"
    perimeter_cfgs = yaml.YAML(typ="safe").load(perimeter_cfg_path.read_text())
    configs.update(perimeter_cfgs)
    push100_cfg_path = pathlib.Path(__file__).parent / "configs" / "shepherd_push100.yaml"
    push100_cfgs = yaml.YAML(typ="safe").load(push100_cfg_path.read_text())
    configs.update(push100_cfgs)
    milestone_cfg_path = pathlib.Path(__file__).parent / "configs" / "shepherd_milestone.yaml"
    milestone_cfgs = yaml.YAML(typ="safe").load(milestone_cfg_path.read_text())
    configs.update(milestone_cfgs)
    threedog_cfg_path = pathlib.Path(__file__).parent / "configs" / "shepherd_threedog.yaml"
    threedog_cfgs = yaml.YAML(typ="safe").load(threedog_cfg_path.read_text())
    configs.update(threedog_cfgs)

    parsed, other = elements.Flags(configs=["defaults"]).parse_known(argv)
    config = elements.Config(configs["defaults"])
    for name in parsed.configs:
        config = config.update(configs[name])
    config = elements.Flags(config).parse(other)
    config = config.update(logdir=(
        config.logdir.format(timestamp=elements.timestamp())))

    logdir = elements.Path(config.logdir)
    print("Logdir:", logdir)
    logdir.mkdir()
    config.save(logdir / "config.yaml")

    def init():
        elements.timer.global_timer.enabled = config.logger.timer

    portal.setup(
        errfile=config.errfile and logdir / "error",
        clientkw=dict(logging_color="cyan"),
        serverkw=dict(logging_color="cyan"),
        initfns=[init],
        ipv6=config.ipv6,
    )

    args = elements.Config(
        **config.run,
        replica=config.replica,
        replicas=config.replicas,
        logdir=config.logdir,
        batch_size=config.batch_size,
        batch_length=config.batch_length,
        report_length=config.report_length,
        consec_train=config.consec_train,
        consec_report=config.consec_report,
        replay_context=config.replay_context,
    )

    embodied.run.train(
        bind(dv3main.make_agent, config),
        bind(dv3main.make_replay, config, "replay"),
        bind(_patched_make_env, config),
        bind(dv3main.make_stream, config),
        bind(dv3main.make_logger, config),
        args,
    )


if __name__ == "__main__":
    main()
