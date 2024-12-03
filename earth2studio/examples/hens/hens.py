# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime

import hydra
import pandas as pd
from dotenv import load_dotenv
from ensemble_utilities import (
    EnsembleBase,
    EnsembleCycloneTracking,
    EnsembleDiagnostics,
)
from loguru import logger
from modulus.distributed import DistributedManager
from omegaconf import DictConfig
from utilities import (
    initialise,
    initialise_output,
    set_perturbation,
    store_tracks,
    update_model_dict,
    write_to_disk,
)

from earth2studio.enterprise.models.dx import CycloneTrackingVorticity
from earth2studio.models.auto import Package


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Parallelised workflow for running an ensemble inference of a forecast model
    using multiple checkpoints, following the approach of Mahesh et al.
    https://arxiv.org/abs/2408.03100

    Parameters
    ----------
    cfg : DictConfig
        config.
    """

    DistributedManager.initialize()
    load_dotenv()

    ensemble_configs, model_dict, data, output_coords = initialise(cfg)

    if cfg.mode == "cyclone_tracking":
        diagnostic_model = CycloneTrackingVorticity()
    elif cfg.mode == "diagnostic":
        diagnostic_model = hydra.utils.get_class(cfg.diagnostic_model.architecture)
        if "package" in cfg.diagnostic_model:
            if cfg.diagnostic_model.package == "default":
                package = diagnostic_model.load_default_package()
            else:
                package = Package(cfg.diagnostic_model.package)
        else:
            package = diagnostic_model.load_default_package()
        diagnostic_model = diagnostic_model.load_model(package)

    # run forecasts
    all_tracks = []
    then = datetime.now()

    # Initialize threadpool for writers
    writer_executor = (
        ThreadPoolExecutor(max_workers=8) if cfg.file_output.thread_io else None
    )
    writer_threads: list[Future] = []

    for pkg, ic, ens_idx in ensemble_configs:
        # TODO: add start time as optional call, so it works for run.ensemble and run_hens
        #       without having to pass start_time in initialisation

        # load new weights if necessary
        model_dict = update_model_dict(model_dict, pkg)

        io = initialise_output(cfg, ic, model_dict)

        perturbation = set_perturbation(
            model=model_dict["model"], data=data, start_time=ic, cfg=cfg
        )
        if cfg.mode == "base":
            run_hens = EnsembleBase(
                time=[ic],
                nsteps=cfg.nsteps,
                nensemble=cfg.nensemble,
                prognostic=model_dict["model"],
                data=data,
                io=io,
                perturbation=perturbation,
                batch_size=cfg.batch_size,
                output_coords=output_coords,
                ensemble_idx_base=ens_idx,
            )
            io = run_hens()
        elif cfg.mode == "diagnostic":
            run_hens = EnsembleDiagnostics(
                time=[ic],
                nsteps=cfg.nsteps,
                nensemble=cfg.nensemble,
                prognostic=model_dict["model"],
                data=data,
                io=io,
                perturbation=perturbation,
                diagnostic=diagnostic_model,
                batch_size=cfg.batch_size,
                output_coords=output_coords,
                ensemble_idx_base=ens_idx,
            )
            io = run_hens()
        elif cfg.mode == "cyclone_tracking":
            run_hens = EnsembleCycloneTracking(
                time=[ic],
                nsteps=cfg.nsteps,
                nensemble=cfg.nensemble,
                prognostic=model_dict["model"],
                data=data,
                io=io,
                perturbation=perturbation,
                diagnostic=diagnostic_model,
                batch_size=cfg.batch_size,
                output_coords=output_coords,
                ensemble_idx_base=ens_idx,
            )
            tracks = run_hens()
            tracks["ic"] = pd.to_datetime(ic)
            all_tracks.append(tracks)
        else:
            raise ValueError(
                f"mode '{cfg.mode}' doesn't exist. Available modes are 'base' and 'cyclone_tracking'"
            )

        # if in-memory flavour of io backend was chosen, write content to disk now
        writer_threads, writer_executor = write_to_disk(
            cfg, ic, model_dict, io, writer_threads, writer_executor
        )

    now = datetime.now()
    logger.info(
        f"Took {(now-then).total_seconds()}s for {len(ensemble_configs)} ics and "
        + f"{cfg.nsteps} steps rollout with ensemble size {cfg.nensemble}."
    )

    if cfg.mode == "cyclone_tracking":
        store_tracks(all_tracks, cfg)

    if writer_executor is not None:
        for thread in list(writer_threads):
            thread.result()
            writer_threads.remove(thread)
        writer_executor.shutdown()


if __name__ == "__main__":
    main()
