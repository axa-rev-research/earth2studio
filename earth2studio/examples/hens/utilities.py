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

import os
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from functools import partial

import hydra
import numpy as np
import pandas as pd
import torch
import xarray as xr
from loguru import logger
from modulus.distributed import DistributedManager
from omegaconf import DictConfig, open_dict

from earth2studio.data import DataSource
from earth2studio.enterprise.perturbation import HemisphericCentredBredVector
from earth2studio.io import IOBackend, KVBackend, XarrayBackend
from earth2studio.models.auto import Package
from earth2studio.models.px import PrognosticModel
from earth2studio.perturbation import (
    BredVector,
    CorrelatedSphericalGaussian,
    Perturbation,
)
from earth2studio.utils.time import to_time_array


def get_noise_vector(
    model: PrognosticModel,
    skill_path: str = None,
    noise_amplification: float = 1.0,
    vars: str | list[str] | None = None,
    lead_time: int = 48,
) -> torch.Tensor:
    """
    obtaining noise vector for HCBV perturbation.

    Parameters
    ----------
    model : PrognosticModel
        forecast model.
    skill_path : str, optional
        path to file containing model skill
    noise_amplification : float, optional
        magnitude by which noise vector gets scaled
    vars : str | list[str] | None, optional
        Variables on which noise will be applied. The elements in the noise
        vector for all other variables will be set to zero. If no variables
        are passed, noise will be applied on all variables
    lead_time : int, optional
        lead time at which model skill is taken.

    Returns
    -------
    torch.Tensor
        Noise vector.
    """
    if skill_path is None:
        raise ValueError(
            f"provide path to data set containing {lead_time}h deterministic [r]mse"
        )

    model_vars = model.input_coords()["variable"]
    if vars is None:
        vars = model_vars
    elif isinstance(vars, str):
        vars = [vars]

    # set noise for variables which shall not be perturbed to 0.
    skill = xr.open_dataset(skill_path)
    scale_vec = torch.Tensor(
        np.asarray(
            [
                skill.sel(channel=var, lead_time=lead_time)["value"].item()
                if var in vars
                else 0.0
                for var in model_vars
            ]
        )
    )

    return scale_vec.reshape(1, 1, 1, -1, 1, 1) * noise_amplification


def set_perturbation(
    model: PrognosticModel,
    data: DataSource,
    start_time: np.ndarray[np.datetime64],
    cfg: DictConfig,
) -> Perturbation:
    """
    Initialise perturbation method. Either choose and set up a bred vector method
    or instantiate any other perturbation defined in the config.

    Parameters
    ----------
    model : PrognosticModel
        forecast model.
    data : DataSource
        Data source from which to obtain ICs
    start_time : np.ndarray[np.datetime64]
        IC times
    cfg : DictConfig
        Hydra config object

    Returns
    -------
    Perturbation
        Perturbation method.
    """
    if "method" in cfg.perturbation:
        if cfg.perturbation.method == "hcbv":
            noise_amp_seed = get_noise_vector(
                model,
                skill_path=cfg.perturbation.skill_path,
                noise_amplification=cfg.perturbation.noise_amplification,
                vars=cfg.perturbation.perturbed_var,
            )
            noise_amp_iter = get_noise_vector(
                model,
                skill_path=cfg.perturbation.skill_path,
                noise_amplification=cfg.perturbation.noise_amplification,
            )

            seed_perturbation = CorrelatedSphericalGaussian(
                noise_amplitude=noise_amp_seed,
                sigma=1.0,
                length_scale=5.0e5,
                time_scale=48.0,
            )

            per = HemisphericCentredBredVector(
                model=model,
                data=data,
                time=start_time,
                noise_amplitude=noise_amp_iter,
                integration_steps=cfg.perturbation.integration_steps,  # use cfg.breeding_steps
                seeding_perturbation_method=seed_perturbation,
            )

        elif cfg.perturbation.method == "bv":
            per = BredVector(
                model=model,
                noise_amplitude=cfg.perturbation.noise_amplification,
                integration_steps=cfg.perturbation.integration_steps,  # use cfg.breeding_steps
                ensemble_perturb=True,
            )
    else:
        per = hydra.utils.instantiate(cfg.perturbation)

    return per


def build_package_list(cfg: DictConfig) -> list[str]:
    """
    Find all available model packages.

    Parameters
    ----------
    cfg : DictConfig
        Hydra config object

    Returns
    -------
    list[str]
        Available model packages.
    """
    if "package" in cfg.forecast_model:  # pointing to single package
        if cfg.forecast_model.package == "default":
            return ["default"]

        elif os.path.isfile(os.path.join(cfg.forecast_model.package, "config.json")):
            return [cfg.forecast_model.package]

        else:  # pointing to directory of packages
            max_num_ckpts = 29
            if "max_num_checkpoints" in cfg.forecast_model:
                max_num_ckpts = cfg.forecast_model.max_num_checkpoints
            packages = []
            for pkg in os.listdir(cfg.forecast_model.package):
                pth = os.path.abspath(os.path.join(cfg.forecast_model.package, pkg))
                if os.path.isdir(pth) and os.path.isfile(
                    os.path.join(pth, "config.json")
                ):
                    packages.append(pth)
            if len(packages) == 0:
                ValueError(
                    f"Found no valid model packages under {cfg.forecast_model.package}."
                )
            return (sorted(packages))[:max_num_ckpts]

    else:
        return ["default"]


def build_model_dict(cfg: DictConfig) -> dict:
    """
    Build a dictionary of loaded model, model class and package name.

    Parameters
    ----------
    cfg : DictConfig
        Hydra config object

    Returns
    -------
    dict
        Dictionary containing model, model class and model package.
    """
    return {
        "model": None,
        "class": hydra.utils.get_class(cfg.forecast_model.architecture),
        "package": None,
    }


def get_model(cfg: DictConfig) -> tuple[dict, list[str]]:
    """
    get a model dictionary and a list of available model packages.

    Parameters
    ----------
    cfg : DictConfig
        Hydra config object

    Returns
    -------
    tuple[dict, list[str]]
        Dictionary containing model, model class and model package.
    """
    return build_model_dict(cfg), build_package_list(cfg)


def set_initial_times(cfg: DictConfig) -> list[np.datetime64]:
    """
    build list of IC times.

    Parameters
    ----------
    cfg : DictConfig
        Hydra config object

    Returns
    -------
    list[np.datetime64]
        Dictionary containing model, model class and model package.
    """
    # list of ICs
    if "start_times" in cfg:
        if "ic_block_start" in cfg:
            raise ValueError(
                "either provide a list of start times or define a block, not both"
            )
        ics = to_time_array(sorted(cfg.start_times))

    # block of ICs
    else:
        ics = to_time_array([cfg.ic_block_start, cfg.ic_block_end])
        ics = np.arange(
            ics[0],
            ics[1] + np.timedelta64(cfg.ic_block_step, "h"),
            np.timedelta64(cfg.ic_block_step, "h"),
        )

    return ics


def initialise_output(
    cfg: DictConfig, time: np.datetime64, model_dict: dict
) -> IOBackend:
    """
    Initialise data output.

    Parameters
    ----------
    cfg : DictConfig
        Hydra config object
    time : np.datetime64
        IC time.
    model_dict:
        dictionary including prognostic model, its class and the name of its package.

    Returns
    -------
    IOBackend
        Object for storing data.
    """
    if cfg.mode == "cyclone_tracking":  # TODO allow variable output
        return None

    # Create the IO handler, store in memory
    if "path" not in cfg.file_output:
        with open_dict(cfg):
            cfg["file_output"]["path"] = "outputs/"

    pkg = model_dict["package"]
    if pkg != "vanilla":
        pkg = "_pkg_" + pkg.split("_")[-1]

    file_name = cfg.project + "_" + str(time)[:13] + pkg

    out_path = os.path.join(cfg.file_output.path, file_name)
    if DistributedManager().rank == 0:
        os.makedirs(cfg.file_output.path, exist_ok=True)

    io = hydra.utils.instantiate(cfg.file_output.format)
    if isinstance(io, partial):  # add out file names
        if cfg.file_output.format._target_.split(".")[-1].startswith("NetCDF"):
            file_name = out_path + ".nc"
        elif cfg.file_output.format._target_.split(".")[-1].startswith("Zarr"):
            file_name = out_path + ".zarr"
        else:
            raise ValueError(
                f"no file name extension implemented for {io}. It's a one-liner tho, do it quickly ;)"
            )
        io = io(file_name=file_name)

    return io


def pair_packages_ics(ics: list, model_packages: list, ensemble_size: int) -> list:
    """
    Pair initial conditions with model packages. In parallel setting, distribute among
    ranks.

    Parameters
    ----------
    ics : list
        Hydra config object
    model_packages : list
        List of available model packages.
    ensemble_size : int
        number of members in ensemble.

    Returns
    -------
    list
        IC - model package configs.
    """
    configs = [
        (pkg, ic, ii * ensemble_size)
        for ii, pkg in enumerate(model_packages)
        for ic in ics
    ]

    dist = DistributedManager()
    if dist.world_size > 1:
        if len(configs) % dist.world_size == 0:
            nconfigs_proc = len(configs) // dist.world_size
        else:
            nconfigs_proc = len(configs) // dist.world_size + 1

        idx = dist.rank * nconfigs_proc
        configs = configs[idx : min(idx + nconfigs_proc, len(configs))]

        if not len(configs) > 0:
            logger.warning(f"nothing to do for rank {dist.rank}. exiting.")
            exit()

    logger.info(
        f"rank {dist.rank}: predicting from following models/initial times: {configs}"
    )

    return configs


def set_random_seed(cfg: DictConfig) -> None:
    """
    if available, initialise with random seed

    Parameters
    ----------
    cfg : DictConfig
        Hydra config object
    """
    dist = DistributedManager()
    if "random_seed" in cfg:
        torch.manual_seed(cfg.random_seed + dist.rank)

    return


def initialise(cfg: DictConfig) -> tuple[list, dict, DataSource, OrderedDict]:
    """
    set initial conditions, load models and set up file output

    Parameters
    ----------
    cfg : DictConfig
        Hydra config object

    Returns
    -------
    tuple[list, dict, DataSource, OrderedDict]
        IC - model package configs.
    """
    if "project" not in cfg:
        raise ValueError("specify a project name in the config: project: project_name")

    set_random_seed(cfg)

    ics = set_initial_times(cfg)

    model_dict, model_packages = get_model(cfg)
    model_dict2 = update_model_dict(model_dict, "default")
    default_model = model_dict2["model"]
    lon_coords = default_model.output_coords(default_model.input_coords())["lon"]
    lat_coords = default_model.output_coords(default_model.input_coords())["lat"]

    if "file_output" in cfg:
        if "cropbox" in cfg["file_output"]:
            cfg_cropbox = cfg.file_output.cropbox

            if not (cfg_cropbox.lat_min >= -90):
                raise ValueError("lat_min needs to be >=-90")
            if not (cfg_cropbox.lat_max <= 90):
                raise ValueError("lat_max needs to be <=90")
            if not (cfg_cropbox.lat_min < cfg_cropbox.lat_max):
                raise ValueError("lat_min needs to be smaller than lat_max")
            if not (cfg_cropbox.lon_min < cfg_cropbox.lon_max):
                raise ValueError("lon_max needs to be larger than lon_min")
            if cfg_cropbox.lon_min < 0:
                if not (cfg_cropbox.lon_min >= -180):
                    raise ValueError("lon_min needs to be >= -180")
                if not (cfg_cropbox.lon_min < 180):
                    raise ValueError("lon_min needs to be < 180")
                if not (cfg_cropbox.lon_max <= 180):
                    raise ValueError("lon_max needs to be <= 180")
            else:
                if not (cfg_cropbox.lon_min >= 0):
                    raise ValueError("lon_min needs to be >= 0")
                if not (cfg_cropbox.lon_min < 360):
                    raise ValueError("lon_min needs to be < 360")
                if not (cfg_cropbox.lon_max <= 360):
                    raise ValueError("lon_max needs to be <= 360")

            # determine longitude ranges of input and output
            lon_range_model = (
                360 if ((lon_coords.max() > 180) and (lon_coords.min() >= 0)) else 180
            )
            lon_range_out = (
                360
                if ((cfg_cropbox.lon_max > 180) and (cfg_cropbox.lon_min >= 0))
                else 180
            )

            if lon_range_model != lon_range_out:
                # determine where to split for conversion
                if lon_range_model == 180 and lon_range_out == 360:
                    split_longitude = 0
                elif lon_range_model == 360 and lon_range_out == 180:
                    split_longitude = 180

                ind_below = (lon_coords < split_longitude).nonzero()[0]
                ind_above = (lon_coords >= split_longitude).nonzero()[0]
                if lon_range_model == 180 and lon_range_out == 360:
                    lon_coords = np.concatenate(
                        [lon_coords[ind_above], lon_coords[ind_below] + 360]
                    )
                elif lon_range_model == 360 and lon_range_out == 180:
                    lon_coords = np.concatenate(
                        [lon_coords[ind_above] - 360, lon_coords[ind_below]]
                    )

            # filter/crop area
            idx_lon = np.where(
                (lon_coords >= cfg_cropbox.lon_min)
                & (lon_coords <= cfg_cropbox.lon_max)
            )
            idx_lat = np.where(
                (lat_coords >= cfg_cropbox.lat_min)
                & (lat_coords <= cfg_cropbox.lat_max)
            )
            lat_coords = lat_coords[min(idx_lat[0]) : max(idx_lat[0]) + 1]
            lon_coords = lon_coords[min(idx_lon[0]) : max(idx_lon[0]) + 1]

    if cfg.mode != "cyclone_tracking":
        output_coords = (
            {"variable": np.array(cfg.file_output.output_vars)}
            if "output_vars" in cfg.file_output
            else {}
        )

    else:
        output_coords = {}
    output_coords["lon"] = lon_coords
    output_coords["lat"] = lat_coords

    ensemble_configs = pair_packages_ics(ics, model_packages, cfg.nensemble)
    data = hydra.utils.instantiate(cfg.data_source)

    return ensemble_configs, model_dict, data, OrderedDict(output_coords)


def update_model_dict(model_dict: dict, package: Package) -> dict:
    """
    check if model on GPU is same as needed for next inference.
    If not, load new model package and update model dict.

    Parameters
    ----------
    model_dict : dict
        dictionary specifying model, model class, and model package
    package : Package
        model package to be used in next inference

    Returns
    -------
    dict
        model dict.
    """
    if package != model_dict["package"]:
        model_dict["package"] = package
        package = (
            model_dict["class"].load_default_package()
            if package == "default"
            else Package(package)
        )
        model_dict["model"] = model_dict["class"].load_model(package=package)

    return model_dict


def store_tracks(tracks: pd.DataFrame, cfg: DictConfig) -> None:
    """
    method which writes cyclone tracks to file.

    Parameters
    ----------
    tracks : pd.DataFrame
        tabular data of cyclone tracks
    cfg : DictConfig
        Hydra config object
    """
    tracks = pd.concat(tracks)
    map_global = (
        tracks.groupby(["ic", "ens_member", "track_id"])
        .count()
        .reset_index()
        .reset_index()
        .rename(columns={"index": "track_id_global"})[
            ["track_id_global", "ic", "track_id", "ens_member"]
        ]
    )

    tracks = tracks.merge(map_global, on=["ic", "track_id", "ens_member"])[
        [
            "ic",
            "ens_member",
            "track_id_global",
            "vt",
            "point_number",
            "tc_lat",
            "tc_lon",
            "tc_msl",
            "tc_speed",
        ]
    ]
    cols_uint16 = ["point_number", "ens_member"]
    tracks[cols_uint16] = tracks[cols_uint16].astype("uint16")
    tracks = tracks.rename(columns={"track_id_global": "track_id"})
    tracks_file = (
        os.path.join(cfg.cyclone_tracking.out_dir, cfg.project)
        + f"_tracks_rank_{str(DistributedManager().rank).zfill(3)}.csv"
    )
    tracks.to_csv(tracks_file, index=False)

    return


def write_to_disk(
    cfg: DictConfig,
    ic: str,
    model_dict: dict,
    io: IOBackend,
    writer_threads: list[Future],
    writer_executor: ThreadPoolExecutor | None,
) -> tuple[list[Future], ThreadPoolExecutor | None]:
    """
    method which writes in-memory backends to file.

    Parameters
    ----------
    cfg : DictConfig
        config.
    ic : str
        initial condition.
    model_dict : dict
        dictionary containing loaded model, its class and its package
    io : IOBackend
        object for data output
    writer_threads : list[Future]
        threads for parallel file output
    writer_executor : ThreadPoolExecutor
        executor for parallel file output
    """
    if cfg.mode == "cyclone_tracking":
        return writer_threads, writer_executor
    pkg = model_dict["package"]
    if pkg != "vanilla":
        pkg = "_pkg_" + pkg.split("_")[-1]

    file_name = cfg.project + "_" + str(ic)[:13] + pkg

    out_path = os.path.join(cfg.file_output.path, file_name)

    kw_args = {"path": out_path + ".nc", "format": "NETCDF4"}

    if writer_executor is not None:
        if isinstance(io, XarrayBackend):
            writer_threads.append(writer_executor.submit(io.root.to_netcdf, **kw_args))
        elif isinstance(io, KVBackend):
            writer_threads.append(
                writer_executor.submit(io.to_xarray().to_netcdf, **kw_args)
            )
    else:
        if isinstance(io, XarrayBackend):
            io.root.to_netcdf(**kw_args)
        elif isinstance(io, KVBackend):
            io.to_xarray().to_netcdf(**kw_args)

    return writer_threads, writer_executor
