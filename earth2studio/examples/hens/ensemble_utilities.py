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

from collections import OrderedDict
from collections.abc import Iterator
from datetime import datetime
from math import ceil

import numpy as np
import pandas as pd
import torch
from loguru import logger
from tqdm import tqdm

from earth2studio.data import DataSource, fetch_data
from earth2studio.enterprise.models.dx.cyclone_tracking import get_tracks_from_positions
from earth2studio.io import IOBackend
from earth2studio.models.dx import DiagnosticModel
from earth2studio.models.px import PrognosticModel
from earth2studio.perturbation import Perturbation
from earth2studio.utils.coords import CoordSystem, map_coords, split_coords
from earth2studio.utils.time import to_time_array

logger.remove()
logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)


class EnsembleBase:
    """
    Ensemble inference pipeline with options to add a diagnostic
    model or tropical cyclone tracking in the loop.

    Parameters
    ----------
    time : list[str] | list[datetime] | list[np.datetime64]
        IC times.
    nsteps : int
        number of forecast steps.
    nensemble : int
        ensemble size.
    prognostic : PrognosticModel
        forecast model.
    data : DataSource
        Data source from which to obtain ICs
    io : IOBackend
        Data object for storing generated data.
    perturbation : Perturbation
        Method for perturbing initial conditions.
    diagnostic : DiagnosticModel, optional
        diagnostic model
    batch_size : int, optional
        batch size
    output_coords : CoordSystem, optional
        coords of data that shall be stored.
    device : torch.device, optional
        device on which to run inference
    ensemble_idx_base : int, optional
        initial value for counting ensemble members
    """

    def __init__(
        self,
        time: list[str] | list[datetime] | list[np.datetime64],
        nsteps: int,
        nensemble: int,
        prognostic: PrognosticModel,
        data: DataSource,
        io: IOBackend,
        perturbation: Perturbation,
        diagnostic: DiagnosticModel | None = None,
        batch_size: int | None = None,
        output_coords: CoordSystem = OrderedDict({}),
        device: torch.device | None = None,
        ensemble_idx_base: int = 0,
    ) -> None:

        logger.info("Setting up HENS.")

        self.io = io
        self.nensemble = nensemble
        self.ensemble_idx = ensemble_idx_base
        self.nsteps = nsteps
        self.output_coords = output_coords
        self.perturbation = perturbation

        # Load model onto the device
        self.move_models_to_device(prognostic, diagnostic, device)

        # Fetch data from data source and load onto device
        self.fetch_ics(data=data, time=time)

        # Set up IO backend with information from output_coords (if applicable).
        self.setup_data_output()

        # Compute batch sizes
        self.set_batch_size(batch_size)

        return

    def move_models_to_device(
        self,
        prognostic: PrognosticModel,
        diagnostic: DiagnosticModel | None = None,
        device: torch.device | None = None,
    ) -> None:
        """
        Move models to device and obtain their input coordinates.

        Parameters
        ----------
        prognostic : PrognosticModel
            forecast model.
        diagnostic : DiagnosticModel | None
            diagnostic model [optional]
        device : torch.device
            device on which to run inference
        """
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        logger.info(f"Inference device: {self.device}")
        self.prognostic = prognostic.to(self.device)
        self.prognositc_ic = prognostic.input_coords()

        if diagnostic is not None:
            self.diagnostic = diagnostic
            self.diagnostic.to(self.device)
            self.diagnostic_ic = self.diagnostic.input_coords()

        return

    def fetch_ics(
        self,
        data: DataSource,
        time: list[str] | list[datetime] | list[np.datetime64],
    ) -> None:
        """
        Fetch initial conditions.

        Parameters
        ----------
        data : DataSource
            Data source from which to obtain ICs
        time : list[str] | list[datetime] | list[np.datetime64]
            IC times
        """
        self.time = to_time_array(time)
        self.x0, self.coords0 = fetch_data(
            source=data,
            time=time,
            variable=self.prognositc_ic["variable"],
            lead_time=self.prognositc_ic["lead_time"],
            device="cpu",
        )
        logger.success(f"Fetched data from {data.__class__.__name__}")

        return

    def setup_data_output(self) -> None:
        """
        Assemble output coords and initialise IO backend with coords.
        """

        # assemble output coords from fetched IC coords and ensemble IDs
        total_coords = {
            "ensemble": np.arange(self.nensemble) + self.ensemble_idx
        } | self.coords0.copy()

        # add lead time dimension
        total_coords["lead_time"] = np.asarray(
            [
                self.prognostic.output_coords(self.prognostic.input_coords())[
                    "lead_time"
                ]
                * ii
                for ii in range(self.nsteps + 1)
            ]
        ).flatten()

        # augment and overwrite total coords with dimensions of output coords
        for key, value in total_coords.items():
            total_coords[key] = self.output_coords.get(key, value)

        # initialise place for variables in io backend
        variables_to_save = total_coords.pop("variable")
        if (
            self.io is not None
        ):  # cyclone tracker still missing field output, to be fixed
            self.io.add_array(total_coords, variables_to_save)

        return

    def set_batch_size(self, batch_size: int | None = None) -> None:
        """
        calculate batch size and number of mini batches to inference.

        Parameters
        ----------
        batch_size : int
            targeted batch size
        """
        if batch_size is None:
            batch_size = self.nensemble
        self.batch_size = min(self.nensemble, batch_size)
        self.number_of_batches = ceil(self.nensemble / self.batch_size)

    def prep_loop(self, batch_id: int) -> tuple[Iterator, int]:
        """
        preparing mini batch for inference by setting ensemble IDs, perturbing
        ICs and creating the inference iterator of the prognostic model.

        Parameters
        ----------
        batch_id : int
            mini batch index

        Returns
        -------
        tuple[Iterator, int]
            Tuple containing iterator of prognostic model and mini batch size.
        """

        # Get fresh batch data
        xx = self.x0.to(self.device)

        # calculate mini batch size and define coords for ensemble
        mini_batch_size = min(
            self.batch_size, self.nensemble - batch_id * self.batch_size
        )
        coords = {
            "ensemble": np.arange(
                batch_id * self.batch_size, batch_id * self.batch_size + mini_batch_size
            )
            + self.ensemble_idx
        } | self.coords0.copy()

        # Unsqueeze xx for batching ensemble
        xx = xx.unsqueeze(0).repeat(mini_batch_size, *([1] * xx.ndim))

        # Map lat and lon if needed
        xx, coords = map_coords(xx, coords, self.prognositc_ic)

        # Perturb ensemble
        xx, coords = self.perturbation(xx, coords)

        # Create prognostic iterator
        model = self.prognostic.create_iterator(xx, coords)

        return model, mini_batch_size

    @torch.inference_mode()
    def __call__(self) -> IOBackend:
        """
        Run ensemble inference pipeline saving specified variables.

        Returns
        -------
        IOBackend
            io object containing data of ensemble inference
        """
        logger.info(
            f"Starting {self.nensemble} Member Ensemble inference with"
            + f" {self.number_of_batches} number of batches."
        )
        for batch_id in tqdm(
            range(0, self.number_of_batches),
            total=self.number_of_batches,
            desc="Total Ensemble Batches",
        ):
            model, nsamples = self.prep_loop(batch_id)
            with tqdm(
                total=self.nsteps + 1,
                desc=f"Inferencing batch {batch_id+1} of {self.number_of_batches} ({nsamples} samples)",
                leave=False,
            ) as pbar:
                for step, (xx, coords) in enumerate(model):
                    xx, coords = map_coords(xx, coords, self.output_coords)
                    self.io.write(*split_coords(xx, coords))
                    pbar.update(1)
                    if step == self.nsteps:
                        break

        logger.success("Inference complete")

        return self.io


def cat_coords(
    xx: torch.Tensor,
    cox: CoordSystem,
    yy: torch.Tensor,
    coy: CoordSystem,
    dim: str = "variable",
) -> tuple[torch.Tensor, CoordSystem]:
    """
    concatenate data along coordinate dimension.

    Parameters
    ----------
    xx : torch.Tensor
        First input tensor which to concatenate
    cox : CoordSystem
        Ordered dict representing coordinate system that describes xx
    yy : torch.Tensor
        Second input tensor which to concatenate
    coy : CoordSystem
        Ordered dict representing coordinate system that describes yy
    dim : str
        name of dimension along which to concatenate

    Returns
    -------
    tuple[torch.Tensor, CoordSystem]
        Tuple containing output tensor and coordinate OrderedDict from
        concatenated data.
    """

    if dim not in cox:
        raise ValueError(f"dim {dim} is not in coords: {list(cox)}.")
    if dim not in coy:
        raise ValueError(f"dim {dim} is not in coords: {list(coy)}.")

    # fix difference in latitude
    _cox = cox.copy()
    _cox["lat"] = coy["lat"]
    xx, cox = map_coords(xx, cox, _cox)

    coords = cox.copy()
    dim_index = list(coords).index(dim)

    zz = torch.cat((xx, yy), dim=dim_index)
    coords[dim] = np.append(cox[dim], coy[dim])

    return zz, coords


def squeeze_coord(
    xx: torch.Tensor, inco: CoordSystem, dim: str
) -> tuple[torch.Tensor, CoordSystem]:
    """
    remove a coordinate dimension of length 1.

    Parameters
    ----------
    xx : torch.Tensor
        Input tensor
    inco : CoordSystem
        Ordered dict representing coordinate system that describes xx
    dim : str
        name of dimension along which to concatenate

    Returns
    -------
    tuple[torch.Tensor, CoordSystem]
        Tuple containing output tensor and coordinate OrderedDict from
        concatenated data.
    """
    idx = list(inco).index(dim)
    ouco = inco.copy()
    ouco.pop(dim)
    if xx.shape[idx] != 1:
        raise ValueError(
            "cannot remove dimension with len>1,"
            + f" dim {dim} has length {xx.shape[idx]}"
        )

    return xx.squeeze(idx), ouco


class EnsembleDiagnostics(EnsembleBase):
    """
    Ensemble inference pipeline with diagnostic model on top.
    """

    @torch.inference_mode()
    def __call__(self) -> IOBackend:
        """
        Run ensemble inference pipeline with diagnostic model on top
        saving specified variables.

        Returns
        -------
        IOBackend
            io object containing data of ensemble inference.
        """
        self.diagnostic_ic.pop("batch")
        logger.info(
            f"Starting {self.nensemble} Member Ensemble inference with"
            + f" {self.number_of_batches} number of batches."
        )
        for batch_id in tqdm(
            range(0, self.number_of_batches),
            total=self.number_of_batches,
            desc="Total Ensemble Batches",
        ):
            model, nsamples = self.prep_loop(batch_id)
            with tqdm(
                total=self.nsteps + 1,
                desc=f"Inferencing batch {batch_id} ({nsamples} samples)",
                leave=False,
            ) as pbar:
                for step, (xx, coords) in enumerate(model):

                    # select input vars, remove lead time dim and apply diagnostic model
                    yy, codia = map_coords(xx, coords, self.diagnostic_ic)
                    yy, codib = squeeze_coord(yy, codia, "lead_time")
                    yy, codib = self.diagnostic(yy, codib)

                    # add lead time dim and concatenate diagnostic variable to forecast vars
                    codia["variable"] = codib["variable"]
                    yy = yy.unsqueeze(2)
                    xx, coords = cat_coords(xx, coords, yy, codia, "variable")

                    # pass output variables to io backend
                    xx, coords = map_coords(xx, coords, self.output_coords)
                    self.io.write(*split_coords(xx, coords))
                    pbar.update(1)
                    if step == self.nsteps:
                        break

        logger.success("Inference complete")

        return self.io


# TODO currently, io is not used. add optional io or add option to pass None in ini
class EnsembleCycloneTracking(EnsembleBase):
    """
    Ensemble inference pipeline with tropical cyclone identification and tracking.
    """

    @torch.inference_mode()
    def __call__(self) -> pd.DataFrame:
        """
        Run ensemble inference pipeline with tropical cyclone identification and tracking.

        Returns
        -------
        pd.DataFrame
            Tabular data containing information about cyclone tracks.
        """
        logger.info(
            f"Starting {self.nensemble} Member Ensemble CYCLONE TRACKING with"
            + f" {self.number_of_batches} number of batches."
        )

        # determine longitude ranges of input and output
        lon_range_in = (
            360
            if (
                (self.diagnostic_ic["lon"].max() > 180)
                and (self.diagnostic_ic["lon"].min() >= 0)
            )
            else 180
        )
        lon_range_out = (
            360
            if (
                (self.output_coords["lon"].max() > 180)
                and (self.output_coords["lon"].min() >= 0)
            )
            else 180
        )

        tracks = []
        for batch_id in tqdm(
            range(0, self.number_of_batches),
            total=self.number_of_batches,
            desc="Total Ensemble Batches",
        ):
            model, nsamples = self.prep_loop(batch_id)

            with tqdm(
                total=self.nsteps + 1,
                desc=f"Inferencing batch {batch_id} ({nsamples} samples)",
                leave=False,
            ) as pbar:
                cntrs, crds = [], None
                for step, (xx, coords) in enumerate(model):
                    tt, track_coords = map_coords(xx, coords, self.diagnostic_ic)
                    cntrs.append(tt)

                    if crds is None:
                        crds = track_coords
                    else:
                        crds["time"] = np.append(
                            crds["time"],
                            track_coords["time"][0] + track_coords["lead_time"][0],
                        )

                    pbar.update(1)
                    if step == self.nsteps:
                        break

                cntrs = torch.cat(cntrs, dim=-5)
                tt, track_coords = self.diagnostic(cntrs, crds)

                del track_coords["lead_time"]
                mem_ids = track_coords.pop("ensemble")
                tt = tt.squeeze()

                # member_id = coords['ensemble'][0]
                for mem in range(tt.shape[0]):
                    tt_mem = tt[mem]
                    # adjust longitude range
                    if lon_range_in != lon_range_out:
                        if lon_range_in == 360 and lon_range_out == 180:
                            tt_mem[:, 1, :] = ((tt_mem[:, 1, :] + 180) % 360) - 180
                        elif lon_range_in == 180 and lon_range_out == 360:
                            tt_mem[:, 1, :] = tt_mem[:, 1, :] % 360
                    # filter area
                    c1 = tt_mem[:, 1, :] >= self.output_coords["lon"].min()
                    c2 = tt_mem[:, 1, :] <= self.output_coords["lon"].max()
                    c3 = tt_mem[:, 0, :] >= self.output_coords["lat"].min()
                    c4 = tt_mem[:, 0, :] <= self.output_coords["lat"].max()
                    tt_mem_filtered = torch.where(
                        (c1 & c2 & c3 & c4).repeat(4, 1, 1).swapaxes(1, 0),
                        tt_mem,
                        np.nan,
                    )

                    tracks_df = get_tracks_from_positions(tt_mem_filtered, track_coords)
                    tracks_df.insert(
                        0, "ens_member", [mem_ids[mem]] * tracks_df.shape[0]
                    )
                    tracks.append(tracks_df)

                    # member_id += 1

        logger.success("Inference complete")
        df = pd.concat(tracks).reset_index(drop=True)
        return df
