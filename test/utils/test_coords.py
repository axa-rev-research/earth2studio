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

import numpy as np
import pytest
import torch

# from earth2studio.utils.coords import map_coords
from coords import map_coords

from earth2studio.utils import handshake_dim


@pytest.mark.parametrize(
    "coords",
    [
        OrderedDict([("batch", []), ("variable", []), ("lat", []), ("lon", [])]),
        OrderedDict([("time", []), ("lat", []), ("lon", [])]),
    ],
)
def test_handshake_dim(coords):
    # Check dims no index
    for dim in list(coords.keys()):
        handshake_dim(coords, dim)
    # Check dims with index
    for i, dim in enumerate(list(coords.keys())):
        handshake_dim(coords, dim, i)
    # Check dims with reverse index
    for i, dim in enumerate(list(coords.keys())[::-1]):
        handshake_dim(coords, dim, -(i + 1))


@pytest.mark.parametrize(
    "coords",
    [
        OrderedDict([("a", []), ("b", []), ("lat", []), ("lon", [])]),
        OrderedDict([("lat", []), ("lon", [])]),
    ],
)
def test_handshake_dim_failure(coords):

    with pytest.raises(KeyError):
        handshake_dim(coords, "fake_dim")

    with pytest.raises(ValueError):
        handshake_dim(coords, "lat", -1)

    with pytest.raises(ValueError):
        handshake_dim(coords, "lat", 5)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_map_nearest(device):
    coords = OrderedDict(
        [("variable", np.array(["a", "b", "c"])), ("lat", np.array([1, 2, 3]))]
    )
    data = torch.randn(3, 3).to(device)

    # No change
    out, outc = map_coords(data, coords, coords)
    assert torch.allclose(out, data)
    assert np.all(outc["variable"] == outc["variable"])

    # Select slice in 1D
    out, outc = map_coords(data, coords, OrderedDict([("variable", np.array(["a"]))]))
    assert torch.allclose(out, data[:1])
    assert np.all(outc["variable"] == np.array(["a"]))

    # Select slice in 1D
    out, outc = map_coords(
        data, coords, OrderedDict([("batch", None), ("variable", np.array(["b", "c"]))])
    )
    assert torch.allclose(out, data[1:])
    assert np.all(outc["variable"] == np.array(["b", "c"]))

    # Select slice in 2D
    out, outc = map_coords(
        data,
        coords,
        OrderedDict([("variable", np.array(["b", "c"])), ("lat", np.array([1]))]),
    )
    assert torch.allclose(out, data[1:, :1])

    # Select index 1D
    out, outc = map_coords(data, coords, OrderedDict([("lat", np.array([1, 3]))]))
    assert torch.allclose(out, torch.cat([data[:, :1], data[:, 2:]], dim=-1))
    assert np.all(outc["lat"] == np.array([1, 3]))

    # Select index 2D
    out, outc = map_coords(
        data,
        coords,
        OrderedDict([("variable", np.array(["a", "c"])), ("lat", np.array([1, 3]))]),
    )
    assert out.shape == torch.Size((2, 2))
    assert np.all(outc["variable"] == np.array(["a", "c"]))
    assert np.all(outc["lat"] == np.array([1, 3]))

    # Select index 1D reverse
    out, outc = map_coords(
        data, coords, OrderedDict([("variable", np.array(["c", "a"]))])
    )
    truth = torch.cat((data[-1:], data[:1]), dim=0)
    assert torch.allclose(out, truth)
    assert np.all(outc["variable"] == np.array(["c", "a"]))

    out, outc = map_coords(
        data,
        coords,
        OrderedDict([("variable", np.array(["b", "c"])), ("lat", np.array([1, 2]))]),
    )
    assert torch.allclose(out, data[1:, :2])

    out, outc = map_coords(
        data,
        coords,
        OrderedDict(
            [("variable", np.array(["b", "c"])), ("lat", np.array([1.1, 2.4]))]
        ),
    )
    assert torch.allclose(out, data[1:, :2])

    out, outc = map_coords(
        data,
        coords,
        OrderedDict([("variable", np.array(["c"])), ("lat", np.array([1.8, 2.4]))]),
    )
    assert torch.allclose(out, torch.stack([data[2:, 1], data[2:, 1]], dim=1))
    # Test out of bounds of coordinate system
    out, outc = map_coords(data, coords, OrderedDict([("lat", np.array([1.8, 4.0]))]))
    assert torch.allclose(out, data[:, 1:])

    out, outc = map_coords(data, coords, OrderedDict([("lat", np.array([-0.1, 1.6]))]))
    assert torch.allclose(out, data[:, :2])


def test_map_errors():
    coords = OrderedDict(
        [("variable", np.array(["a", "b", "c"])), ("lat", np.array([1, 2, 3]))]
    )
    data = torch.arange(0, 9).reshape((3, 3))

    with pytest.raises(KeyError):
        map_coords(data, coords, OrderedDict([("foo", np.array(["c"]))]))

    with pytest.raises(ValueError):
        map_coords(data, coords, OrderedDict([("variable", np.array(["d"]))]))


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_longitude_mapping_forward(device):
    lat_values = np.arange(-90, 90, 0.25)
    lon_values = np.arange(0, 360, 0.25)
    lon_values_out = np.arange(-180, 180, 0.25)
    variable_values = np.array(["a", "b", "c"])
    coords_in = OrderedDict(
        [("variable", variable_values), ("lat", lat_values), ("lon", lon_values)]
    )
    coords_out = OrderedDict(
        [("variable", variable_values), ("lat", lat_values), ("lon", lon_values_out)]
    )
    data = torch.arange(0, 360, 0.25).repeat(3, len(lat_values), 1).to(device)
    out, outc = map_coords(data, coords_in, coords_out)
    assert out[:, :, 0].unique() == torch.tensor(180.0)
    assert out[:, :, -1].unique() == torch.tensor(179.75)
    assert out[:, :, 720].unique() == torch.tensor(0)
    assert out[:, :, 719].unique() == torch.tensor(359.75)
    assert outc["lon"][0] == -180.0
    assert outc["lon"][-1] == 179.75
    assert outc["lon"][719] == -0.25
    assert outc["lon"][720] == 0.0


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("lon_range", [(10, 20), (-10, 10), (-180, -170)])
def test_longitude_mapping_forward_subregion(lon_range, device):
    lon_min, lon_max = lon_range
    lat_values = np.arange(-90, 90, 0.25)
    lon_values = np.arange(0, 360, 0.25)
    lon_values_out = np.arange(lon_min, lon_max, 0.25)
    variable_values = np.array(["a", "b", "c"])
    coords_in = OrderedDict(
        [("variable", variable_values), ("lat", lat_values), ("lon", lon_values)]
    )
    coords_out = OrderedDict(
        [("variable", variable_values), ("lat", lat_values), ("lon", lon_values_out)]
    )
    data = torch.arange(0, 360, 0.25).repeat(3, len(lat_values), 1).to(device)
    out, outc = map_coords(data, coords_in, coords_out)
    if lon_range == (10, 20):
        assert out.shape == torch.Size((3, 720, 40))
        assert out[:, :, 0].unique() == torch.tensor(10.0)
        assert out[:, :, -1].unique() == torch.tensor(19.75)
        assert outc["lon"][0] == 10.0
        assert outc["lon"][-1] == 19.75
    elif lon_range == (-10, 10):
        assert out.shape == torch.Size((3, 720, 80))
        assert out[:, :, 0].unique() == torch.tensor(350.0)
        assert out[:, :, -1].unique() == torch.tensor(9.75)
        assert outc["lon"][0] == -10.0
        assert outc["lon"][-1] == 9.75
    elif lon_range == (-180, -170):
        assert out.shape == torch.Size((3, 720, 40))
        assert out[:, :, 0].unique() == torch.tensor(180.0)
        assert out[:, :, -1].unique() == torch.tensor(189.75)
        assert outc["lon"][0] == -180.0
        assert outc["lon"][-1] == -170.25


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_longitude_mapping_backward(device):
    lat_values = np.arange(-90, 90, 0.25)
    lon_values_out = np.arange(0, 360, 0.25)
    lon_values = np.arange(-180, 180, 0.25)
    variable_values = np.array(["a", "b", "c"])
    coords_in = OrderedDict(
        [("variable", variable_values), ("lat", lat_values), ("lon", lon_values)]
    )
    coords_out = OrderedDict(
        [("variable", variable_values), ("lat", lat_values), ("lon", lon_values_out)]
    )
    data = torch.arange(-180, 180, 0.25).repeat(3, len(lat_values), 1).to(device)
    out, outc = map_coords(data, coords_in, coords_out)
    assert out[:, :, 0].unique() == torch.tensor(0.0)
    assert out[:, :, -1].unique() == torch.tensor(-0.25)
    assert out[:, :, 720].unique() == torch.tensor(-180.0)
    assert out[:, :, 719].unique() == torch.tensor(179.75)
    assert outc["lon"][0] == 0.0
    assert outc["lon"][-1] == 359.75
    assert outc["lon"][719] == 179.75
    assert outc["lon"][720] == 180.0
