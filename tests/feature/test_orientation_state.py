# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright 2018 Kornia Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import torch

from kornia.feature.orientation import PatchDominantGradientOrientation


def test_orientation_forward_does_not_mutate_smoother():
    module = PatchDominantGradientOrientation(13)
    weight = module.angular_smooth.weight.detach().clone()

    out = module(torch.rand(2, 1, 13, 13, dtype=torch.float64))

    assert out.dtype == torch.float64
    assert module.angular_smooth.weight.dtype == torch.float32
    assert torch.equal(module.angular_smooth.weight, weight)


def test_orientation_mixed_dtype_calls_keep_module_state():
    module = PatchDominantGradientOrientation(13)

    module(torch.rand(1, 1, 13, 13, dtype=torch.float64))
    out = module(torch.rand(1, 1, 13, 13, dtype=torch.float32))

    assert out.dtype == torch.float32
    assert module.angular_smooth.weight.dtype == torch.float32
