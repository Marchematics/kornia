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

from kornia.feature import DenseSIFTDescriptor, SIFTDescriptor


def test_dense_sift_pooling_kernel_follows_to():
    descriptor = DenseSIFTDescriptor().to(torch.float64)

    kernel = descriptor.get_pooling_kernel()

    assert kernel.dtype == torch.float64
    assert kernel.device == descriptor._bin_pooling_kernel_weight.device


def test_dense_sift_pooling_kernel_is_a_copy():
    descriptor = DenseSIFTDescriptor()
    expected = descriptor._bin_pooling_kernel_weight.clone()

    descriptor.get_pooling_kernel().zero_()

    assert torch.equal(descriptor._bin_pooling_kernel_weight, expected)


def test_sift_kernel_accessors_return_copies():
    descriptor = SIFTDescriptor(16)
    pooling = descriptor.pk.weight.detach().clone()
    weighting = descriptor.gk.detach().clone()

    descriptor.get_pooling_kernel().zero_()
    descriptor.get_weighting_kernel().zero_()

    assert torch.equal(descriptor.pk.weight, pooling)
    assert torch.equal(descriptor.gk, weighting)
