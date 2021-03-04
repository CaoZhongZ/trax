# coding=utf-8
# Copyright 2021 The Trax Authors.
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

# Lint as: python3
"""Tests for RL environments created from supervised data-sets."""

from absl.testing import absltest
from trax.rl.envs import data_envs


class SequenceDataEnvTest(absltest.TestCase):

  def test_copy_task_correct(self):
    """Test sequence data env on the copying task, correct replies."""
    env = data_envs.SequenceDataEnv(data_envs.copy_stream(2, n=1), 16)
    x1 = env.reset()
    x2, r0, d0, _ = env.step(0)
    self.assertEqual(r0, 0.0)
    self.assertEqual(d0, False)
    eos, r1, d1, _ = env.step(0)
    self.assertEqual(eos, 1)
    self.assertEqual(r1, 0.0)
    self.assertEqual(d1, False)
    y1, r2, d2, _ = env.step(x1)
    self.assertEqual(y1, x1)
    self.assertEqual(r2, 0.0)
    self.assertEqual(d2, False)
    y2, r3, d3, _ = env.step(x2)
    self.assertEqual(y2, x2)
    self.assertEqual(r3, 0.0)
    self.assertEqual(d3, False)
    eos2, r4, d4, _ = env.step(1)
    self.assertEqual(eos2, 1)
    self.assertEqual(r4, 1.0)
    self.assertEqual(d4, True)

  def test_copy_task_mixed(self):
    """Test sequence data env on the copying task, mixed replies."""
    env = data_envs.SequenceDataEnv(data_envs.copy_stream(2, n=2), 16)
    x1 = env.reset()
    x2, _, _, _ = env.step(0)
    eos, _, _, _ = env.step(0)
    _, _, _, _ = env.step(x1)
    _, _, _, _ = env.step(x2 + 1)  # incorrect
    x1, r1, d1, _ = env.step(1)
    self.assertEqual(r1, 0.5)
    self.assertEqual(d1, False)
    x2, _, _, _ = env.step(0)
    eos, _, _, _ = env.step(0)
    _, _, _, _ = env.step(x1 + 1)  # incorrect
    _, _, _, _ = env.step(x2 + 1)  # incorrect
    eos, r2, d2, _ = env.step(1)
    self.assertEqual(eos, 1)
    self.assertEqual(r2, 0.0)
    self.assertEqual(d2, True)


if __name__ == '__main__':
  absltest.main()
