# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

import time
import unittest
import random
import numpy as np
import os
import shutil

import paddle
import paddle.nn as nn
import paddle.utils as utils
import paddle.static as static
import paddle.nn.functional as F
import paddle.distributed.auto_parallel as auto

import paddle.fluid as fluid

from paddle.distributed import fleet
from paddle.fluid.initializer import NumpyArrayInitializer
from paddle.distributed.auto_parallel.utils import save_distributed_checkpoint, load_distributed_checkpoint, load_checkpoint_into_program
from paddle.distributed.auto_parallel.utils import get_dist_attr, merge_and_slice_parameter, load_parameter_into_program
from args_mlp import parse_args
import logging


logging.getLogger().setLevel(logging.INFO)
paddle.enable_static()
_global_parallel_strategy = None
_global_process_mesh = None

np.set_printoptions(suppress=True)

def load_checkpoint(checkpoint_path: str, main_program):
    print(f"checkpoint_path={checkpoint_path}")
    latest_step_dir = max(os.listdir(checkpoint_path), key=lambda x: int(x.split("_")[1]))
    latest_step = int(latest_step_dir.split("_")[1])
    latest_ckpt_dir = os.path.join(checkpoint_path, latest_step_dir)
    if os.path.isdir(latest_ckpt_dir):
        ckpt_file_list = [os.path.join(latest_ckpt_dir, ckpt_dir) for ckpt_dir in os.listdir(latest_ckpt_dir) if "model_state" in ckpt_dir]
        dist_attr_list = [os.path.join(latest_ckpt_dir, attr_dir) for attr_dir in os.listdir(latest_ckpt_dir) if "dist_attr" in attr_dir]
        print(f"=> loading checkpoint from: {latest_ckpt_dir}, ckpt_file_list={ckpt_file_list}")
        print(f"=> loading attribution from: {latest_ckpt_dir}, dist_attr_list={dist_attr_list}")
        load_checkpoint_into_program(ckpt_file_list, dist_attr_list, main_program)
        print(f"=> loaded checkpoint from: {latest_ckpt_dir}")
    return latest_step


class MLPLayer(nn.Layer):
    def __init__(self,
                 hidden_size=128,
                 intermediate_size=4 * 128,
                 initializer_range=0.02):
        super(MLPLayer, self).__init__()
        d_model = hidden_size
        dim_feedforward = intermediate_size
        np.random.seed(2021)
        arr0 = np.random.normal(0, 0.02, size=(d_model, dim_feedforward))
        arr1 = np.random.normal(0, 0.02, size=(dim_feedforward, d_model))
        weight_attr0 = paddle.ParamAttr(initializer=NumpyArrayInitializer(arr0))
        weight_attr1 = paddle.ParamAttr(initializer=NumpyArrayInitializer(arr1))
        bias_attr = None
        self.linear0 = nn.Linear(
            d_model, dim_feedforward, weight_attr0, bias_attr=bias_attr)
        self.linear1 = nn.Linear(
            dim_feedforward, d_model, weight_attr1, bias_attr=bias_attr)
        self.linear2 = nn.Linear(
            d_model, dim_feedforward, weight_attr0, bias_attr=bias_attr)
        self.linear3 = nn.Linear(
            dim_feedforward, d_model, weight_attr1, bias_attr=bias_attr)
        self.linear4 = nn.Linear(
            d_model, dim_feedforward, weight_attr0, bias_attr=bias_attr)
        self.linear5 = nn.Linear(
            dim_feedforward, d_model, weight_attr1, bias_attr=bias_attr)
        self.norm0 = nn.LayerNorm(d_model, epsilon=1e-5)
        self.norm1 = nn.LayerNorm(d_model, epsilon=1e-5)
        self.norm2 = nn.LayerNorm(d_model, epsilon=1e-5)

    def forward(self, input):
        if _global_parallel_strategy == "dp_mp":
            auto.shard_tensor(
                self.linear0.weight,
                dist_attr={
                    "process_mesh": _global_process_mesh,
                    "dims_mapping": [-1, 1]
                })
            auto.shard_tensor(
                self.linear1.weight,
                dist_attr={
                    "process_mesh": _global_process_mesh,
                    "dims_mapping": [1, -1]
                })
            auto.shard_tensor(
                self.linear2.weight,
                dist_attr={
                    "process_mesh": _global_process_mesh,
                    "dims_mapping": [-1, 1]
                })
            auto.shard_tensor(
                self.linear3.weight,
                dist_attr={
                    "process_mesh": _global_process_mesh,
                    "dims_mapping": [1, -1]
                })
            auto.shard_tensor(
                self.linear4.weight,
                dist_attr={
                    "process_mesh": _global_process_mesh,
                    "dims_mapping": [-1, 1]
                })
            auto.shard_tensor(
                self.linear5.weight,
                dist_attr={
                    "process_mesh": _global_process_mesh,
                    "dims_mapping": [1, -1]
                })

        out = self.norm0(input)
        out = self.linear0(out)
        out = F.gelu(out, approximate=True)
        out = self.linear1(out)

        out = self.norm1(out)
        out = self.linear2(out)
        out = F.gelu(out, approximate=True)
        out = self.linear3(out)

        out = self.norm2(out)
        out = self.linear4(out)
        out = F.gelu(out, approximate=True)
        out = self.linear5(out)
        return out


def mlp_forward(args, train_program, start_program):
    with static.program_guard(train_program,start_program), \
        utils.unique_name.guard():
        hidden_size = 128
        input = static.data(
            name="input", shape=[args.global_batch_size, hidden_size], dtype='float32')
        label = static.data(
            name="label", shape=[args.global_batch_size, 1], dtype='float32')
        input.stop_gradient=False

        if _global_parallel_strategy == "dp_mp":
            auto.shard_tensor(
                input,
                dist_attr={
                    "process_mesh": _global_process_mesh,
                    "dims_mapping": [0, -1]
                })

        if _global_parallel_strategy == "dp":
            auto.shard_tensor(
                input,
                dist_attr={
                    "process_mesh": _global_process_mesh,
                    "dims_mapping": [0, -1]
                })


        mlp = MLPLayer(
            hidden_size=hidden_size,
            intermediate_size=4 * hidden_size,
            initializer_range=0.02)
        predict = mlp(input)
        error_cost = paddle.nn.functional.square_error_cost(predict, label)
        loss = paddle.mean(error_cost)
    return loss, train_program, start_program


def get_distributed_program(args):
    train_program = static.Program()
    startup_program = static.Program()
    dist_strategy = fleet.DistributedStrategy()
    dist_strategy.semi_auto = True
    dist_strategy.gradient_merge = True
    dist_strategy.gradient_merge_configs = {"k_steps": 4, "avg": True}
    fleet.init(is_collective=True, strategy=dist_strategy)

    loss, train_program, startup_program = mlp_forward(args, train_program,
                                                       startup_program)

    with open(args.output_dir + "/serial_main_program.txt.%d" % (paddle.distributed.get_rank()), 'w') as f:
        f.write(str(train_program))
    with open(args.output_dir + "/serial_startup_program.txt.%d" % (paddle.distributed.get_rank()), 'w') as f:
        f.write(str(startup_program))

    optimizer = paddle.fluid.optimizer.SGDOptimizer(learning_rate=0.01)
    #optimizer = paddle.fluid.optimizer.Adam(learning_rate=0.01)
    optimizer = fleet.distributed_optimizer(optimizer)
    _, _, dist_startup_prog, dist_main_prog = optimizer.minimize(
        loss, startup_program)

    with open(args.output_dir + "/main_program.txt.%d" % (paddle.distributed.get_rank()), 'w') as f:
        f.write(str(paddle.fluid.default_main_program()))
    with open(args.output_dir + "/startup_program.txt.%d" % (paddle.distributed.get_rank()), 'w') as f:
        f.write(str(paddle.static.default_startup_program()))

    return dist_main_prog, dist_startup_prog, loss


def main(args):
    paddle.seed(2021)
    random.seed(2021)
    np.random.seed(2021)
    fluid.default_main_program().random_seed = 2021
    fluid.default_startup_program().random_seed = 2021

    global _global_parallel_strategy
    global _global_process_mesh
    world_size = paddle.distributed.get_world_size()
    print(f"world_size={world_size}")
    if world_size == 1:
        _global_parallel_strategy = "dp"
        _global_process_mesh = auto.ProcessMesh([0])
    elif world_size == 2:
        _global_parallel_strategy = "dp"
        _global_process_mesh = auto.ProcessMesh([0, 1])
    elif world_size == 4:
        _global_parallel_strategy = "dp_mp"
        _global_process_mesh = auto.ProcessMesh([[0, 1], [2, 3]])
    elif world_size == 8:
        _global_parallel_strategy = "dp_mp"
        _global_process_mesh = auto.ProcessMesh([[0, 1, 2, 3], [4, 5, 6, 7]])
    elif world_size == 2 or world_size == 16:
        _global_parallel_strategy = "dp_mp"
        _global_process_mesh = auto.ProcessMesh([[0,1,2,3,4,5,6,7], [8,9,10,11,12,13,14,15]])
    else:
        raise ValueError("We only support dp2mp2 or dp4mp2")

    input = np.random.random(size=(8000, 128)).astype('float32')
    label = np.random.random(size=(8000, 1)).astype('float32')

    dist_main_prog, dist_start_prog, loss = get_distributed_program(args)

    with open(args.output_dir + "/dist_main_program.txt.%d" % (paddle.distributed.get_rank()), 'w') as f:
        f.write(str(dist_main_prog))
    with open(args.output_dir + "/dist_startup_program.txt.%d" % (paddle.distributed.get_rank()), 'w') as f:
        f.write(str(dist_start_prog))

    place = paddle.set_device("gpu")
    exe = paddle.static.Executor(place)
    exe.run(dist_start_prog)
    print("========================end of start up prog========================")

    eval_step = 0
    """
    ckpt_dir = os.path.join(args.checkpoint_path, "ckpt_dir")
    if args.elastic_train:
        if os.path.exists(ckpt_dir):
            eval_step = load_checkpoint(ckpt_dir, dist_main_prog)
            print("========================end of load parameter========================")
        else:
            print(f"=> skip load_checkpoint, ckpt_dir:{ckpt_dir} not exists")
    """
    scope = fluid.global_scope()
    avg_loss = 0
    for step in range(eval_step, args.max_steps):
        """
        if step != eval_step and (step % args.save_steps == 0 or step >= args.max_steps):
            output_dir = os.path.join(ckpt_dir, "step_%d" % step)
            os.makedirs(output_dir, exist_ok=True)
            save_distributed_checkpoint(
                dist_main_prog, output_dir, dist_attr_path=output_dir)
            
            time.sleep(20)
        """
        res = exe.run(dist_main_prog,
                        feed={
                            "input": input[step * args.global_batch_size:(step + 1) * args.global_batch_size, :],
                            "label": label[step * args.global_batch_size:(step + 1) * args.global_batch_size, :]
                        },
                        fetch_list=[loss, 
                            dist_main_prog.global_block().var("linear_5.w_0")])
        if step % 4 == 3:
            avg_loss += res[0]
            print(f"avgloss -------->{avg_loss/4}")
            avg_loss = 0
        else:
            avg_loss += res[0]

        print(step, " ", res[0])
        print("------------>" + str(res[1]))
        #print("step=%d, mean_0.tmp_0@GRAD@GradientMerge: %s" % (step, scope.var("mean_0.tmp_0@GRAD@GradientMerge").get_tensor().__array__()))
        #print("step=%d, mean_0.tmp_0: %s" % (step, scope.var("mean_0.tmp_0").get_tensor().__array__()))


if __name__ == "__main__":
    config = parse_args()
    main(config)
