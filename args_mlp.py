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

import argparse


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the training logs and checkpoints will be written."
    )

    parser.add_argument(
        "--micro_batch_size",
        default=1,
        type=int,
        help="Batch size per device for one step training.", )
    parser.add_argument(
        "--local_batch_size",
        default=None,
        type=int,
        help="Global batch size for all training process. None for not check the size is valid. If we only use data parallelism, it should be device_num * micro_batch_size.", )
    parser.add_argument(
        "--global_batch_size",
        default=None,
        type=int,
        help="Global batch size for all training process. None for not check the size is valid. If we only use data parallelism, it should be device_num * micro_batch_size."
    )

    parser.add_argument(
        "--max_steps",
        default=500000,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs."
    )

    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save checkpoint every X updates steps.")

    parser.add_argument(
        "--device",
        type=str,
        default="gpu",
        choices=["cpu", "gpu", "xpu"],
        help="select cpu, gpu, xpu devices.")

    parser.add_argument(
        "--auto_search",
        type=str2bool,
        nargs='?',
        const=False,
        help="Using the auto search function to find strategy.")

    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Where to load model checkpoint.")

    parser.add_argument(
        "--elastic_train",
        type=str2bool,
        nargs='?',
        const=False,
        help="Where to load model checkpoint.")

    args = parser.parse_args()
    return args