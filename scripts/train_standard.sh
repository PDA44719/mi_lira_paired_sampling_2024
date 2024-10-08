# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/bin/bash
CUDA_VISIBLE_DEVICES='0' python3 -u train.py --dataset=cifar10 --epochs=35 --save_steps=35 --arch wrn28-2 --num_experiments 16 --seed=$1 --expid 0 --logdir exp/cifar10/standard &> logs/standard/log_0
CUDA_VISIBLE_DEVICES='0' python3 -u train.py --dataset=cifar10 --epochs=35 --save_steps=35 --arch wrn28-2 --num_experiments 16 --seed=$1 --expid 1 --logdir exp/cifar10/standard &> logs/standard/log_1
CUDA_VISIBLE_DEVICES='0' python3 -u train.py --dataset=cifar10 --epochs=35 --save_steps=35 --arch wrn28-2 --num_experiments 16 --seed=$1 --expid 2 --logdir exp/cifar10/standard &> logs/standard/log_2
CUDA_VISIBLE_DEVICES='0' python3 -u train.py --dataset=cifar10 --epochs=35 --save_steps=35 --arch wrn28-2 --num_experiments 16 --seed=$1 --expid 3 --logdir exp/cifar10/standard &> logs/standard/log_3
CUDA_VISIBLE_DEVICES='0' python3 -u train.py --dataset=cifar10 --epochs=35 --save_steps=35 --arch wrn28-2 --num_experiments 16 --seed=$1 --expid 4 --logdir exp/cifar10/standard &> logs/standard/log_4
CUDA_VISIBLE_DEVICES='0' python3 -u train.py --dataset=cifar10 --epochs=35 --save_steps=35 --arch wrn28-2 --num_experiments 16 --seed=$1 --expid 5 --logdir exp/cifar10/standard &> logs/standard/log_5
CUDA_VISIBLE_DEVICES='0' python3 -u train.py --dataset=cifar10 --epochs=35 --save_steps=35 --arch wrn28-2 --num_experiments 16 --seed=$1 --expid 6 --logdir exp/cifar10/standard &> logs/standard/log_6
CUDA_VISIBLE_DEVICES='0' python3 -u train.py --dataset=cifar10 --epochs=35 --save_steps=35 --arch wrn28-2 --num_experiments 16 --seed=$1 --expid 7 --logdir exp/cifar10/standard &> logs/standard/log_7
CUDA_VISIBLE_DEVICES='0' python3 -u train.py --dataset=cifar10 --epochs=35 --save_steps=35 --arch wrn28-2 --num_experiments 16 --seed=$1 --expid 8 --logdir exp/cifar10/standard &> logs/standard/log_8
CUDA_VISIBLE_DEVICES='0' python3 -u train.py --dataset=cifar10 --epochs=35 --save_steps=35 --arch wrn28-2 --num_experiments 16 --seed=$1 --expid 9 --logdir exp/cifar10/standard &> logs/standard/log_9
CUDA_VISIBLE_DEVICES='0' python3 -u train.py --dataset=cifar10 --epochs=35 --save_steps=35 --arch wrn28-2 --num_experiments 16 --seed=$1 --expid 10 --logdir exp/cifar10/standard &> logs/standard/log_10
CUDA_VISIBLE_DEVICES='0' python3 -u train.py --dataset=cifar10 --epochs=35 --save_steps=35 --arch wrn28-2 --num_experiments 16 --seed=$1 --expid 11 --logdir exp/cifar10/standard &> logs/standard/log_11
CUDA_VISIBLE_DEVICES='0' python3 -u train.py --dataset=cifar10 --epochs=35 --save_steps=35 --arch wrn28-2 --num_experiments 16 --seed=$1 --expid 12 --logdir exp/cifar10/standard &> logs/standard/log_12
CUDA_VISIBLE_DEVICES='0' python3 -u train.py --dataset=cifar10 --epochs=35 --save_steps=35 --arch wrn28-2 --num_experiments 16 --seed=$1 --expid 13 --logdir exp/cifar10/standard &> logs/standard/log_13
CUDA_VISIBLE_DEVICES='0' python3 -u train.py --dataset=cifar10 --epochs=35 --save_steps=35 --arch wrn28-2 --num_experiments 16 --seed=$1 --expid 14 --logdir exp/cifar10/standard &> logs/standard/log_14
CUDA_VISIBLE_DEVICES='0' python3 -u train.py --dataset=cifar10 --epochs=35 --save_steps=35 --arch wrn28-2 --num_experiments 16 --seed=$1 --expid 15 --logdir exp/cifar10/standard &> logs/standard/log_15
