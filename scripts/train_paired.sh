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
CUDA_VISIBLE_DEVICES='0' python3 -u train.py --dataset=cifar10 --epochs=35 --save_steps=35 --arch wrn28-2 --num_experiments 16 --seed=$1 --paired_sampling=True --target_record=$2 --expid 1 --logdir exp/cifar10/paired/record_$2 &> logs/paired/record_$2/log_1
CUDA_VISIBLE_DEVICES='0' python3 -u train.py --dataset=cifar10 --epochs=35 --save_steps=35 --arch wrn28-2 --num_experiments 16 --seed=$1 --paired_sampling=True --target_record=$2 --expid 3 --logdir exp/cifar10/paired/record_$2 &> logs/paired/record_$2/log_3
CUDA_VISIBLE_DEVICES='0' python3 -u train.py --dataset=cifar10 --epochs=35 --save_steps=35 --arch wrn28-2 --num_experiments 16 --seed=$1 --paired_sampling=True --target_record=$2 --expid 5 --logdir exp/cifar10/paired/record_$2 &> logs/paired/record_$2/log_5
CUDA_VISIBLE_DEVICES='0' python3 -u train.py --dataset=cifar10 --epochs=35 --save_steps=35 --arch wrn28-2 --num_experiments 16 --seed=$1 --paired_sampling=True --target_record=$2 --expid 7 --logdir exp/cifar10/paired/record_$2 &> logs/paired/record_$2/log_7
CUDA_VISIBLE_DEVICES='0' python3 -u train.py --dataset=cifar10 --epochs=35 --save_steps=35 --arch wrn28-2 --num_experiments 16 --seed=$1 --paired_sampling=True --target_record=$2 --expid 9 --logdir exp/cifar10/paired/record_$2 &> logs/paired/record_$2/log_9
CUDA_VISIBLE_DEVICES='0' python3 -u train.py --dataset=cifar10 --epochs=35 --save_steps=35 --arch wrn28-2 --num_experiments 16 --seed=$1 --paired_sampling=True --target_record=$2 --expid 11 --logdir exp/cifar10/paired/record_$2 &> logs/paired/record_$2/log_11
CUDA_VISIBLE_DEVICES='0' python3 -u train.py --dataset=cifar10 --epochs=35 --save_steps=35 --arch wrn28-2 --num_experiments 16 --seed=$1 --paired_sampling=True --target_record=$2 --expid 13 --logdir exp/cifar10/paired/record_$2 &> logs/paired/record_$2/log_13
CUDA_VISIBLE_DEVICES='0' python3 -u train.py --dataset=cifar10 --epochs=35 --save_steps=35 --arch wrn28-2 --num_experiments 16 --seed=$1 --paired_sampling=True --target_record=$2 --expid 15 --logdir exp/cifar10/paired/record_$2 &> logs/paired/record_$2/log_15
