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

# pylint: skip-file
# pyformat: disable

import os
import scipy.stats

import numpy as np
from sklearn.metrics import auc, roc_curve
import functools

from absl import app, flags
FLAGS = flags.FLAGS


def sweep(score, x):
    """
    Compute a ROC curve and then return the FPR, TPR, AUC, and ACC.
    """
    fpr, tpr, _ = roc_curve(x, -score)
    acc = np.max(1-(fpr+(1-tpr))/2)
    return fpr, tpr, auc(fpr, tpr), acc

def load_data(p):
    """
    Load our saved keep and scores and put them into two global matrices.
    """
    global scores, keep
    scores = []
    keep = []

    for root,ds,_ in os.walk(p):
        for f in ds:
            if not f.startswith("experiment"): continue
            if not os.path.exists(os.path.join(root,f,"scores")): continue
            last_epoch = sorted(os.listdir(os.path.join(root,f,"scores")))
            if len(last_epoch) == 0: continue
            scores.append(np.load(os.path.join(root,f,"scores",last_epoch[-1])))
            keep.append(np.load(os.path.join(root,f,"keep.npy")))

    scores = np.array(scores)
    keep = np.array(keep)[:,:scores.shape[1]]

    return scores, keep


def load_data_non_global(p):
    """
    Load our saved keep and scores, and put them into two local matrices.
    """
    scores = []
    keep = []

    for root,ds,_ in os.walk(p):
        for f in ds:
            if not f.startswith("experiment"): continue
            if not os.path.exists(os.path.join(root,f,"scores")): continue
            last_epoch = sorted(os.listdir(os.path.join(root,f,"scores")))
            if len(last_epoch) == 0: continue
            scores.append(np.load(os.path.join(root,f,"scores",last_epoch[-1])))
            keep.append(np.load(os.path.join(root,f,"keep.npy")))

    scores = np.array(scores)
    keep = np.array(keep)[:,:scores.shape[1]]

    return scores, keep


def generate_online(keep, scores, check_keep, check_scores, target_record, in_size=100000, out_size=100000,
                  fix_variance=False):
    """
    Fit a two predictive models using keep and scores in order to predict
    if the examples in check_scores were training data or not, using the
    ground truth answer from check_keep.
    """
    dat_in = []
    dat_out = []

    dat_in.append(scores[keep[:,target_record],target_record,:])
    dat_out.append(scores[~keep[:,target_record],target_record,:])

    dat_in = np.array([x[:7] for x in dat_in])
    dat_out = np.array([x[:7] for x in dat_out])

    mean_in = np.median(dat_in, 1)
    mean_out = np.median(dat_out, 1)

    if fix_variance:
        std_in = np.std(dat_in)
        std_out = np.std(dat_in)
    else:
        std_in = np.std(dat_in, 1)
        std_out = np.std(dat_out, 1)

    pr_in = -scipy.stats.norm.logpdf(check_scores[0][target_record], mean_in, std_in+1e-30)
    pr_out = -scipy.stats.norm.logpdf(check_scores[0][target_record], mean_out, std_out+1e-30)
    score = pr_in-pr_out

    prediction = score.mean(1)
    answer = check_keep[0][target_record]

    return prediction, answer


def calculate_results(fn, keep, scores, sweep_fn=sweep):
    """
    For a set of standard attacks, print two lists: the accuracy results and the 
    AUC results of the target records
    """
    acc_list, auc_list = [], []
    for record in FLAGS.target_records:
        output = []
        target_predictions = []
        target_answers = []
        for i in range(16):
            try:
                prediction, answer = fn(np.array(list(keep)[0:i] + list(keep)[i+1:]),
                                        np.array(list(scores)[0:i] + list(scores)[i+1:]),
                                        keep[i:i+1],
                                        scores[i:i+1],
                                        record)

                target_predictions.append(prediction)
                target_answers.append(answer)
            except:
                pass


        _, _, auc, acc = sweep_fn(np.array(target_predictions), np.array(target_answers, dtype=bool))

        acc_list.append(acc)
        auc_list.append(auc)

    print(acc_list)
    print(auc_list)
    return output

def calculate_results_paired(fn, keep, scores, sweep_fn=sweep):
    """
    For a set of paired attacks, print two lists: the accuracy results and the 
    AUC results of the target records
    """
    acc_list, auc_list = [], []
    for record in FLAGS.target_records:
        # For paired attacks, data has to be reloaded after every calculation
        scores, keep = load_data_non_global(f'{FLAGS.exp_dir}/record_{record}')
        output = []
        target_predictions = []
        target_answers = []
        for i in range(16):
            try:
                prediction, answer = fn(np.array(list(keep)[0:i] + list(keep)[i+1:]),
                                        np.array(list(scores)[0:i] + list(scores)[i+1:]),
                                        keep[i:i+1],
                                        scores[i:i+1],
                                        record)

                target_predictions.append(prediction)
                target_answers.append(answer)
            except:
                pass


        _, _, auc, acc = sweep_fn(np.array(target_predictions), np.array(target_answers, dtype=bool))

        acc_list.append(acc)
        auc_list.append(auc)

    print(acc_list)
    print(auc_list)
    return output

#def fig_fpr_tpr():

#    calculate_results(generate_online,
#            keep, scores,
#    )

#    calculate_results(functools.partial(generate_online, fix_variance=True),
#            keep, scores,
#    )


#def fig_fpr_tpr_paired():

#    calculate_results_paired(generate_online,
#            keep, scores,
#    )

#    calculate_results_paired(functools.partial(generate_online, fix_variance=True),
#            keep, scores,
#    )


def main(argv):
    del argv
    try:
        FLAGS.target_records = [int(record) for record in FLAGS.target_records]
    except:
        print('--target_records error: The list provided must contain comma-separated integer values')
        return

    if FLAGS.paired_sampling == False:
        load_data(FLAGS.exp_dir)
        calculate_results(generate_online, keep, scores)
    else:
        load_data(FLAGS.exp_dir)
        calculate_results_paired(generate_online, keep, scores)


if __name__ == '__main__':
    flags.DEFINE_list('target_records', None, 'Target records used in the experiments.')
    flags.DEFINE_string('exp_dir', 'exp/cifar10/standard', 'Directory where experiments were saved.')
    flags.DEFINE_bool('paired_sampling', False, 'Paired sampling attack?')
    app.run(main)