""" Utils for parsing PBA augmentation schedules"""

import copy
from augmentation_transforms_hp import NUM_HP_TRANSFORM
import json
import os
import numpy as np
import pickle
import random


def parse_log(file, epochs):
    #                     0             1                 2                            3          4          5
    # input contains [trial_name, trial_to_clone_name, trial_epochs, trial_to_clone_epochs, old_config, new_config]
    # return list containing lines: [start epoch, start_epoch_clone, policy]
    raw_policy = open(file, "rb").readlines()
    raw_policy = [eval(line) for line in raw_policy]
    policy = []
    # sometimes files have extra lines in the beginning
    to_truncate = None
    for i in range(len(raw_policy) - 1):
        if raw_policy[i][0] != raw_policy[i+1][1]:
            to_truncate = i
    if to_truncate is not None:
        raw_policy = raw_policy[to_truncate+1:]

    # initial policy for trial_to_clone_epochs
    policy.append([raw_policy[0][3], raw_policy[0][4]])

    current = raw_policy[0][3]
    for i in range(len(raw_policy)-1):
        # end at next line's trial epoch, start from this clone epoch
        this_iter = raw_policy[i+1][3] - raw_policy[i][3]
        assert this_iter >= 0, (i, raw_policy[i+1][3], raw_policy[i][3])
        assert raw_policy[i][0] == raw_policy[i +
                                              1][1], (i, raw_policy[i][0], raw_policy[i+1][1])
        policy.append([this_iter, raw_policy[i][5]])
        current += this_iter
    # last cloned trial policy is run for (end - clone iter of last logged line)
    policy.append([epochs - raw_policy[-1][3], raw_policy[-1][5]])
    current += epochs - raw_policy[-1][3]
    assert epochs == sum([p[0] for p in policy])
    return policy


def parse_log_schedule(file, epochs, multiplier=1):
    policy = parse_log(file, epochs)
    schedule = []
    for num_iters, pol in policy:
        for _ in range(num_iters * multiplier):
            schedule.append(pol)
    return schedule


def ablation_shuffle(file_path):
    schedule = parse_log(file_path, epochs=200)
    schedules = []
    for i in range(256):
        shuffled_schedule = copy.copy(schedule)
        random.shuffle(shuffled_schedule)
        schedules.append(shuffled_schedule)
    print(len(schedules))
    with open("schedules/ablations/shuffled.p", "wb") as f:
        pickle.dump(schedules, f)


def ablation_random_schedule():
    schedules = []
    for i in range(256):
        total = 0
        one_schedule = []
        while total < 200:
            t = random.randint(1, 40)
            if total + t < 200:
                total += t
            else:
                t = 200 - total
                total = 200
            this_s = []
            for i in range(NUM_HP_TRANSFORM*4):
                if i % 2 == 0:
                    this_s.append(random.randint(0, 10))
                else:
                    this_s.append(random.randint(0, 9))
            s = [t, this_s]
            one_schedule.append(s)
        schedules.append(one_schedule)
    print(len(schedules))
    with open("schedules/ablations/rand.p", "wb") as f:
        pickle.dump(schedules, f)
