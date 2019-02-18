import os
import json
import collections

p = '/data/dho/ray_results_2/svhn_grid_search/svhn_gs/'
runs = os.listdir(p)
vals = collections.defaultdict(list)
for run in runs:

  with open(p + run + '/result.json') as f:
    for line in f:
      data = line
  data = json.loads(data)
  data = data['test_acc']
#  print(run)
  lr_start = run.find('=')
  lr_end = run.find(',', lr_start)
  wd_start = run.find('=', lr_end)
  wd_end = run.find('_', wd_start)
  lr = run[lr_start+1:lr_end]
  wd = run[wd_start+1:wd_end]
#  print(lr, wd)
  vals[(lr,wd)].append(data)

for k, v in vals.iteritems():
  v = [str(one_v) for one_v in v]
  assert(len(v)==3)
  print("{} {} {}".format(k[0], k[1], '+'.join(v)))

