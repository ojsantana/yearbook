import numpy as np

from utils import get_msg_mgr
from .metric import cuda_dist

def evaluate_Yearbook(data, dataset, metric='euc'):

  runners, sequences, features = data['labels'], data['types'], data['embeddings']

  yearbook = {}
  for runner, sequence, feature in zip(runners, sequences, features): yearbook.setdefault(runner, {})[sequence] = feature

  for probe_year in range(1,5):
    for gallery_year in range(1,5):

      R1 = []
      R5 = []
      mAP = []
      output_header = "    "
      single_year = (probe_year == gallery_year)

      valid_runners = sorted({runner for runner,sequences in yearbook.items()
                             if  sum(sequence.startswith(str(probe_year))   for sequence in sequences) == 3
                             and sum(sequence.startswith(str(gallery_year)) for sequence in sequences) == 3})

      probe_y = np.array(valid_runners)
      gallery_y = probe_y.copy()

      for probe_sequence in range(1, 4):
        for gallery_sequence in range(1, 4):

          if single_year and (probe_sequence == gallery_sequence): continue

          output_header += f"    {probe_sequence}vs{gallery_sequence}"

          probe_x = np.array([yearbook[runner][f"{probe_year}.{probe_sequence}"] for runner in valid_runners])
          gallery_x = np.array([yearbook[runner][f"{gallery_year}.{gallery_sequence}"] for runner in valid_runners])

          ranks = gallery_x.shape[0]

          dist = cuda_dist(probe_x, gallery_x, metric)
          index = dist.argsort(dim=1).cpu().numpy()

          matches = (probe_y[:, None] == gallery_y[index])

          cmc_curve = np.cumsum(matches, axis=1) > 0
          cmc = cmc_curve.mean(axis=0) * 100
          R1.append(cmc[0])
          R5.append(cmc[4])

          precision_at_k = np.cumsum(matches, axis=1) / (np.arange(ranks) + 1)
          ap = np.sum(precision_at_k * matches, axis=1) / np.maximum(np.sum(matches, axis=1), 1)
          mAP.append(ap.mean() * 100)

      msg_mgr = get_msg_mgr()
      metrics = [("R1", R1), ("R5", R5), ("mAP", mAP)]

      if single_year: msg_mgr.log_info(f"================ Yearbook:  year #{probe_year} ================")
      else:           msg_mgr.log_info(f"====================== Yearbook:  year #{probe_year} vs. year #{gallery_year} ======================")
      msg_mgr.log_info(output_header)

      for name, values in metrics: msg_mgr.log_info(f"{name:<3}:" + "".join(f" {v:6.2f}%" for v in values))

      msg_mgr.log_info("=" * 52 if single_year else "=" * 76)

      for name, values in metrics:
        values = np.array(values)
        mean = values.mean()
        std = values.std(ddof=1)
        msg_mgr.log_info(f"{name:<3}: {mean:6.2f}% ± {std:5.2f}%")

      print()