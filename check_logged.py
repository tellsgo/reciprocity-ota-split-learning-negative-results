import csv
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent / 'reference_results'
FAILURES = 0


def check(label, values, expected, decimals):
    global FAILURES
    passed = all(f'{value:.{decimals}f}' == f'{target:.{decimals}f}'
                 for value, target in zip(values, expected))
    FAILURES += not passed
    print(label, values, 'PASS' if passed else 'FAIL')


analysis = json.loads((ROOT / 'A.json').read_text())['cases']
singular_values = json.loads((ROOT / 'A.json').read_text())['A_proxy_singular_values']
check('A weakest singular value', (min(singular_values),), (0.00915798,), 8)
phase = json.loads((ROOT / 'p2_phase_gradient.json').read_text())
check('B local bright/dark ratio and interval',
      (phase['bright_dark_ratio'], *phase['ratio_ci95']), (0.08694, 0.08320, 0.09042), 5)
for name, expected in [('2_polar', (0.506785, 3.262474)),
                       ('2_TSVD', (0.502829, 3.258438))]:
    case = analysis[name]
    check('A ' + name, (case['second_moment_radius'], case['stationary_energy'] * 1e4), expected, 6)
for name, expected in [('3_polar', 1.025053), ('3_TSVD', 1.000000)]:
    check('A ' + name, (analysis[name]['second_moment_radius'],), (expected,), 6)
with (ROOT / 'A_sgd.csv').open() as stream:
    sgd_rows = list(csv.DictReader(stream))
sgd = next(row for row in sgd_rows if row['optimizer'] == 'SGD'
           and row['case'] == 'isotropic/0.38' and row['contrast'] == 'pinv_64 - polar_64')
check('A SGD paired difference and logged interval',
      (statistics.mean(float(sgd[f'seed_{index}']) for index in range(8)),
       float(sgd['lower']), float(sgd['upper'])), (-0.000813, -0.001853, 0.000060), 6)

with (ROOT / 'B.csv').open() as stream:
    channel_rows = list(csv.DictReader(stream))
for spacing, expected in [('0.2', (-0.647, 100.000)), ('0.5', (0.970, 99.899))]:
    rows = [row for row in channel_rows if row['family'] == 'isotropic'
            and row['version'] == 'vector4' and row['spacing'] == spacing]
    means = {arm: statistics.mean(float(row['p_positive']) for row in rows if row['arm'] == arm) * 100
             for arm in ('max_snr', 'random', 'slide')}
    check('B spacing ' + spacing, (means['max_snr'] - means['random'], means['slide']), expected, 3)
replay = json.loads((ROOT / 'B_replay.json').read_text())
check('B replay accuracy and alignment',
      (statistics.mean(replay['accuracy']['per_seed']) * 100,
       statistics.mean(replay['p_positive']['per_seed']) * 100), (89.169, 63.641), 3)
check('B replay cosine', (statistics.mean(replay['mean_cosine']['per_seed']),), (0.29121,), 5)

for name, methods, expected in [
    ('E1', ('farfield_polar', 'nearfield_polar'), (56.5598, 56.3568, -0.2029)),
    ('E2', ('farfield_steering', 'nearfield_focus_polar'), (99.7955, 99.7864, -0.0092)),
]:
    with (ROOT / (name + '.csv')).open() as stream:
        rows = list(csv.DictReader(stream))
    means = [statistics.mean(float(row['accuracy']) for row in rows if row['method'] == method) * 100
             for method in methods]
    check(name, (*means, means[1] - means[0]), expected, 4)

for filename, key, expected in [
    ('G1', 'MA_bias_0.383', (50.3235, 47.1573, 53.4624)),
    ('G1', 'MA_jitter_0.383', (98.4222, 98.2681, 98.5779)),
    ('G2', 'MA_bias_crit', (53.2944, 50.2869, 57.9941)),
    ('G2', 'MA_rigid_jitter_0.383', (98.4909, 98.3658, 98.6221)),
    ('G2', 'MA_indep_jitter_0.383', (98.4711, 98.3688, 98.5886)),
]:
    accuracy = json.loads((ROOT / (filename + '.json')).read_text())[key]['accuracy']
    check(filename + ' ' + key,
          (statistics.mean(accuracy['per_seed']) * 100, *(x * 100 for x in accuracy['ci95'])),
          expected, 4)

f1 = next(row for row in json.loads((ROOT / 'F1.json').read_text())
          if row['method'] == 'S_perm' and row['angle'] == 30)
accuracy = f1['accuracy']
check('F1 permutation accuracy and logged interval',
      (statistics.mean(accuracy['per_seed']) * 100, *(x * 100 for x in accuracy['ci95'])),
      (99.5392, 99.4781, 99.6155), 4)
check('F1 negative updates', (statistics.mean(f1['negative_fraction']['per_seed']) * 100,), (2.1875,), 4)
with (ROOT / 'F2.csv').open() as stream:
    f2 = [row for row in csv.DictReader(stream) if row['method'] == 'S_LAP']
groups = {(row['n'], row['angle']) for row in f2}
means = [statistics.mean(float(row['accuracy']) for row in f2
                         if (row['n'], row['angle']) == group) * 100 for group in groups]
check('F2 accuracy range', (min(means), max(means)), (99.5422, 99.6841), 4)

raise SystemExit(bool(FAILURES))
