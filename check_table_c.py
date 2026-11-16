import json
import statistics
from pathlib import Path

EXPECTED = {
    'C1': (90.079, 89.233, 90.921),
    'C2': (88.008, 86.319, 89.548),
    'C3': (91.336, 90.640, 92.081),
    'C4': (90.280, 88.768, 91.626),
    'C5': (90.646, 90.302, 91.069),
    'C6': (93.181, 92.703, 93.674),
    'C7': (88.155, 87.280, 89.249),
}


def main():
    failures = 0
    for name, expected in EXPECTED.items():
        record = json.loads((Path(__file__).parent / 'reference_results' / (name + '.json')).read_text())
        accuracy = record['accuracy']
        values = (statistics.mean(accuracy['per_seed']) * 100,
                  accuracy['ci95'][0] * 100, accuracy['ci95'][1] * 100)
        passed = all(f'{actual:.3f}' == f'{wanted:.3f}' for actual, wanted in zip(values, expected))
        failures += not passed
        print(f"{name}: {values[0]:.3f} [{values[1]:.3f}, {values[2]:.3f}] {'PASS' if passed else 'FAIL'}")
    return int(bool(failures))


if __name__ == '__main__':
    raise SystemExit(main())
