"""Analyze short six-channel captures while preserving counters across invalid samples.

The channel labels and byte widths are inferred, not an official register specification.
"""
import argparse
import json
import math
from pathlib import Path
import re
if __package__:
    from .reproduce_perfstat_table import CHANNELS, HEADER, RECORD, decode_capture
else:
    from reproduce_perfstat_table import CHANNELS, HEADER, RECORD, decode_capture


def analyze(data, clock_hz):
    if not math.isfinite(clock_hz) or clock_hz <= 0:
        raise ValueError('A positive, verified sampling clock is required')
    reference = decode_capture(data)  # Reuse format, sequence, mask and padding checks.
    count = reference['sample_count']
    duration = count * reference['period_ticks'] / clock_hz
    previous = [None] * len(CHANNELS)
    totals = [[0, 0] for _ in CHANNELS]
    recoveries = [0] * len(CHANNELS)
    last_valid = [False] * len(CHANNELS)
    examples = [[] for _ in CHANNELS]
    for i in range(count):
        values = RECORD.unpack_from(data, HEADER.size + i * RECORD.size)
        for c, (_, ri, wi, _, bit) in enumerate(CHANNELS):
            valid = bool(values[52] & (1 << bit))
            if not valid:
                last_valid[c] = False
                continue
            current = [values[ri], values[wi]]
            old = previous[c]
            if old is None:
                previous[c] = current
            else:
                if not last_valid[c]:
                    recoveries[c] += 1
                    if len(examples[c]) < 3:
                        examples[c].append(dict(sample_index=i, previous_valid=old[:], current=current,
                                                increment=[max(a-b, 0) for a,b in zip(current,old)]))
                # shortcut: transient stale snapshots are tolerated; true counter resets/wraps require a specification.
                for j in range(2):
                    totals[c][j] += max(current[j] - old[j], 0)
                    old[j] = max(old[j], current[j])
            last_valid[c] = True
    channels = []
    for c, (name, _, _, width, bit) in enumerate(CHANNELS):
        rd, wr = [n * width for n in totals[c]]
        channels.append(dict(name=name, assumed_port_id=64+bit, inferred_bytes_per_count=width,
                             read_counts=totals[c][0], write_counts=totals[c][1],
                             read_bytes=rd, write_bytes=wr,
                             read_GiBps=rd/duration/2**30, write_GiBps=wr/duration/2**30,
                             invalid_samples=reference['channels'][c]['invalid_samples'],
                             validity_recoveries=recoveries[c], recovery_examples=examples[c],
                             screenshot_rule_read_GiBps=reference['channels'][c]['read_GiBps'],
                             screenshot_rule_write_GiBps=reference['channels'][c]['write_GiBps']))
    conservation = {}
    for direction in ('read', 'write'):
        source = sum(c[direction+'_bytes'] for c in channels[:4])
        combined = sum(c[direction+'_bytes'] for c in channels[4:])
        conservation[direction] = dict(cpu_tcu_aicore_bytes=source, ibqe_pair_bytes=combined,
                                       difference_bytes=combined-source,
                                       relative_difference=abs(combined-source)/max(source,combined,1),
                                       ibqe_pair_GiBps=combined/duration/2**30)
    return dict(sha256=reference['sha256'], sample_count=count, period_ticks=reference['period_ticks'],
                sampling_clock_hz=clock_hz, duration_seconds=duration,
                method='First valid sample initializes each baseline. Invalid samples preserve it. Positive increments update it; transient decreases are ignored.',
                limitations='Inferred six-channel format and byte widths; short unwrapped captures only; true counter resets/wraps and physical DDR attribution are not resolved.',
                channels=channels, conservation=conservation)


def analyze_directory(directory):
    kernel = (directory / 'kernel_perfstat.log').read_text(encoding='utf-8')
    clocks = re.findall(r'\bclk_perf_stat is (\d+)', kernel)
    if not clocks:
        raise ValueError('No sampling clock in kernel log: ' + str(directory))
    result = analyze((directory / 'perf_data_hs.data').read_bytes(), int(clocks[-1]))
    (directory / 'counter_bandwidth.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path, nargs='+')
    args = parser.parse_args()
    for directory in args.directory:
        result = analyze_directory(directory)
        print(directory.name, f"{result['duration_seconds']:.6f} s")
        for channel in result['channels']:
            print('{name:14s} read={read_GiBps:.6f} write={write_GiBps:.6f} GiB/s'.format(**channel))
        print('Conservation:', json.dumps(result['conservation']))
