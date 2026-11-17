"""Run prepared train-block OOF models in separate sequential processes."""
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import time

from scripts.prepare_train_blocks import digest, verify_block
from src.common.feature_cache import cache_lock
from src.common.preparation import write_json


def run_blocks(plan, *, only=None, iterations=92, depth=6, learning_rate=.1, seed=42, threads=2, memory_mb=512):
    if iterations < 1 or depth < 1 or threads < 1 or memory_mb < 128 or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError('invalid OOF model or resource parameters')
    plan = Path(plan).resolve()
    prepared_report = json.loads(plan.read_text(encoding='utf-8'))
    if not prepared_report.get('complete'):
        raise ValueError('complete audited block preparation required')
    identity = prepared_report['identity']
    blocks = list(prepared_report['blocks']) if only is None else list(only)
    if not blocks or len(set(blocks)) != len(blocks) or not set(blocks) <= set(prepared_report['blocks']):
        raise ValueError('unknown or duplicate train block')
    prepared = Path(identity['prepared'])
    source_path = prepared / 'manifest.json'
    if digest(source_path) != identity['prepared_manifest_sha256']:
        raise ValueError('prepared manifest changed')
    source = json.loads(source_path.read_text(encoding='utf-8'))
    output = Path(identity['oof_out'])
    run_identity = {'plan_sha256': digest(plan), 'implementation_sha256': digest(__file__),
                    'iterations': iterations, 'depth': depth, 'learning_rate': learning_rate,
                    'seed': seed, 'threads': threads, 'memory_mb': memory_mb}
    with cache_lock(output):
        report_path = output / 'blocks_run.json'
        report = {'identity': run_identity, 'blocks': {}}
        if report_path.exists():
            report = json.loads(report_path.read_text(encoding='utf-8'))
            if report['identity'] != run_identity:
                raise ValueError('block OOF parameters changed; use a new prepared plan and model directory')
        report.update(complete=False, status='running', requested_blocks=blocks)
        write_json(report_path, report)
        try:
            for tag in blocks:
                block = prepared_report['blocks'][tag]
                cohort = block['cohort']
                selected = verify_block(prepared, source, block, identity['shared'])
                directory = output / tag
                command = [sys.executable, '-m', 'scripts.run_joint_oof', '--prepared', str(prepared),
                           '--raw', identity['raw'], '--train-parts', str(len(selected)), '--train-start', str(cohort['start']),
                           '--folds', '4', '--iterations', str(iterations), '--depth', str(depth), '--learning-rate', str(learning_rate),
                           '--seed', str(seed), '--threads', str(threads), '--memory-mb', str(memory_mb),
                           '--out', str(directory), '--resume']
                for root in block['feature_roots']:
                    command.extend(('--feature-root', root))
                report.update(current_block=tag, stage='oof', command=command)
                write_json(report_path, report)
                started = time.monotonic()
                subprocess.run(command, check=True)
                result = json.loads((directory / 'run.json').read_text(encoding='utf-8'))
                if not result.get('complete') or result['submission_validator'] != 'passed':
                    raise ValueError('block OOF did not complete validation')
                report['blocks'][tag] = {'run': str(directory / 'run.json'), 'validation': result['validation'],
                                        'prediction_sha256': result['prediction_sha256'],
                                        'last_call_seconds': round(time.monotonic() - started, 2)}
                write_json(report_path, report)
            if digest(plan) != run_identity['plan_sha256']:
                raise ValueError('prepared block plan changed during OOF')
            report.pop('error', None)
            report.update(status='complete', complete=True, stage='complete')
            write_json(report_path, report)
            return report
        except BaseException as error:
            report.update(status='failed', complete=False, error=repr(error))
            write_json(report_path, report)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--only', nargs='+', help='optional prepared block names')
    parser.add_argument('--iterations', type=int, default=92)
    parser.add_argument('--depth', type=int, default=6)
    parser.add_argument('--learning-rate', type=float, default=.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--memory-mb', type=int, default=512)
    args = parser.parse_args()
    run_blocks(**vars(args))


if __name__ == '__main__':
    main()
