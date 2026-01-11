# pyright: reportMissingImports=false
import argparse
import contextlib
import gc
import importlib.util
import io
import os
import re
import sys
import time
import torch  # type: ignore
from itertools import product
from pathlib import Path


ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT / "run_logs"


class Tee:
    """stdout/stderr를 파일과 콘솔 양쪽에 기록하기 위한 간단한 Tee 클래스."""

    def __init__(self):
        self.buffer = io.StringIO()

    def write(self, data):
        sys.__stdout__.write(data)
        sys.__stdout__.flush()
        self.buffer.write(data)

    def flush(self):
        sys.__stdout__.flush()
        self.buffer.flush()


def load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def extract_last_avg_loss(log_text: str):
    matches = re.findall(r"Avg Loss:\s*([0-9.]+)", log_text)
    if not matches:
        return None
    try:
        return float(matches[-1])
    except ValueError:
        return None


def extract_last_hybrid_metrics(log_text: str):
    matches = re.findall(
        r"Box Diff:\s*([0-9.]+)\s*px\s*\|\s*Class Diff:\s*([0-9.]+)", log_text
    )
    if not matches:
        return None, None
    try:
        box, cls = matches[-1]
        return float(box), float(cls)
    except ValueError:
        return None, None


def run_single(script_path: Path, overrides: dict, epochs_override: int | None, tag: str):
    tee = Tee()
    module_name = f"{script_path.stem}_{int(time.time() * 1000)}"
    cwd_backup = Path.cwd()
    os.makedirs(LOG_DIR, exist_ok=True)

    try:
        os.chdir(script_path.parent)
        module = load_module(script_path, module_name)
        module.CONFIG.update(overrides)
        if epochs_override is not None:
            module.CONFIG["epochs"] = epochs_override

        # 모델 저장 파일명을 실험 tag에 따라 고유하게 변경
        if hasattr(module, "torch"):
            original_save = module.torch.save

            def patched_save(obj, path, *args, **kwargs):
                path = Path(path)
                if path.name.startswith("dist_temp"):
                    path = path.with_name(f"{path.stem}_{tag}{path.suffix}")
                elif path.name.startswith("sem_temp"):
                    path = path.with_name(f"{path.stem}_{tag}{path.suffix}")
                else:
                    path = path.with_name(f"{path.stem}_{tag}{path.suffix}")
                return original_save(obj, path, *args, **kwargs)

            module.torch.save = patched_save

        with contextlib.redirect_stdout(tee), contextlib.redirect_stderr(tee):
            module.main()

    except Exception as e:
        tee.write(f"\n[runner] ERROR in {tag}: {e}\n")
    finally:
        os.chdir(cwd_backup)
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass

    log_text = tee.buffer.getvalue()
    log_path = LOG_DIR / f"{tag}.log"
    log_path.write_text(log_text)
    avg_loss = extract_last_avg_loss(log_text)
    box_diff, class_diff = extract_last_hybrid_metrics(log_text)
    return avg_loss, box_diff, class_diff, log_path


def parse_lambda_sets(raw_list: list[str]):
    lambda_sets = []
    for raw in raw_list:
        parts = raw.split(",")
        if len(parts) != 3:
            raise ValueError(f"lambda 설정은 p3,p4,p5 형태여야 합니다: {raw}")
        lp3, lp4, lp5 = (float(x) for x in parts)
        lambda_sets.append({"lambda_p3": lp3, "lambda_p4": lp4, "lambda_p5": lp5})
    return lambda_sets


def append_live_result(results_path: Path, record: dict):
    lines = []
    if not results_path.exists():
        lines.append(f"# Live results (started {time.strftime('%Y-%m-%d %H:%M:%S')})\n")
        lines.append("task,tag,comp,layer,lambda_p3,lambda_p4,lambda_p5,avg_loss,box_diff,class_diff,log\n")
    lp3 = record.get("lambdas", {}).get("lambda_p3") if record.get("lambdas") else ""
    lp4 = record.get("lambdas", {}).get("lambda_p4") if record.get("lambdas") else ""
    lp5 = record.get("lambdas", {}).get("lambda_p5") if record.get("lambdas") else ""
    lines.append(
        f"{record.get('task','')},"
        f"{record.get('tag','')},"
        f"{record.get('compression_ratio','')},"
        f"{record.get('layer','')},"
        f"{lp3},"
        f"{lp4},"
        f"{lp5},"
        f"{record.get('avg_loss','')},"
        f"{record.get('box_diff','')},"
        f"{record.get('class_diff','')},"
        f"{record.get('log','')}\n"
    )
    with results_path.open("a") as f:
        f.writelines(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Train_Dist / Train_Sem 실험 자동화 러너 (avg_loss 최소값 탐색)"
    )
    parser.add_argument("--task", choices=["dist", "sem", "all"], default="all")
    parser.add_argument(
        "--epochs", type=int, default=100, help="CONFIG epochs 강제 오버라이드 (기본 100)"
    )
    parser.add_argument(
        "--dist-layers",
        nargs="+",
        type=int,
        default=[3, 4, 5],
        help="Train_Dist: student_layer_num 후보 리스트",
    )
    parser.add_argument(
        "--dist-compressions",
        nargs="+",
        type=int,
        default=[2, 4, 8],
        help="Train_Dist: compression_ratio 후보 리스트",
    )
    parser.add_argument(
        "--sem-compressions",
        nargs="+",
        type=int,
        default=[2, 4, 8],
        help="Train_Sem: compression_ratio 후보 리스트",
    )
    parser.add_argument(
        "--sem-lambdas",
        nargs="+",
        default=[
            "1,1,1",
            "0.5,1,1",
            "1,0.5,1",
            "1,1,0.5",
            "1,0.5,0.5",
            "0.5,1,0.5",
            "0.5,0.5,1",
        ],
        help="Train_Sem: lambda_p3,lambda_p4,lambda_p5 세트 (쉼표로 구분)",
    )
    args = parser.parse_args()

    results = []
    results_path = ROOT / "results.txt"

    if args.task in ("dist", "all"):
        for layer, comp in product(args.dist_layers, args.dist_compressions):
            tag = f"dist_l{layer}_cr{comp}"
            overrides = {
                "student_layer_num": layer,
                "compression_ratio": comp,
            }
            avg_loss, box_diff, class_diff, log_path = run_single(
                script_path=ROOT / "Train_Dist.py",
                overrides=overrides,
                epochs_override=args.epochs,
                tag=tag,
            )
            results.append(
                {
                    "task": "dist",
                    "tag": tag,
                    "layer": layer,
                    "compression_ratio": comp,
                    "avg_loss": avg_loss,
                    "box_diff": box_diff,
                    "class_diff": class_diff,
                    "log": str(log_path),
                }
            )
            append_live_result(
                results_path,
                {
                    "task": "dist",
                    "tag": tag,
                    "layer": layer,
                    "compression_ratio": comp,
                    "avg_loss": avg_loss,
                    "box_diff": box_diff,
                    "class_diff": class_diff,
                    "log": str(log_path),
                },
            )
            print(
                f"[dist] {tag} -> avg_loss={avg_loss} box={box_diff} cls={class_diff} log={log_path}"
            )

    if args.task in ("sem", "all"):
        lambda_sets = parse_lambda_sets(args.sem_lambdas)
        for comp, lambdas in product(args.sem_compressions, lambda_sets):
            tag = f"sem_cr{comp}_lp3{lambdas['lambda_p3']}_lp4{lambdas['lambda_p4']}_lp5{lambdas['lambda_p5']}"
            overrides = {"compression_ratio": comp, **lambdas}
            avg_loss, box_diff, class_diff, log_path = run_single(
                script_path=ROOT / "Train_Sem.py",
                overrides=overrides,
                epochs_override=args.epochs,
                tag=tag,
            )
            results.append(
                {
                    "task": "sem",
                    "tag": tag,
                    "compression_ratio": comp,
                    "lambdas": lambdas,
                    "avg_loss": avg_loss,
                    "box_diff": box_diff,
                    "class_diff": class_diff,
                    "log": str(log_path),
                }
            )
            append_live_result(
                results_path,
                {
                    "task": "sem",
                    "tag": tag,
                    "compression_ratio": comp,
                    "lambdas": lambdas,
                    "avg_loss": avg_loss,
                    "box_diff": box_diff,
                    "class_diff": class_diff,
                    "log": str(log_path),
                },
            )
            print(
                f"[sem] {tag} -> avg_loss={avg_loss} box={box_diff} cls={class_diff} log={log_path}"
            )

    if results:
        # 전체 정렬 출력
        print("\n====== 정렬된 결과 (avg_loss 오름차순) ======")
        sorted_results = sorted(
            results, key=lambda x: (x["avg_loss"] if x["avg_loss"] is not None else float("inf"))
        )
        for r in sorted_results:
            print(r)

        # compression_ratio 기준 베스트 선택
        best_by_task_comp = {}
        for r in results:
            key = (r["task"], r["compression_ratio"])
            if key not in best_by_task_comp:
                best_by_task_comp[key] = r
            else:
                prev = best_by_task_comp[key]
                prev_loss = prev["avg_loss"] if prev["avg_loss"] is not None else float("inf")
                curr_loss = r["avg_loss"] if r["avg_loss"] is not None else float("inf")
                if curr_loss < prev_loss:
                    best_by_task_comp[key] = r

        best_by_comp_overall = {}
        for r in results:
            key = r["compression_ratio"]
            if key not in best_by_comp_overall:
                best_by_comp_overall[key] = r
            else:
                prev = best_by_comp_overall[key]
                prev_loss = prev["avg_loss"] if prev["avg_loss"] is not None else float("inf")
                curr_loss = r["avg_loss"] if r["avg_loss"] is not None else float("inf")
                if curr_loss < prev_loss:
                    best_by_comp_overall[key] = r

        lines = []
        lines.append(f"# Results generated at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        lines.append("## 상세 결과 (정렬됨)\n")
        for r in sorted_results:
            lines.append(
                f"[{r['task']}] tag={r['tag']} comp={r['compression_ratio']} "
                f"avg_loss={r['avg_loss']} box_diff={r['box_diff']} class_diff={r['class_diff']} log={r['log']}"
            )

        lines.append("\n## compression_ratio별 최고 (task 별)\n")
        for key, r in sorted(best_by_task_comp.items(), key=lambda x: (x[0][0], x[0][1])):
            task, comp = key
            lines.append(
                f"[{task}] comp={comp} best_tag={r['tag']} avg_loss={r['avg_loss']} "
                f"box_diff={r['box_diff']} class_diff={r['class_diff']} log={r['log']}"
            )

        lines.append("\n## compression_ratio별 최고 (task 구분 없이)\n")
        for comp, r in sorted(best_by_comp_overall.items(), key=lambda x: x[0]):
            lines.append(
                f"[all] comp={comp} best_tag={r['tag']} avg_loss={r['avg_loss']} "
                f"box_diff={r['box_diff']} class_diff={r['class_diff']} log={r['log']}"
            )

        results_path.write_text("\n".join(lines))
        print(f"\nresults.txt 작성 완료: {results_path}")
    else:
        print("실행된 실험이 없습니다. 인자를 확인하세요.")


if __name__ == "__main__":
    main()

