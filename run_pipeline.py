"""
run_pipeline.py — Master runner
Runs all 5 steps in order. You can also run each step individually.

Usage:
    python run_pipeline.py           # run all steps
    python run_pipeline.py --step 4  # run only step 4
"""

import sys, time, argparse

def run_step(n):
    if n == 1:
        import step1_augment as m
    elif n == 2:
        import step2_extract as m
    elif n == 3:
        import step3_dedupe_balance as m
    elif n == 4:
        import step4_train as m
    elif n == 5:
        import step5_analytics as m
    else:
        print(f"Unknown step {n}")
        return

    t0 = time.time()
    print(f"\n{'='*60}")
    print(f"  STEP {n}")
    print(f"{'='*60}")
    m.run()
    print(f"\n  ⏱  Step {n} finished in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=int, default=0,
                        help="Run a single step (1-5). Omit to run all.")
    args = parser.parse_args()

    if args.step:
        run_step(args.step)
    else:
        for step in range(1, 6):
            run_step(step)

    print("\n✅  Pipeline complete.")
