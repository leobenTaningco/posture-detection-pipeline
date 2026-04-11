import subprocess

steps = [
    # "steps/step1_augment.py",
    "steps/step2_extract.py",
    "steps/step3_dedupe_balance.py",
    "steps/stepextra.py",
    "steps/step4_train.py",
    "steps/step5_analytics.py"
]

for step in steps:
    print(f"Running {step}…")
    result = subprocess.run(["python", step])
    if result.returncode != 0:
        print(f"❌ {step} failed. Stopping pipeline.")
        break
else:
    print("✅ All steps completed successfully!")