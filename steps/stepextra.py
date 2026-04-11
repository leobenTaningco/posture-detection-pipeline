import pandas as pd
#( LABEL_GOOD, LABEL_BAD, RANDOM_SEED)
LABEL_GOOD   = 1
LABEL_BAD    = 0
RANDOM_SEED  = 42

FEATURES_CSV = "./dataset/features.csv"
df = pd.read_csv(FEATURES_CSV)

# Drop any leftover synthetics
df = df[df["filename"] != "synthetic"].copy()

good_df = df[df["label"] == LABEL_GOOD]
bad_df  = df[df["label"] == LABEL_BAD]

print(f"Before: good={len(good_df)}, bad={len(bad_df)}")

# Undersample bad to match good count
bad_df = bad_df.sample(n=len(good_df), random_state=RANDOM_SEED)

balanced = pd.concat([good_df, bad_df]).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

print(f"After:  good={len(balanced[balanced.label==LABEL_GOOD])}, bad={len(balanced[balanced.label==LABEL_BAD])}")

balanced.to_csv(FEATURES_CSV, index=False)
print("Saved.")