"""Rebuild kuairand_1k.inter with a k-core item+user floor (H2/D6 1K fix).

The original 1K .inter had NO item floor -> 65% of items appear once and 33.5% of
held-out test targets are never in training -> unrankable -> degenerate ~0 HR.
This applies an iterative k-core (item >= MIN_ITEM, user >= MIN_USER) until stable
and writes kuairand_1k_kcore.inter, then reports the viability diagnostic (what
fraction of leave-one-out test targets are now trainable).
"""
import sys
import pandas as pd

SRC = r"C:/Users/msc/Desktop/G00403092/CAFREC/data/recbole/kuairand_1k/kuairand_1k.inter"
OUT = r"C:/Users/msc/Desktop/G00403092/CAFREC/data/recbole/kuairand_1k_kcore/kuairand_1k_kcore.inter"
MIN_ITEM = 10
MIN_USER = 5


def kcore(df):
    while True:
        n0 = len(df)
        ic = df["item_id"].value_counts()
        df = df[df["item_id"].isin(ic[ic >= MIN_ITEM].index)]
        uc = df["user_id"].value_counts()
        df = df[df["user_id"].isin(uc[uc >= MIN_USER].index)]
        if len(df) == n0:
            return df


def viability(df):
    """Leave-one-out: last interaction per user is the test target. Fraction of
    test targets that also appear in the remaining (train) rows -> rankable."""
    df = df.sort_values("timestamp")
    last_idx = df.groupby("user_id").tail(1).index
    test = df.loc[last_idx]
    train_item_counts = df.drop(last_idx)["item_id"].value_counts()
    trainable = test["item_id"].map(train_item_counts).fillna(0) > 0
    return trainable.mean()


def main():
    df = pd.read_csv(SRC, sep="\t")
    df.columns = ["user_id", "item_id", "timestamp"]
    print(f"BEFORE: rows={len(df):,} users={df.user_id.nunique():,} "
          f"items={df.item_id.nunique():,}")
    ic = df["item_id"].value_counts()
    print(f"  items appearing once: {(ic == 1).mean():.1%}; "
          f"mean item freq {len(df)/df.item_id.nunique():.2f}")
    print(f"  test-target trainable: {viability(df):.1%}")

    df = kcore(df)
    print(f"\nAFTER k-core (item>={MIN_ITEM}, user>={MIN_USER}): rows={len(df):,} "
          f"users={df.user_id.nunique():,} items={df.item_id.nunique():,}")
    ic = df["item_id"].value_counts()
    print(f"  items appearing once: {(ic == 1).mean():.1%}; "
          f"mean item freq {len(df)/df.item_id.nunique():.2f}")
    print(f"  test-target trainable: {viability(df):.1%}")

    if "--write" in sys.argv:
        import os
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        out = df.rename(columns={"user_id": "user_id:token",
                                 "item_id": "item_id:token",
                                 "timestamp": "timestamp:float"})
        out.to_csv(OUT, sep="\t", index=False)
        print(f"\nwrote {OUT} ({len(out):,} rows)")


if __name__ == "__main__":
    main()
