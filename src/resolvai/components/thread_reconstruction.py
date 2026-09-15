import re
import pandas as pd

from resolvai import logger
from resolvai.entity.config_entity import ThreadReconstructionConfig
from resolvai.components.data_ingestion import BOILERPLATE_PATTERN

MENTION_RE = re.compile(r"@\w+")
URL_RE = re.compile(r"https?://\S+")
WHITESPACE_RE = re.compile(r"\s+")


def clean_text(t: str) -> str:
    t = MENTION_RE.sub("", str(t))
    t = URL_RE.sub("", t)
    t = WHITESPACE_RE.sub(" ", t).strip()
    return t


class ThreadReconstruction:
    def __init__(self, config: ThreadReconstructionConfig):
        self.config = config

    def reconstruct_and_clean(self) -> pd.DataFrame:
        subsample = pd.read_csv(self.config.subsample_path, low_memory=False)
        # IDs round-tripped through CSV can silently become floats when a column mixes
        # NaN with numbers (e.g. "1" -> 1.0 -> "1.0"), which breaks the string-equality
        # join below. Force both ID columns through the same numeric->Int64->str path
        # so "1" and "1" actually match instead of "1" vs "1.0".
        subsample["tweet_id"] = (
            pd.to_numeric(subsample["tweet_id"], errors="coerce").astype("Int64").astype(str).replace("<NA>", pd.NA)
        )
        subsample["in_response_to_tweet_id"] = (
            pd.to_numeric(subsample["in_response_to_tweet_id"], errors="coerce")
            .astype("Int64").astype(str).replace("<NA>", pd.NA)
        )
        subsample["inbound"] = subsample["inbound"].astype(bool)

        tweet_lookup = subsample.set_index("tweet_id").to_dict("index")
        brand_rows = subsample[subsample["inbound"] == False]

        pairs = []
        for _, row in brand_rows.iterrows():
            parent_id = row["in_response_to_tweet_id"]
            if parent_id in (None, "nan", "<NA>"):
                continue
            parent = tweet_lookup.get(parent_id)
            if parent is None or parent["inbound"] != True:
                continue
            pairs.append(
                {
                    "customer_tweet_id": parent_id,
                    "customer_text": parent["text"],
                    "brand_tweet_id": row["tweet_id"],
                    "brand_text": row["text"],
                }
            )

        pairs_df = pd.DataFrame(pairs)
        logger.info(f"reconstructed {len(pairs_df)} raw pairs")

        pairs_df["customer_text_clean"] = pairs_df["customer_text"].apply(clean_text)
        pairs_df["brand_text_clean"] = pairs_df["brand_text"].apply(clean_text)
        pairs_df["is_boilerplate"] = pairs_df["brand_text_clean"].str.contains(BOILERPLATE_PATTERN, na=False)

        pairs_df = pairs_df[
            (pairs_df["customer_text_clean"].str.len() > 5) & (pairs_df["brand_text_clean"].str.len() > 5)
        ]
        pairs_df = pairs_df.drop_duplicates(subset=["customer_text_clean", "brand_text_clean"]).reset_index(drop=True)

        pairs_df.to_csv(self.config.cleaned_pairs_path, index=False)
        logger.info(f"cleaned pairs: {len(pairs_df)} -> {self.config.cleaned_pairs_path} "
                    f"(boilerplate ratio: {pairs_df['is_boilerplate'].mean():.2f})")
        return pairs_df
