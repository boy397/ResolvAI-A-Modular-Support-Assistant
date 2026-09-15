import re
import pandas as pd

from resolvai import logger
from resolvai.entity.config_entity import DataIngestionConfig

BOILERPLATE_PATTERN = re.compile(
    r"sorry to hear|please dm|send us a (?:dm|message)|we'd like to help|reach out via dm",
    re.IGNORECASE,
)


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def _load(self) -> pd.DataFrame:
        df = pd.read_csv(self.config.raw_data_path, low_memory=False)
        df["inbound"] = df["inbound"].astype(bool)
        df["tweet_id"] = df["tweet_id"].astype("Int64").astype(str).replace("<NA>", pd.NA)
        df["in_response_to_tweet_id"] = (
            df["in_response_to_tweet_id"].astype("Int64").astype(str).replace("<NA>", pd.NA)
        )
        df["response_tweet_id"] = df["response_tweet_id"].astype(str).replace("nan", pd.NA)
        logger.info(f"loaded raw dataset: {len(df):,} rows")
        return df

    def profile_brands(self) -> pd.DataFrame:
        """Volume / boilerplate-ratio / thread-validity table for candidate brands.
        Read this before trusting `chosen_brand` in config.yaml."""
        df = self._load()
        brands = df[df["inbound"] == False]
        candidates = [b for b in self.config.candidate_brands if b in brands["author_id"].values]
        if not candidates:
            candidates = list(brands["author_id"].value_counts().head(4).index)

        records = []
        for brand in candidates:
            brand_replies = brands[brands["author_id"] == brand]
            boilerplate_pct = brand_replies["text"].str.contains(BOILERPLATE_PATTERN, na=False).mean() * 100
            threading_pct = brand_replies["in_response_to_tweet_id"].notna().mean() * 100
            records.append(
                {
                    "brand": brand,
                    "total_replies": len(brand_replies),
                    "boilerplate_pct": round(boilerplate_pct, 2),
                    "threading_valid_pct": round(threading_pct, 2),
                }
            )
        profile_df = pd.DataFrame(records)
        logger.info(f"brand profile:\n{profile_df.to_string(index=False)}")
        return profile_df

    def create_subsample(self) -> pd.DataFrame:
        df = self._load()
        brands = df[df["inbound"] == False]
        mask = brands["author_id"] == self.config.chosen_brand
        brand_tweet_ids = brands[mask]["tweet_id"]
        customer_tweet_ids = brands[mask]["in_response_to_tweet_id"].dropna()

        subsample = df[df["tweet_id"].isin(brand_tweet_ids) | df["tweet_id"].isin(customer_tweet_ids)].copy()
        subsample.to_csv(self.config.subsample_path, index=False)
        logger.info(f"subsample for brand={self.config.chosen_brand}: {len(subsample):,} rows -> {self.config.subsample_path}")
        return subsample
