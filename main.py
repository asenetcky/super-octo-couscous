import tensorflow_data_validation as tfdv
import polars as pl


stats = pl.read_csv("data/spam.csv", encoding = "latin-1").to_pandas()
stats = tfdv.generate_statistics_from_dataframe(stats)


print(
    stats.datasets[0].features[0].string_stats.rank_histogram
)

tfdv.visualize_statistics(stats)
