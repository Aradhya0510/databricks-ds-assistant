from ..common.util_io import maybe_spark
import polars as pl

class DataConnector:
  @staticmethod
  def load_uc_table(table:str, sample_rows:int=50000):
    with maybe_spark() as spark:
      if spark is None: raise RuntimeError("No Spark session. Run on Databricks.")
      df = spark.table(table)
      n_rows = df.count(); cols = df.columns
      sample = df.limit(sample_rows).toPandas()
    return pl.from_pandas(sample), {"rows": n_rows, "cols": len(cols), "columns": cols}

  @staticmethod
  def load_file(path:str, file_format:str|None=None, sample_rows:int=50000):
    fmt = (file_format or ("csv" if path.lower().endswith(".csv") else "parquet")).lower()
    lf = pl.scan_csv(path) if fmt=="csv" else pl.scan_parquet(path)
    return lf.fetch(sample_rows), {"rows": None, "cols": None}
