import os, contextlib
@contextlib.contextmanager
def maybe_spark():
  try:
    from pyspark.sql import SparkSession
    spark = SparkSession.getActiveSession() or SparkSession.builder.getOrCreate()
    yield spark
  except Exception:
    yield None
