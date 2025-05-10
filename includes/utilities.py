# Databricks notebook source
# DBTITLE 1,LIBRARIES
# ENTER YOUR CODE HERE

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import sys
import json
import threading

#Pyspark.SQL

from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.window import Window
from pyspark.sql import SparkSession
from pyspark.sql.streaming import StreamingQueryListener


# Delta and MLflow
from delta.tables import DeltaTable
import mlflow
import mlflow.pyfunc

# Utilities
import time
import re
from datetime import datetime
from datetime import timedelta

#Hugging Face:
from transformers import pipeline

# ML-Flow Section 

from sklearn.metrics import (accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report)

#Additional Setup for Date-Time processing:
spark.conf.set("spark.sql.session.timeZone", "UTC")





# COMMAND ----------

# DBTITLE 1,PATH FOLDERS/MODEL
# Specify the raw tweet path
TWEET_SOURCE_PATH = f"dbfs:/FileStore/tables/raw_tweets/"


USER_NAME = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get().split('@')[0]
USER_DIR = f'/tmp/{USER_NAME}/'
TEST_PATH=USER_DIR + "temp_files/"
BRONZE_CHECKPOINT = USER_DIR + 'bronze.checkpoint'
BRONZE_DELTA = USER_DIR + 'bronze.delta'

SILVER_CHECKPOINT = USER_DIR + 'silver.checkpoint'
SILVER_DELTA = USER_DIR + 'silver.delta'

GOLD_CHECKPOINT = USER_DIR + 'gold.checkpoint'
GOLD_DELTA = USER_DIR + 'gold.delta'

MODEL_NAME = "HF_Sentiment"
#"HF_TWEET_SENTIMENT" #USER_NAME + "_Model"

# https://huggingface.co/finiteautomata/bertweet-base-sentiment-analysis
HF_MODEL_NAME = "finiteautomata/bertweet-base-sentiment-analysis"

# COMMAND ----------

# DBTITLE 1,GIVEN FUNCTIONS

# Function to optimize Delta table if it exists
def optimize_table(path):
    if os.path.exists(path):
        DeltaTable.forPath(spark, path).optimize().executeCompaction()
        print(f"Optimized Delta Table at {path}")
    else:
        print(f"Delta Table at {path} does not exist.")

# This routine requires the paths defined in the includes notebook
# and it clears data from the previous run.
def clear_previous_run() -> bool:
    # delete previous run 
    dbutils.fs.rm(BRONZE_CHECKPOINT, True)
    dbutils.fs.rm(BRONZE_DELTA, True)
    dbutils.fs.rm(SILVER_CHECKPOINT, True)
    dbutils.fs.rm(SILVER_DELTA, True)
    dbutils.fs.rm(GOLD_CHECKPOINT, True)
    dbutils.fs.rm(GOLD_DELTA, True)
    return True

def stop_all_streams() -> bool:
    stopped = False
    for stream in spark.streams.active:
        stopped = True
        stream.stop()
    return stopped


def stop_named_stream(spark: SparkSession, namedStream: str) -> bool:
    stopped = False
    for stream in spark.streams.active:
        if stream.name == namedStream:
            stopped = True 
            stream.stop()
    return stopped

def wait_stream_start(spark: SparkSession, namedStream: str) -> bool:
    started = False
    count = 0
    if started == False and count <= 3:
        for stream in spark.streams.active:
            if stream.name == namedStream:
                started = True
        count += 1
        time.sleep(10)
    return started    

# Function to wait for the Delta table to be ready
def wait_for_delta_table(path, timeout=30, check_interval=2):
    """
    Waits for a Delta table to be available before proceeding.

    Args:
        path (str): Path to the Delta table.
        timeout (int): Maximum wait time in seconds.
        check_interval (int): Time interval to check for table availability.

    Returns:
        bool: True if the table is ready, False otherwise.
    """
    elapsed_time = 0
    while elapsed_time < timeout:
        try:
            if spark.read.format("delta").load(path).count() > 0:
                return True
        except:
            pass
        time.sleep(check_interval)
        elapsed_time += check_interval
    return False

# Function to retrieve streaming statistics
def get_streaming_stats():
    """
    Retrieves streaming statistics such as elapsed time, input row count, and processing time.

    Returns:
        pd.DataFrame: A dataframe containing streaming statistics for active queries.
    """
    data = []
    start_time = None  # Track when the job started

    for q in spark.streams.active:
        progress = q.recentProgress
        if progress:
            for p in progress:
                timestamp = datetime.strptime(p["timestamp"], "%Y-%m-%dT%H:%M:%S.%fZ")

                # Set the start time on the first iteration
                if start_time is None:
                    start_time = timestamp

                elapsed_time = (timestamp - start_time).total_seconds()  # Convert to seconds

                # Check if 'addBatch' exists in 'durationMs' before accessing it
                processing_time = p["durationMs"].get("addBatch", None) if "durationMs" in p else None

                data.append({
                    "query": q.name,
                    "elapsed_time": elapsed_time,  # Time in seconds since job start
                    "input_rows": p.get("numInputRows", 0),  # Default to 0 if missing
                    "processing_time": processing_time,  # Could be None if not available
                    "memory_used": p.get("aggregatedStateOperators", [{}])[0].get("stateMemory", 0) if p.get("aggregatedStateOperators") else 0
                })

    return pd.DataFrame(data)



# COMMAND ----------

# DBTITLE 1,TESTING FUNCTIONS:
#1) Feeding function for the Test folder:
def copy_new_test_files(num_files=50, verbose=True):
    """
    Copy the most recent `num_files` from TWEET_SOURCE_PATH to TEST_PATH.
    Files are renamed using timestamp + counter to simulate streaming ingestion.
    Safe for use with S3/cloudFiles streaming ingestion.
    """
  
    # Ensure test directory exists (don't delete it in streaming context)
    dbutils.fs.mkdirs(TEST_PATH)

    try:
        # Load and convert file listing
        files = dbutils.fs.ls(TWEET_SOURCE_PATH)
        files_df = spark.createDataFrame(files)
    except Exception as e:
        print("❌ Error listing source path files:", str(e))
        return

    # Filter and select most recent JSON files
    top_files = (
        files_df
        .filter(col("name").endswith(".json"))
        .orderBy("modificationTime", ascending=False)
        .limit(num_files)
        .collect()
    )

    if not top_files:
        print("⚠️ No .json files found in source directory.")
        return

    # Timestamp once, then increment counter for filenames
    base_ts = datetime.now().strftime("%Y%m%d%H%M%S")

    for i, f in enumerate(top_files):
        new_name = f"{base_ts}_{i:04d}.json"  # e.g., 20250503_0003.json
        target_path = TEST_PATH + new_name
        dbutils.fs.cp(f.path, target_path)

    if verbose:
        print(f"✅ Copied {len(top_files)} test files to: {TEST_PATH}")


#2) Keep feeding:
# import threading

def continuously_copy_test_files(interval_sec=10, total_rounds=5, num_files=20):
    for _ in range(total_rounds):
        copy_new_test_files(num_files=num_files)
        print(f"📁 Injected {num_files} files.")
        time.sleep(interval_sec)




   


# COMMAND ----------

# DBTITLE 1,UDF's  for HF MODEL

# ╔════════════════════════════════════════════════════════════╗
# ║  1 ▸ build the HF sentiment pipeline once and cache it     ║
# ╚════════════════════════════════════════════════════════════╝
# example (commented):
#   demo_pipe = pipeline("sentiment-analysis",
#                        model=HF_MODEL_NAME,
#                        return_all_scores=True)

def build_hf_sentiment_pipe(batch_size: int):
    """
    Create and return a Hugging Face `pipeline` object configured
    for sentiment analysis with the chosen `batch_size`.
    """
    return pipeline(
        task="sentiment-analysis",
        model=HF_MODEL_NAME,
        return_all_scores=True,
        batch_size=batch_size
    )

# instantiate once so it is shared by the UDF below
hf_pipe = build_hf_sentiment_pipe(batch_size=32)


# ╔════════════════════════════════════════════════════════════╗
# ║  2 ▸ helper UDF – map NEU→POS/NEG and emit a binary flag   ║
# ╚════════════════════════════════════════════════════════════╝
@pandas_udf("struct<label:string, score:double, binary:int>")
def classify_sentiment_udf(text_col: pd.Series) -> pd.DataFrame:
    """
    Return the top sentiment label/score plus a binary target:
      POS ➜ 1, NEG ➜ 0, NEU ➜ POS?1:0
    """
    # make predictions in one go
    preds = hf_pipe(text_col.fillna("").tolist())

    def _transform(scores):
        scores_sorted = sorted(scores, key=lambda s: -s["score"])
        top = scores_sorted[0]
        lbl, scr = top["label"], top["score"]

        if lbl == "POS":
            bin_flag = 1
        elif lbl == "NEG":
            bin_flag = 0
        elif lbl == "NEU":
            bin_flag = 1 if scores_sorted[1]["label"] == "POS" else 0
        else:               # fallback for any unexpected label
            bin_flag = 0
        return lbl, scr, bin_flag

    labels, scores, binaries = zip(*(_transform(row) for row in preds))

    return pd.DataFrame(
        {"label": labels, "score": scores, "binary": binaries}
    )

# ╔════════════════════════════════════════════════════════════╗
# ║  3 ▸ small UDF to pull all “@mentions” from a tweet text   ║
# ╚════════════════════════════════════════════════════════════╝
@udf(ArrayType(StringType()))
def grab_mentions(txt: str):
    return re.findall(r"@\w+", txt) if txt else []
    

# COMMAND ----------

# DBTITLE 1,3) BRONZE STREAM
# ╔════════════════════════════════════════════════════════════╗
# ║  4 ▸ streaming utilities                                  ║
# ╚════════════════════════════════════════════════════════════╝
def start_bronze_stream(src_path=TEST_PATH, sleep_sec=5):
    """Stop running streams, clear checkpoints, and kick off Bronze."""
    print("🔄 halting current streams …")
    for s in spark.streams.active:
        s.stop()
    print("✅ streams stopped")

    for cp in (BRONZE_CHECKPOINT, SILVER_CHECKPOINT, GOLD_CHECKPOINT):
        dbutils.fs.rm(cp, recurse=True)
    print("🧹 checkpoints removed")

    bronze_schema = StructType([
        StructField("date",      StringType()),
        StructField("sentiment", StringType()),
        StructField("text",      StringType()),
        StructField("user",      StringType())
    ])

    bronze_src = (
        spark.readStream.format("cloudFiles")
             .schema(bronze_schema)
             .option("cloudFiles.format", "json")
             .load(src_path)
             .withColumn("source_file", input_file_name())
             .withColumn("processing_time", current_timestamp())
    )

    query = (bronze_src.writeStream.format("delta")
             .outputMode("append")
             .option("checkpointLocation", BRONZE_CHECKPOINT)
             .option("mergeSchema", "true")
             .queryName("bronze_stream")
             .start(BRONZE_DELTA))
    print("🚀 Bronze streaming query started")
    bronze_src.printSchema()
    return query


# COMMAND ----------

# DBTITLE 1,4) SILVER STREAM
def start_silver_stream(sleep_sec=5):
    if not wait_for_delta_table(BRONZE_DELTA, timeout=60):
        print("⚠️  Bronze not ready → skip Silver")
        return
    time.sleep(sleep_sec)

    silver_df = (
        spark.readStream.format("delta").load(BRONZE_DELTA)
             .withColumn(
                 "timestamp",
                 to_timestamp(regexp_replace(col("date"), r" [A-Z]{3} ", " "),
                              "EEE MMM dd HH:mm:ss yyyy"))
             .withColumn("mention", explode(grab_mentions(col("text"))))
             .withColumn("cleaned_text", regexp_replace(col("text"), "@\\w+", ""))
             .filter(col("cleaned_text").isNotNull() & (length(trim(col("cleaned_text"))) > 0))
             .select("timestamp", "mention", "cleaned_text", "sentiment")
    )

    q = (silver_df.writeStream.format("delta")
         .outputMode("append")
         .option("checkpointLocation", SILVER_CHECKPOINT)
         .option("mergeSchema", "true")
         .queryName("silver_stream")
         .start(SILVER_DELTA))
    print("🚀 Silver streaming query started")
    silver_df.printSchema()
    return q



# COMMAND ----------

# DBTITLE 1,5) GOLD STREAM
def start_gold_stream(sleep_sec=5):
    if not wait_for_delta_table(SILVER_DELTA, timeout=60):
        print("⚠️  Silver not ready → skip Gold")
        return
    time.sleep(sleep_sec)

    gold_df = (
        spark.readStream.format("delta").load(SILVER_DELTA)
             .filter(col("cleaned_text").isNotNull() & (length(trim(col("cleaned_text"))) > 0))
             .withColumn("sentiment_struct", classify_sentiment_udf(col("cleaned_text")))
             .withColumn("predicted_sentiment",    col("sentiment_struct.label"))
             .withColumn("predicted_score",        col("sentiment_struct.score"))
             .withColumn("predicted_sentiment_id", col("sentiment_struct.binary"))
             .withColumn(
                 "sentiment_id",
                 expr("CASE WHEN lower(sentiment) = 'positive' THEN 1 ELSE 0 END")
             )
             .drop("sentiment_struct")
    )

    q = (gold_df.writeStream.format("delta")
         .outputMode("append")
         .option("checkpointLocation", GOLD_CHECKPOINT)
         .option("mergeSchema", "true")
         .queryName("gold_stream")
         .trigger(processingTime="10 seconds")
         .start(GOLD_DELTA))
    print("🚀 Gold streaming query started")
    gold_df.printSchema()
    return q


# COMMAND ----------

def monitor_streams(interval_sec=10):
    """
    Poll Spark streams one time, sleeping interval_sec beforehand,
    and return a list of dicts with keys:
      - timestamp (str)
      - query     (stream.name)
      - input_rows
      - processing_time (ms)
    """
    import time
    from datetime import datetime

    # wait for the interval (so you get fresh data)
    time.sleep(interval_sec)

    metrics = []
    for stream in spark.streams.active:
        prog = stream.lastProgress
        if prog:
            metrics.append({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "query":      stream.name or "<unnamed>",
                "input_rows": prog.get("numInputRows", 0),
                "processing_time": prog
                     .get("durationMs", {})\
                     .get("addBatch", None)
            })
    return metrics

def plot_stream_metrics(progress_log):
    """
    Given the output of monitor_streams() (a list of dicts),
    clear the current figure and redraw both charts in-place.
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    from IPython.display import display, clear_output



    # turn on interactive mode
    plt.ion()

    df = pd.DataFrame(progress_log)
    if df.empty:
        print("No data to plot yet.")
        return

    # parse timestamps
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # clear previous figures
    plt.clf()

    # two subplots: input_rows (top) and processing_time (bottom)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    for name, grp in df.groupby("query"):
        ax1.plot(grp["timestamp"], grp["input_rows"],    label=name)
        ax2.plot(grp["timestamp"], grp["processing_time"], label=name)

    ax1.set_ylabel("Input Rows")
    ax1.set_title("Stream: Input Rows Over Time")
    ax1.legend(); ax1.grid(True)

    ax2.set_ylabel("Processing Time (ms)")
    ax2.set_title("Stream: Processing Time Over Time")
    ax2.legend(); ax2.grid(True)

    # rotate & label the x-axis ticks
    plt.xticks(rotation=30)
    plt.xlabel("Time")

    # force draw and pause so the UI updates
    clear_output(wait=True)
    display(fig)

# COMMAND ----------

# DBTITLE 1,6.a) MONITORING:
# ╔════════════════════════════════════════════════════════════╗
# ║  5 ▸ lightweight monitoring helpers                       ║
# ╚════════════════════════════════════════════════════════════╝

def capture_stream_snapshots(poll_interval: int = 10):
    """
    Sleep for poll_interval seconds, then grab one progress snapshot
    per active Spark stream. Returns a list of dicts with:
      - timestamp (str)
      - query     (stream.name)
      - input_rows
      - processing_time (ms)
    """
    import time
    from datetime import datetime

    time.sleep(poll_interval)
    return [
        {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "query":      s.name or "<unnamed>",
            "input_rows": p.get("numInputRows", 0),
            "processing_time": p.get("durationMs", {}).get("addBatch")
        }
        for s in spark.streams.active
        if (p := s.lastProgress)
    ]

# COMMAND ----------

# DBTITLE 1,6.b) PLOTTING RPOGRESS:

def render_data_stream_charts(log_entries):
    
    import pandas as pd
    import matplotlib.pyplot as plt
    from IPython.display import display

    plt.ion()
    df = pd.DataFrame(log_entries)
    if df.empty:
        print("No data to plot yet.")
        return

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    for name, grp in df.groupby("query"):
        ax1.plot(grp["timestamp"], grp["input_rows"],    label=name)
        ax2.plot(grp["timestamp"], grp["processing_time"], label=name)

    ax1.set_ylabel("Input Rows")
    ax1.set_title("Stream Input Rows Over Time")
    ax1.legend(); ax1.grid(True)

    ax2.set_ylabel("Processing Time (ms)")
    ax2.set_title("Stream Processing Time Over Time")
    ax2.legend(); ax2.grid(True)

    plt.xticks(rotation=30)
    plt.xlabel("Time")
    display(fig)
    plt.close(fig)


# COMMAND ----------

# DBTITLE 1,8) MLFlow REGISTRATION
# ╔════════════════════════════════════════════════════════════╗
# ║  6 ▸ MLflow helper                                         ║
# ╚════════════════════════════════════════════════════════════╝
def log_experiment_to_mlflow(
        y_true, y_pred,
        model_name: str,
        silver_delta_path: str,
        run_name: str,
        extra_params: dict | None = None
):
    """Record metrics, confusion matrix and params for one run."""
    with mlflow.start_run(run_name=run_name):
        rep = classification_report(y_true, y_pred, output_dict=True)
        for m in ("accuracy",):
            mlflow.log_metric(m, rep[m])
        for m in ("precision", "recall", "f1-score"):
            mlflow.log_metric(m, rep["weighted avg"][m])

        # confusion matrix plot
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Confusion Matrix")
        fig.tight_layout(); fig.savefig("cm.png"); mlflow.log_artifact("cm.png"); plt.close(fig)

        mlflow.log_params({"model_name": model_name, "mlflow_version": mlflow.__version__})
        if extra_params:
            mlflow.log_params(extra_params)

        version = DeltaTable.forPath(spark, silver_delta_path)\
                            .history(1).collect()[0]["version"]
        mlflow.log_param("silver_table_version", version)


# COMMAND ----------



# COMMAND ----------

# DBTITLE 1,LOAD DATAFILES
def fetch_bronze_table():
    return spark.read.format("delta").load(BRONZE_DELTA)

def fetch_silver_table():
    return spark.read.format("delta").load(SILVER_DELTA)

def fetch_gold_table():
    return spark.read.format("delta").load(GOLD_DELTA)


# COMMAND ----------

# MAGIC %md
# MAGIC  **Bench Marking HF BatchSize:**
# MAGIC   
# MAGIC | Batch Size | Time (sec) | Tweets per Second |
# MAGIC |------------|------------|-------------------|
# MAGIC | 1          | 51.38      | 9.96              |
# MAGIC | 8          | 16.68      | 30.70             |
# MAGIC | 16         | 13.88      | 36.89             |
# MAGIC | 32         | 16.06      | 31.88             |
# MAGIC | 64         | 12.42      | 41.23             |
# MAGIC | 128        | 12.16      | 42.11             |
# MAGIC

# COMMAND ----------

# DBTITLE 1,BENCHMARK HF BATCH SIZE
# ╔════════════════════════════════════════════════════════════╗
# ║  7 ▸ tiny benchmarking utility                             ║
# ╚════════════════════════════════════════════════════════════╝
def benchmark_hf_pipe():
    """Run the same sentence through different batch sizes and time it."""
    texts = ["I love this!"] * 512
    results = []
    for bs in [1, 8, 16, 32, 64, 128]:
        p = pipeline("sentiment-analysis", model=HF_MODEL_NAME,
                     return_all_scores=True, batch_size=bs)
        t0 = time.time(); p(texts); dt = time.time() - t0
        results.append({"batch_size": bs, "time_sec": dt,
                        "tweets_per_sec": len(texts) / dt})
    df = pd.DataFrame(results); display(df); return df




# COMMAND ----------

# DBTITLE 1,IDLE WAIT and STOP
# ╔════════════════════════════════════════════════════════════╗
# ║  8 ▸ watch a StreamingQuery until idle                     ║
# ╚════════════════════════════════════════════════════════════╝
def wait_until_idle(query,
                    idle_rounds=3,
                    poll_sec=10,
                    verbose=True):
    """Stop `query` after `idle_rounds` successive polls with 0 new rows."""
    idle = 0; history = []

    while query.isActive:
        prog = query.lastProgress or {}
        rows = prog.get("numInputRows", 0)
        dur  = prog.get("durationMs", {}).get("addBatch")
        ts   = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

        if prog:
            history.append({"timestamp": ts, "input_rows": rows,
                            "processing_time_ms": dur, "query": query.name})
            if verbose:
                print(f"🕒 {ts} | rows={rows} | addBatch={dur} ms")
        else:
            if verbose: print("⏳ waiting for first progress message …")

        idle = idle + 1 if rows == 0 else 0
        if idle >= idle_rounds:
            if verbose: print("✅ stream idle → stopping")
            query.stop(); break
        time.sleep(poll_sec)

    return history

# COMMAND ----------

