# Databricks notebook source
# MAGIC %md
# MAGIC ## DSCC202-402 Data Science at Scale Final Project
# MAGIC ### Tracking Tweet sentiment at scale using a pretrained transformer (classifier)
# MAGIC <p>Consider the following illustration of the end to end system that you will be building.  Each student should do their own work.  The project will demonstrate your understanding of Spark Streaming, the medalion data architecture using Delta Lake, Spark Inference at Scale using an MLflow packaged model as well as Exploritory Data Analysis and System Tracking and Monitoring.</p>
# MAGIC <br><br>
# MAGIC <img src="https://data-science-at-scale.s3.amazonaws.com/images/pipeline.drawio.png">
# MAGIC
# MAGIC <p>
# MAGIC You will be pulling an updated copy of the course GitHub repositiory: <a href="https://github.com/lpalum/dscc202-402-spring2025">The Repo</a>.  
# MAGIC
# MAGIC Once you have updated your fork of the repository you should see the following template project that is resident in the final_project directory.
# MAGIC </p>
# MAGIC
# MAGIC <img src="https://data-science-at-scale.s3.amazonaws.com/images/notebooks.drawio.png">
# MAGIC
# MAGIC <p>
# MAGIC You can then pull your project into the Databrick Workspace using the <a href="https://github.com/apps/databricks">Databricks App on Github</a> or by cloning the repo to your laptop and then uploading the final_project directory and its contents to your workspace using file imports.  Your choice.
# MAGIC
# MAGIC <p>
# MAGIC Work your way through this notebook which will give you the steps required to submit a complete and compliant project.  The following illustration and associated data dictionary specifies the transformations and data that you are to generate for each step in the medallion pipeline.
# MAGIC </p>
# MAGIC <br><br>
# MAGIC <img src="https://data-science-at-scale.s3.amazonaws.com/images/dataframes.drawio.png">
# MAGIC
# MAGIC #### Bronze Data - raw ingest
# MAGIC - date - string in the source json
# MAGIC - user - string in the source json
# MAGIC - text - tweet string in the source json
# MAGIC - sentiment - the given sentiment of the text as determined by an unknown model that is provided in the source json
# MAGIC - source_file - the path of the source json file the this row of data was read from
# MAGIC - processing_time - a timestamp of when you read this row from the source json
# MAGIC
# MAGIC #### Silver Data - Bronze Preprocessing
# MAGIC - timestamp - convert date string in the bronze data to a timestamp
# MAGIC - mention - every @username mentioned in the text string in the bronze data gets a row in this silver data table.
# MAGIC - cleaned_text - the bronze text data with the mentions (@username) removed.
# MAGIC - sentiment - the given sentiment that was associated with the text in the bronze table.
# MAGIC
# MAGIC #### Gold Data - Silver Table Inference
# MAGIC - timestamp - the timestamp from the silver data table rows
# MAGIC - mention - the mention from the silver data table rows
# MAGIC - cleaned_text - the cleaned_text from the silver data table rows
# MAGIC - sentiment - the given sentiment from the silver data table rows
# MAGIC - predicted_score - score out of 100 from the Hugging Face Sentiment Transformer
# MAGIC - predicted_sentiment - string representation of the sentiment
# MAGIC - sentiment_id - 0 for negative and 1 for postive associated with the given sentiment
# MAGIC - predicted_sentiment_id - 0 for negative and 1 for positive assocaited with the Hugging Face Sentiment Transformer
# MAGIC
# MAGIC #### Application Data - Gold Table Aggregation
# MAGIC - min_timestamp - the oldest timestamp on a given mention (@username)
# MAGIC - max_timestamp - the newest timestamp on a given mention (@username)
# MAGIC - mention - the user (@username) that this row pertains to.
# MAGIC - negative - total negative tweets directed at this mention (@username)
# MAGIC - neutral - total neutral tweets directed at this mention (@username)
# MAGIC - positive - total positive tweets directed at this mention (@username)
# MAGIC
# MAGIC When you are designing your approach, one of the main decisions that you will need to make is how you are going to orchestrate the streaming data processing in your pipeline.  There are several valid approaches to triggering your steams and how you will gate the execution of your pipeline.  Think through how you want to proceed and ask questions if you need guidance. The following references may be helpful:
# MAGIC - [Spark Structured Streaming Programming Guide](https://spark.apache.org/docs/latest/structured-streaming-programming-guide.html)
# MAGIC - [Databricks Autoloader - Cloudfiles](https://docs.databricks.com/en/ingestion/auto-loader/index.html)
# MAGIC - [In class examples - Spark Structured Streaming Performance](https://dbc-f85bdc5b-07db.cloud.databricks.com/editor/notebooks/2638424645880316?o=1093580174577663)
# MAGIC
# MAGIC ### Be sure your project runs end to end when *Run all* is executued on this notebook! (7 points)
# MAGIC
# MAGIC ### This project is worth 25% of your final grade.
# MAGIC - DSCC-202 Students have 55 possible points on this project (see points above and the instructions below)
# MAGIC - DSCC-402 Students have 60 possible points on this project (one extra section to complete)

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

# DBTITLE 1,Pull in the Includes & Utiltites
# MAGIC %run ./includes/includes

# COMMAND ----------

asdasd

# COMMAND ----------

import time

# Set the notebooks start time.
START_TIME = time.time()

# COMMAND ----------

# DBTITLE 1,MODIFIED UTILITIES
# MAGIC %run ./includes/utilities

# COMMAND ----------

# MAGIC %md
# MAGIC ### ADJUST THE MODEL BATCH SIZE:
# MAGIC

# COMMAND ----------

batch_size=256
hf_sentiment=build_hf_sentiment_pipe(batch_size=batch_size)

# COMMAND ----------

# DBTITLE 1,Notebook Control Widgets (maybe helpful)
"""
Adding a widget to the notebook to control the clearing of a previous run.
or stopping the active streams using routines defined in the utilities notebook
"""
dbutils.widgets.removeAll()

#Additional widget:

dbutils.widgets.dropdown("clear_previous_run", "No", ["No","Yes"])
if (getArgument("clear_previous_run") == "Yes"):
    clear_previous_run()
    print("Cleared all previous data.")

dbutils.widgets.dropdown("stop_streams", "No", ["No","Yes"])
if (getArgument("stop_streams") == "Yes"):
    stop_all_streams()
    print("Stopped all active streams.")

dbutils.widgets.dropdown("optimize_tables", "No", ["No","Yes"])
if (getArgument("optimize_tables") == "Yes"):
    # Suck up those small files that we have been appending.
    # Optimize the tables
    optimize_table(BRONZE_DELTA)
    optimize_table(SILVER_DELTA)
    optimize_table(GOLD_DELTA)
    print("Optimized all of the Delta Tables")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.0 Import your libraries here (2 points)
# MAGIC - Are your shuffle partitions consistent with your cluster and your workload?
# MAGIC - Do you have the necessary libraries to perform the required operations in the pipeline/application?

# COMMAND ----------

#Current Partitions as default:
spark.conf.get("spark.sql.shuffle.partitions")


# COMMAND ----------

# Check number of cores
sc.defaultParallelism


# COMMAND ----------

#Then setting it to 8, which is x2 of cores:
spark.conf.set("spark.sql.shuffle.partitions", 8)
spark.conf.set("spark.sql.legacy.timeParserPolicy", "LEGACY")

# COMMAND ----------


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

# MAGIC %md
# MAGIC ## 2.0 Define and execute utility functions (3 points)
# MAGIC - Read the source file directory listing
# MAGIC - Count the source files (how many are there?)
# MAGIC - print the contents of one of the files
# MAGIC

# COMMAND ----------

#A) Read the source file directory

files = dbutils.fs.ls(TWEET_SOURCE_PATH)


# COMMAND ----------

# ───── overview of the source folder ─────
print("The folder contains:")
display(files[:2])

file_total = len(files)
print("*" * 40)
print(f"B) Source directory holds {file_total} files.")
print("*" * 40)

# ───── peek at a single JSON example ─────
print("C) A sample JSON record looks like:")
example_path = files[0].path
example_df   = spark.read.json(example_path)

display(example_df)
print("- Columns present in the JSON:", example_df.columns)


# COMMAND ----------

spark.streams.active

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.0 Transform the Raw Data to Bronze Data using a stream  (8 points)
# MAGIC - define the schema for the raw data
# MAGIC - setup a read stream using cloudfiles and the source data format
# MAGIC - setup a write stream using delta lake to append to the bronze delta table
# MAGIC - enforce schema
# MAGIC - allow a new schema to be merged into the bronze delta table
# MAGIC - Use the defined BRONZE_CHECKPOINT and BRONZE_DELTA paths defined in the includes
# MAGIC - name your raw to bronze stream as bronze_stream
# MAGIC - transform the raw data to the bronze data using the data definition at the top of the notebook
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### ANSWER :
# MAGIC - Written relevant functions in utilities file and imported them here to keep code modular
# MAGIC
# MAGIC
# MAGIC

# COMMAND ----------

# bronze_restart(source_path=TEST_PATH, delay_between_starts=5)
bronze_query=start_bronze_stream(src_path=TWEET_SOURCE_PATH, sleep_sec=5)


# COMMAND ----------

# ── Inspect currently running Spark streams ──
for idx, q in enumerate(spark.streams.active, start=1):
    print(f"▶ Stream {idx}")
    print(f"  • name      : {q.name}")
    print(f"  • is_active : {q.isActive}")
    print(f"  • status    : {q.status}")
    # print(q.recentProgress)  # uncomment to see full progress JSON
    print("‒" * 30)


# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.0 Transform the Bronze Data to Silver Data using a stream (5 points)
# MAGIC - setup a read stream on your bronze delta table
# MAGIC - setup a write stream to append to the silver delta table
# MAGIC - Use the defined SILVER_CHECKPOINT and SILVER_DELTA paths in the includes
# MAGIC - name your bronze to silver stream as silver_stream
# MAGIC - transform the bronze data to the silver data using the data definition at the top of the notebook

# COMMAND ----------

# MAGIC %md
# MAGIC ### ANSWER :
# MAGIC - Written relevant functions in utilities file and imported them here to keep code modular
# MAGIC

# COMMAND ----------

silver_query=start_silver_stream(sleep_sec=5)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5.0 Transform the Silver Data to Gold Data using a stream (7 points)
# MAGIC - setup a read stream on your silver delta table
# MAGIC - setup a write stream to append to the gold delta table
# MAGIC - Use the defined GOLD_CHECKPOINT and GOLD_DELTA paths defines in the includes
# MAGIC - name your silver to gold stream as gold_stream
# MAGIC - transform the silver data to the gold data using the data definition at the top of the notebook
# MAGIC - Load the pretrained transformer sentiment classifier from the MODEL_NAME at the production level from the MLflow registry
# MAGIC - Use a spark UDF to parallelize the inference across your silver data

# COMMAND ----------

# MAGIC %md
# MAGIC #### ASNWER :
# MAGIC - Written relevant functions in utilities file and imported them here to keep code modular
# MAGIC
# MAGIC
# MAGIC
# MAGIC

# COMMAND ----------


gold_query=start_gold_stream(sleep_sec=5)


# COMMAND ----------

print(gold_query.name)
print(gold_query.status)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6.0 Monitor your Streams (5 points)
# MAGIC
# MAGIC - Setup a loop that runs at least every 10 seconds
# MAGIC - Print a timestamp of the monitoring query along with the list of streams, rows processed on each, and the processing time on each
# MAGIC - Run the loop until all of the data is processed (0 rows read on each active stream)
# MAGIC - Plot a line graph that shows the data processed by each stream over time
# MAGIC - Plot a line graph that shows the average processing time on each stream over time

# COMMAND ----------

# MAGIC %md
# MAGIC ### ANSWER 6
# MAGIC
# MAGIC - - Written relevant functions in utilities file and imported them here to keep code modular
# MAGIC

# COMMAND ----------

for s in spark.streams.active:
    print(s.name)
    print(s.status)

# COMMAND ----------

# ENTER YOUR CODE HERE

# CHECK UTILITIES FILE for the funtction:

progress_log = []                     # cumulative history for plotting
idle_rounds  = {}                     # per‑query consecutive‑idle counter
IDLE_LIMIT   = 3                      # how many 0‑row batches → we call it “done”

while True:
    # 1️⃣ poll once (sleep happens inside monitor_streams)
    latest_metrics = monitor_streams(interval_sec=10)
    progress_log.extend(latest_metrics)
    plot_stream_metrics(progress_log)          # live charts
    
    # 2️⃣ print human‑readable summary for this tick
    if latest_metrics:
        print("\n---- Batch Summary -----------------------------------")
        for m in latest_metrics:
            print(f"{m['timestamp']} | {m['query']:<20} "
                  f"rows={m['input_rows']:<6} proc_ms={m['processing_time']}")
    else:
        print("⏳ Streams have not produced progress yet…")

    # 3️⃣ update idle counters
    for m in latest_metrics:
        q = m["query"]
        idle_rounds[q] = idle_rounds.get(q, 0) + (1 if m["input_rows"] == 0 else -idle_rounds.get(q, 0))
    
    # 4️⃣ check global‑idle condition
    if spark.streams.active and all(idle_rounds.get(q.name, 0) >= IDLE_LIMIT
                                    for q in spark.streams.active):
        print("\n✅ All active streams have been idle for "
              f"{IDLE_LIMIT} consecutive polls — monitoring loop exiting.")
        break

    # If there are no active queries at all, break immediately
    if not spark.streams.active:
        print("\n✅ No active streams — monitoring loop exiting.")
        break

# COMMAND ----------

import os
os.getcwd()

# COMMAND ----------

# MAGIC %md
# MAGIC ### **LOAD DATA:**
# MAGIC
# MAGIC

# COMMAND ----------

# Apply IDLE CHECK AND TURN OFF:

gold_log = wait_until_idle(gold_query, idle_rounds=3, poll_sec=10)


# COMMAND ----------

gold_query.awaitTermination()

# COMMAND ----------

# 1) create the DataFrames
bronze_df = fetch_bronze_table()
silver_df = fetch_silver_table()
gold_df   = fetch_gold_table()

# 2) optionally, put them in a list to DRY up your reporting
for name, df in [("Bronze", bronze_df),
                 ("Silver", silver_df),
                 ("Gold",   gold_df)]:
    cnt = df.count()
    print(f"Total tweets in {name}: {cnt}")
    print("-" * 30)
    df.printSchema()


# COMMAND ----------

# MAGIC %md
# MAGIC ## ################################

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7.0 Bronze Data Exploratory Data Analysis (5 points)
# MAGIC
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### Answer:
# MAGIC
# MAGIC 1. **Total tweets in Bronze:**  
# MAGIC    - Determine how many records are stored in the Bronze table.
# MAGIC
# MAGIC 2. **Null/empty field check:**  
# MAGIC    - Identify which columns contain `NULL` or empty strings.  
# MAGIC    - For Silver transformations, explain how I plan to handle those missing values.
# MAGIC
# MAGIC 3. **Filtering in Gold:**  
# MAGIC    - In my Gold dataset, I removed any rows where `Cleaned_Text` is empty (i.e., tweets that became blank after stripping out all `@mentions`), since the model cannot accept null inputs.
# MAGIC
# MAGIC 4. **Tweet count by user:**  
# MAGIC    - Count tweets per unique user handle.  
# MAGIC    - Sort users by descending tweet volume.
# MAGIC
# MAGIC 5. **Mention presence:**  
# MAGIC    - Count how many tweets include at least one `@mention` versus those with none.
# MAGIC
# MAGIC 6. **Top tweeters visualization:**  
# MAGIC    - Create a horizontal bar chart displaying the top 20 users by tweet count.
# MAGIC

# COMMAND ----------

from pyspark.sql.functions import col, when, desc as desc_order
import matplotlib.pyplot as plt

# 1. Count total rows in the Bronze table
tot_tweets = bronze_df.count()
print(f"Total number of tweets in Bronze set: {tot_tweets}")

# 2. Compute null/empty counts for each column using a comprehension
null_counts = [
    (field,
     bronze_df
       .filter(col(field).isNull() | (col(field) == ""))
       .count()
    )
    for field in bronze_df.columns
]
for field, cnt in null_counts:
    print(f"Column `{field}` → {cnt} null/empty values")

# 3. Aggregate by user and sort descending
frequency_by_user = (
    bronze_df
      .groupBy("user")
      .count()
      .orderBy(desc_order("count"))
)
frequency_by_user.show(20, False)

# 4. Mark tweets with at least one '@' and summarize
mention_breakdown = (
    bronze_df
      .withColumn(
          "has_mention",
          when(col("text").contains("@"), 1).otherwise(0)
      )
      .groupBy("has_mention")
      .count()
      .orderBy("has_mention")
)
mention_breakdown.show()

# 5. Visualize the top 20 tweeters
top_20 = frequency_by_user.limit(20).toPandas()

plt.figure(figsize=(12, 8))
plt.barh(y=top_20["user"], width=top_20["count"])
plt.gca().invert_yaxis()             # highest frequency at the top
plt.xlabel("Number of Tweets")
plt.title("Top 20 Tweeters by Tweet Count")
plt.tight_layout()
plt.show()


# COMMAND ----------

# 1) Count tweets that become empty once all “@mentions” are stripped
mention_cleaned = bronze_df.withColumn(
    "text_no_mentions",
    regexp_replace(col("text"), r"@\w+", "")
).withColumn(
    "is_only_mention",
    trim(col("text_no_mentions")) == ""
)
dropped_for_mentions = mention_cleaned.filter(col("is_only_mention")).count()
print(f"Records dropped (only mentions): {dropped_for_mentions}")

# COMMAND ----------

# 2) After mention-removal, tally tweet lengths and show the first 10 lengths
length_distribution = bronze_df.withColumn(
    "text_no_mentions",
    regexp_replace(col("text"), r"@\w+", "")
).withColumn(
    "char_length",
    length(trim(col("text_no_mentions")))
).groupBy("char_length") \
 .count() \
 .orderBy("char_length")
length_distribution.show(10)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8.0 Capture the accuracy metrics from the gold table in MLflow  (4 points)
# MAGIC Store the following in an MLflow experiment run:
# MAGIC - Store the precision, recall, and F1-score as MLflow metrics
# MAGIC - Store an image of the confusion matrix as an MLflow artifact
# MAGIC - Store the model name and the MLflow version that was used as an MLflow parameters
# MAGIC - Store the version of the Delta Table (input-silver) as an MLflow parameter

# COMMAND ----------

# MAGIC %md
# MAGIC ### Answer:
# MAGIC
# MAGIC -- Written relevant functions in utilities file and imported them here to keep code modular
# MAGIC

# COMMAND ----------

#to make sure Gold Table has finised processing the tweets:
gold_query.awaitTermination()

# COMMAND ----------

# 3) Pull the true vs. predicted labels into pandas for downstream eval
eval_df = (
    gold_df
      .select(
          col("sentiment_id").alias("y_true"),
          col("predicted_sentiment_id").alias("y_pred")
      )
      .toPandas()
)
y_true = eval_df["y_true"]
y_pred = eval_df["y_pred"]

# COMMAND ----------

# MAGIC %md
# MAGIC ### ELAPSED TIME TILL END OF GOLD PROCESSING:

# COMMAND ----------

endtime = time.time()
elapsed_seconds = endtime - START_TIME
print(f"Total run time: {timedelta(seconds=elapsed_seconds)}")

# COMMAND ----------


#Call MlFLow :
log_experiment_to_mlflow(
    y_true, y_pred,
    model_name=HF_MODEL_NAME,
    silver_delta_path=SILVER_DELTA,
    run_name=f"batch_size={batch_size}",
    extra_params={"batch_size": batch_size,"elapsed_seconds": timedelta(seconds=elapsed_seconds)}
)


# COMMAND ----------

# MAGIC %md
# MAGIC ## 9.0 Application Data Processing and Visualization (6 points)
# MAGIC - How many mentions are there in the gold data total?
# MAGIC - Count the number of neutral, positive and negative tweets for each mention in new columns
# MAGIC - Capture the total for each mention in a new column
# MAGIC - Sort the mention count totals in descending order
# MAGIC - Plot a bar chart of the top 20 mentions with positive sentiment (the people who are in favor)
# MAGIC - Plot a bar chart of the top 20 mentions with negative sentiment (the people who are the vilians)
# MAGIC
# MAGIC *note: A mention is a specific twitter user that has been "mentioned" in a tweet with an @user reference.

# COMMAND ----------

gold_df.printSchema()

# COMMAND ----------

# 1) Build a mention × sentiment pivot table (counts filled with 0)
mention_freq_df = (
    gold_df
      .filter(col("mention").isNotNull())
      .groupBy("mention")
      .pivot("predicted_sentiment")  # values like "POS", "NEG", "NEU"
      .count()
      .fillna(0)
)

# 2) Add a “total” column summing across sentiment pivots, then sort
sentiment_columns = [c for c in mention_freq_df.columns if c != "mention"]
mention_summary_df = (
    mention_freq_df
      .withColumn(
          "total",
          expr(" + ".join(f"`{c}`" for c in sentiment_columns))
      )
      .orderBy(col("total").desc())
)

# 3) Print overall mention count in the Gold table
total_mentions = gold_df.filter(col("mention").isNotNull()).count()
print(f"✅ Total mention occurrences in Gold: {total_mentions}")

# 4) Display the top 20 mentions by total activity
mention_summary_df.limit(20).show(truncate=False)


# COMMAND ----------

# 5) Plot top mentions for positive sentiment
if "POS" in mention_summary_df.columns:
    pos_df = (
        mention_summary_df
          .select("mention", "POS")
          .orderBy(col("POS").desc())
          .limit(20)
          .toPandas()
    )
    plt.figure(figsize=(10, 6))
    plt.barh(pos_df["mention"], pos_df["POS"])
    plt.gca().invert_yaxis()
    plt.xlabel("Positive Tweet Count")
    plt.title("Top 20 Mentions by Positive Sentiment")
    plt.tight_layout()
    plt.show()


# COMMAND ----------

# 6) Plot top mentions for negative sentiment
if "NEG" in mention_summary_df.columns:
    neg_df = (
        mention_summary_df
          .select("mention", "NEG")
          .orderBy(col("NEG").desc())
          .limit(20)
          .toPandas()
    )
    plt.figure(figsize=(10, 6))
    plt.barh(neg_df["mention"], neg_df["NEG"])
    plt.gca().invert_yaxis()
    plt.xlabel("Negative Tweet Count")
    plt.title("Top 20 Mentions by Negative Sentiment")
    plt.tight_layout()
    plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC ## 10.0 Clean up and completion of your pipeline (3 points)
# MAGIC - using the utilities what streams are running? If any.
# MAGIC - Stop all active streams
# MAGIC - print out the elapsed time of your notebook. Note: In the includes there is a variable START_TIME that captures the starting time of the notebook.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### ANSWER 10:
# MAGIC - In the Monitoring and Plotting section above, **(see ANSWER 6)** I have stopped the streams after checking for idle time.
# MAGIC
# MAGIC

# COMMAND ----------

# 7) List all active streaming queries
print("🔄 Active streams:")
for st in spark.streams.active:
    print(f" • {st.name or '(unnamed)'} | Active: {st.isActive}")

# 8) Stop every running stream
stop_all_streams()

# 9) Show total elapsed time for the notebook
elapsed = time.time() - START_TIME
print(f"⏱️ Notebook Elapsed Time: {timedelta(seconds=int(elapsed))}")



# COMMAND ----------

# 10) Verify no streams remain
remaining = spark.streams.active
if not remaining:
    print("✅ No active streams.")
else:
    print("⚠️ Remaining streams:")
    for st in remaining:
        print(f" – {st.name or '(unnamed)'}")

# COMMAND ----------

# ENTER YOUR CODE HERE
#just restarting streams to keep cluster available:

gold_query = start_gold_stream(sleep_sec=5)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11.0 How Optimized is your Spark Application (Grad Students Only) (5 points)
# MAGIC Graduate students (registered for the DSCC-402 section of the course) are required to do this section.  This is a written analysis using the Spark UI (link to screen shots) that support your analysis of your pipelines execution and what is driving its performance.
# MAGIC Recall that Spark Optimization has 5 significant dimensions of considertation:
# MAGIC - Spill: write to executor disk due to lack of memory
# MAGIC - Skew: imbalance in partition size
# MAGIC - Shuffle: network io moving data between executors (wide transforms)
# MAGIC - Storage: inefficiency due to disk storage format (small files, location)
# MAGIC - Serialization: distribution of code segments across the cluster
# MAGIC
# MAGIC Comment on each of the dimentions of performance and how your impelementation is or is not being affected.  Use specific information in the Spark UI to support your description.  
# MAGIC
# MAGIC Note: you can take sreenshots of the Spark UI from your project runs in databricks and then link to those pictures by storing them as a publicly accessible file on your cloud drive (google, one drive, etc.)
# MAGIC
# MAGIC References:
# MAGIC - [Spark UI Reference Reference](https://spark.apache.org/docs/latest/web-ui.html#web-ui)
# MAGIC - [Spark UI Simulator](https://www.databricks.training/spark-ui-simulator/index.html)

# COMMAND ----------

# MAGIC %md
# MAGIC # Answer: Submited seperate word document for analysis

# COMMAND ----------

# MAGIC %md
# MAGIC ### ENTER YOUR MARKDOWN HERE