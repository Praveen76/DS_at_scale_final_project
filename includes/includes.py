# Databricks notebook source
!pip install --upgrade pip mlflow transformers==4.35.2 emoji==0.6.0 --quiet 

# COMMAND ----------

# restart the kernel after loading dependencies
dbutils.library.restartPython()

# COMMAND ----------

import time
# Set the notebooks starting time.
START_TIME = time.time()

# COMMAND ----------

# Specify the raw tweet path
TWEET_SOURCE_PATH = f"dbfs:/FileStore/tables/raw_tweets/"

# setup storage for this user
#username = spark.sql("SELECT regexp_replace(current_user(), '[^a-zA-Z0-9]', '_')").first()[0]

USER_NAME = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get().split('@')[0]
USER_DIR = f'/tmp/{USER_NAME}/'

BRONZE_CHECKPOINT = USER_DIR + 'bronze.checkpoint'
BRONZE_DELTA = USER_DIR + 'bronze.delta'

SILVER_CHECKPOINT = USER_DIR + 'silver.checkpoint'
SILVER_DELTA = USER_DIR + 'silver.delta'

GOLD_CHECKPOINT = USER_DIR + 'gold.checkpoint'
GOLD_DELTA = USER_DIR + 'gold.delta'

MODEL_NAME = "HF_TWEET_SENTIMENT" #USER_NAME + "_Model"

# https://huggingface.co/finiteautomata/bertweet-base-sentiment-analysis
HF_MODEL_NAME = "finiteautomata/bertweet-base-sentiment-analysis"

# COMMAND ----------

print(f"TWEET_SOURCE_PATH = {TWEET_SOURCE_PATH}")

# COMMAND ----------

# MAGIC %run ./utilities

# COMMAND ----------

displayHTML(f"""
<H2>VERY IMPORTANT TO UNDERSTAND THE USE OF THESE VARIABLES!<br> Please ask if you are confused about their use.</H2>
<table border=1>
<tr><td><b>Variable Name</b></td><td><b>Value</b></td><td><b>Description</b></td></tr>
<tr><td>TWEET_SOURCE_PATH</td><td>{TWEET_SOURCE_PATH}</td><td>Path where the tweets are coming into your system.</td></tr>
<tr><td>USER_DIR</td><td>{USER_DIR}</td><td>Path to the local storage (dbfs) for your project.</td></tr>
<tr><td>BRONZE_CHECKPOINT</td><td>{BRONZE_CHECKPOINT}</td><td>Store your Bronze Checkpoint data here.</td></tr>
<tr><td>BRONZE_DELTA</td><td>{BRONZE_DELTA}</td><td>Store your Bronze Delta Table here.</td></tr>
<tr><td>SILVER_CHECKPOINT</td><td>{SILVER_CHECKPOINT}</td><td>Store your Silver Checkpoint data here.</td></tr>
<tr><td>SILVER_DELTA</td><td>{SILVER_DELTA}</td><td>Store your Silver Delta Table here.</td></tr>
<tr><td>GOLD_CHECKPOINT</td><td>{GOLD_CHECKPOINT}</td><td>Store your Gold Checkpoint data here.</td></tr>
<tr><td>GOLD_DELTA</td><td>{GOLD_DELTA}</td><td>Store your Gold Delta Table here.</td></tr>
<tr><td>MODEL_NAME</td><td>{MODEL_NAME}</td><td>Load this production model</td></tr>
<tr><td>HF_MODEL_NAME</td><td>{HF_MODEL_NAME}</td><td>The Hugging Face Model for Tweet sentiment classification: https://huggingface.co/finiteautomata/bertweet-base-sentiment-analysis </td></tr>
</table>
""")

# COMMAND ----------

print('the includes are included')