import pandas as pd
import json
import time

df = pd.read_csv(r"D:\data science\data engineering\twitter_training.csv")

df.columns = ["id", "entity", "sentiment", "tweet_content"]

tweets = []

for index, row in df.iterrows():

    tweet = {
        "tweet_id": row["id"],
        "entity": row["entity"],
        "sentiment": row["sentiment"],
        "tweet": row["tweet_content"]
    }

    tweets.append(tweet)

    print(json.dumps(tweet, indent=4))

    time.sleep(1)

with open("tweets.json", "w", encoding="utf-8") as f:
    json.dump(tweets, f, indent=4)

print("Dataset converted successfully!")