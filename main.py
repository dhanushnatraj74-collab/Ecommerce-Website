import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode, regexp_extract, sum as Fsum
from pyspark.ml.feature import StringIndexer
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator


def main():

    print("\nStarting Recommendation Engine...")

    # s Start Spark Session
    spark = (
        SparkSession.builder
        .appName("EcommerceRecommendationSystem")
        .master("local[*]")
        .config("spark.hadoop.validateOutputSpecs", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    # Load Dataset
    data_path = r"C:/Users/Admin/Desktop/EcommercePlatform/data/retaildata.csv"
    df_raw = spark.read.csv(data_path, header=True, inferSchema=True)

    print("\n STEP 1: RAW DATA SAMPLE")
    df_raw.show(10, truncate=False)

    #  Data Cleaning
    df_clean = (
        df_raw.dropna(subset=["CustomerID", "StockCode", "Quantity"])
        .filter(~col("InvoiceNo").startswith("C"))
        .filter(col("Quantity") > 0)
    )

    print("\n STEP 2: CLEANED DATA SAMPLE")
    df_clean.show(10, truncate=False)

    # Dataset Summary
    
    print(" DATASET SUMMARY")
    
    print(f"➡ Total Rows: {df_raw.count()}")
    print(f"➡ Unique Customers: {df_raw.select('CustomerID').distinct().count()}")
    print(f"➡ Unique Products: {df_raw.select('StockCode').distinct().count()}")

    # FIXED NULL REPORT (Correct Version)
    print("\n NULL VALUE REPORT")
    null_counts = df_raw.select(
        *[(Fsum(col(c).isNull().cast("int"))).alias(c) for c in df_raw.columns]
    )
    null_counts.show(truncate=False)

    # Product Category Extract
    df_clean = df_clean.withColumn("Category", regexp_extract(col("StockCode"), r"([A-Za-z]+)", 1))

    # CLV Insights
    print("\n TOP CUSTOMERS (CLV)")
    df_clean.groupBy("CustomerID") \
        .agg(Fsum(col("Quantity") * col("UnitPrice")).alias("TotalSpend")) \
        .orderBy(col("TotalSpend").desc()) \
        .show(10)

    #  User-Product Interaction Matrix
    interactions = (
        df_clean.groupBy("CustomerID", "StockCode")
        .agg(Fsum("Quantity").alias("rating"))
    )
    print("\n Interaction Matrix Sample:")
    interactions.show(10)

    # Encoding IDs
    user_indexer = StringIndexer(inputCol="CustomerID", outputCol="userId", handleInvalid="skip")
    item_indexer = StringIndexer(inputCol="StockCode", outputCol="itemId", handleInvalid="skip")

    data_indexed = user_indexer.fit(interactions).transform(interactions)
    data_indexed = item_indexer.fit(data_indexed).transform(data_indexed)

    # Add product names
    product_info = df_clean.select("StockCode", "Description").dropDuplicates(["StockCode"])
    data_indexed = data_indexed.join(product_info, "StockCode", "left")

    # Train/Test
    ratings = data_indexed.select("userId", "itemId", "rating")
    train, test = ratings.randomSplit([0.8, 0.2], seed=42)

    #  ALS Model
    als = ALS(
        userCol="userId",
        itemCol="itemId",
        ratingCol="rating",
        rank=20,
        maxIter=10,
        regParam=0.1,
        coldStartStrategy="drop",
        implicitPrefs=True
    )

    model = als.fit(train)

    # Evaluate
    predictions = model.transform(test)
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
    rmse = evaluator.evaluate(predictions)
    print(f"\n MODEL ACCURACY (RMSE): {rmse:.4f}")

    #  Generate Recommendations
    user_recs = model.recommendForAllUsers(5)

    user_lookup = data_indexed.select("userId", "CustomerID").dropDuplicates()
    item_lookup = data_indexed.select("itemId", "StockCode", "Description").dropDuplicates()

    final_recs = (
        user_recs
        .select("userId", explode("recommendations").alias("rec"))
        .select("userId", col("rec.itemId"), col("rec.rating").alias("predicted_score"))
        .join(user_lookup, "userId")
        .join(item_lookup, "itemId")
    )

    print("\nFINAL RECOMMENDATION SAMPLE")
    final_recs.show(20, truncate=False)

    #  Save Output as CSV (Safe Windows Method)
    print("\n Saving result to CSV...")
    pdf = final_recs.toPandas()

    output_path = r"C:/Users/Admin/Desktop/EcommercePlatform/output/user_recommendations.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    pdf.to_csv(output_path, index=False)

    print(f"\n SUCCESS! File saved at:\n➡ {output_path}")

    spark.stop()
    print("\n Process Complete — No Errors!")


if __name__ == "__main__":
    main()

