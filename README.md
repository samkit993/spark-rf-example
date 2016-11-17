# spark-rf-example
Spark Random Forest Example Code
Pass training data location, target column name and test data location as command line arguments:

To run it in cluster mode, following command can be used:
$SPARK_HOME/bin/spark-submit --jars /home/username/.ivy2/jars/org.apache.commons_commons-csv-1.1.jar,/home/username/.ivy2/jars/com.univocity_univocity-parsers-1.5.1.jar,/home/username/.ivy2/jars/com.databricks_spark-csv_2.11-1.5.0.jar,/home/username/.ivy2/jars/slf4j-log4j12/jars/slf4j-log4j12-1.7.16.jar,/home/username/.ivy2/jars/slf4j-api-1.7.16.jar --conf spark.driver.extraClassPath="/home/username/.ivy2/jars/org.apache.commons_commons-csv-1.1.jar:/home/username/.ivy2/jars/com.univocity_univocity-parsers-1.5.1.jar:/home/username/.ivy2/jars/com.databricks_spark-csv_2.11-1.5.0.jar:/home/username/.ivy2/jars/slf4j-api-1.7.16.jar:/home/username/.ivy2/jars/slf4j-log4j12-1.7.16.jar" --conf spark.executor.extraClassPath="/home/username/.ivy2/jars/org.apache.commons_commons-csv-1.1.jar:/home/username/.ivy2/jars/com.univocity_univocity-parsers-1.5.1.jar:/home/username/.ivy2/jars/com.databricks_spark-csv_2.11-1.5.0.jar:/home/username/.ivy2/jars/slf4j-log4j12-1.7.16.jar:/home/username/.ivy2/jars/slf4j-api-1.7.16.jar" --deploy-mode cluster --master spark://127.0.0.1:7077 --conf "spark.driver.memory=512m" --conf "spark.executor.memory=512m" --class rf_regressor target/scala-2.10/random-forest-regression-example_2.10-1.0.jar ~/codes/dataset/linear_regression_data.csv label ~/codes/dataset/linear_regression_data.csv