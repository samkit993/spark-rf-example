import org.apache.spark._
import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.ml.{Pipeline, PipelineStage}
import org.apache.spark.ml.attribute._
import org.apache.spark.ml.regression.{RandomForestRegressor, RandomForestRegressionModel}
import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer, StringIndexerModel}
import org.apache.spark.ml.feature.{PCA, VectorAssembler}
import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.{HashingTF, Tokenizer}
import org.apache.spark.mllib.linalg.Vector
import org.slf4j.LoggerFactory
import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel, ParamGridBuilder}
import org.apache.spark.sql.{DataFrame, Row, SQLContext}
import org.apache.spark.sql.types.{StructField, DoubleType}

object rf_regressor{
  def main(args:Array[String]){
    
    val trainingDataLocation = args(0)

    val targetColName = args(1)

    val testDataLocation = args(2)

    val conf = new SparkConf().setAppName("RF Regressor")

    val sc = new SparkContext(conf)

    val sqlContext = new SQLContext(sc)

    lazy val logger = LoggerFactory.getLogger(getClass)


    val trainingDf = sqlContext.read
                        .option("header", "true") // Use first line of all files as header
                        .option("inferSchema", "true")
                        .format("com.databricks.spark.csv")
                        .load(trainingDataLocation)
                        .withColumnRenamed(targetColName, "label")

    logger.info("##############################")
    logger.info("Before Metadata:" + trainingDf.schema.map{s => s.name + ":" + s.metadata.toString}.mkString(","))
    logger.info("##############################")

    val testDf = sqlContext.read
                        .option("header", "true") // Use first line of all files as header
                        .option("inferSchema", "true")
                        .format("com.databricks.spark.csv")
                        .load(trainingDataLocation)
                        .withColumnRenamed(targetColName, "label")

   /*
    val trainingDf = _trainingDf.schema(targetColName).dataType.toString match {
      case "IntType" | "IntegerType" => _trainingDf.withColumn(targetColName, _trainingDf(targetColName).cast(DoubleType))
      case _ => _trainingDf
    }
    */

     val colsWithStringIndexerModel = trainingDf.dtypes.filter{case (dname, dtype) => dname != "label"}.map{case (dname, dtype) =>{
                                                                          logger.info(dname + ":" + dtype)
                                                                          if(dtype == "StringType")
                                                                            (dname, dtype, Some(new StringIndexer().setInputCol(dname)))
                                                                          else (dname, dtype , None)
                                                  }
     }
    
    logger.info(s"Columns With StringIndexerModels: ${colsWithStringIndexerModel.mkString(",")}")
    
    // List of column names to assemble as part of feature vector.
    val featuresToAssemble = colsWithStringIndexerModel map {
                                 case(colName, colType, colStrIndexerModel) => if (colStrIndexerModel.isDefined)
                                                                                   colStrIndexerModel.get.getOutputCol
                                                                               else colName
                                 }

    // Define the column transformers for the string type columns.
    val strIndexerModels = colsWithStringIndexerModel.filter{ case(colName, colType, colStrIndexerModel) => colStrIndexerModel.isDefined}
                                                     .map{case(colName, colType, colStrIndexerModel) => colStrIndexerModel.get}

    // Define the assembler to create features vector.
    val assmbModel = new VectorAssembler().setInputCols(featuresToAssemble.toArray).setOutputCol("features")

    val rf = new RandomForestRegressor().setFeaturesCol("features")
                                        .setLabelCol("label")
                                        .setPredictionCol("prediction")

    val paramGrid = new ParamGridBuilder()
                          .addGrid(rf.maxDepth, Array(27))
                          .addGrid(rf.maxBins, Array(800))
                          .addGrid(rf.minInstancesPerNode, Array(10))
                          .build()


     val evaluator = new RegressionEvaluator()
          .setLabelCol("label")
          .setPredictionCol("prediction")
          .setMetricName("rmse")


    val cv = new CrossValidator()
                  .setEstimator(rf)
                  .setEvaluator(evaluator)
                  .setEstimatorParamMaps(paramGrid)
                  .setNumFolds(2)  // Use 3+ in practice

    val preprocessingPipelineStages: List[PipelineStage] = strIndexerModels.toList :+ assmbModel

    val preprocessingPipeline = new Pipeline().setStages(preprocessingPipelineStages.toArray)

    // Run cross-validation, and choose the best set of parameters.
    val preprocessingPipelineModel = preprocessingPipeline.fit(trainingDf)

    val trainingDfTransformed = preprocessingPipelineModel.transform(trainingDf)
    logger.info("##############################")
    logger.info(s"Metadata: ${trainingDfTransformed.schema("features").metadata}")
    logger.info("##############################")

    //logger.info(" cvModel.bestModel.paraent.params:" + chosen_lr.params)
    val mlalgoPipeline = new Pipeline().setStages(Array(cv))

    val mlalgoPipelineModel = mlalgoPipeline.fit(preprocessingPipelineModel.transform(trainingDf))

    logger.info("Used Params:" + mlalgoPipelineModel.explainParams())

    val predictions = mlalgoPipelineModel.stages(0).asInstanceOf[CrossValidatorModel].bestModel.transform(preprocessingPipelineModel.transform(testDf))

    val rmse = evaluator.evaluate(predictions)

    logger.info("RMSE: " + rmse)

  }

}
