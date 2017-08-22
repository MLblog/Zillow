// Databricks notebook source
// MAGIC %md # XGBoost with Spark DataFrames
// MAGIC 
// MAGIC ![alt text](http://i.imgur.com/iFZNBVx.png "XGBoost")
// MAGIC #### Before You Start: Build and Install XGBoost
// MAGIC In order to run this notebook, you will need to build and install XGBoost. For instructions on how to do this, please refer to **Installing** and **Testing** sections of the [Databricks XGBoost Docs](https://docs.databricks.com/user-guide/faq/xgboost.html). 

// COMMAND ----------

// MAGIC %md ## Build XGBoost Model & Pipeline
// MAGIC #### Import XGBoost Libraries and Prepare Data

// COMMAND ----------

import ml.dmlc.xgboost4j.scala.spark.{DataUtils, XGBoost}

// COMMAND ----------

// Proccessed file with some extra features (like Month) and filled missing values (-1)
val filePath = "/FileStore/tables/kyw38hip1502445633193/train.csv"

// Raw input file as downloaded from kaggle
// val filePath = "/FileStore/tables/jgaz4ouf1503054075960/train_features.csv"

val df = spark.read
        .format("csv")
        .option("header", "true")
        .option("mode", "DROPMALFORMED")
        .option("inferSchema", true)
        .csv(filePath).withColumnRenamed("logerror", "label")

df.printSchema()

// COMMAND ----------

// MAGIC %md
// MAGIC 
// MAGIC <h3> Split the dataset into training and validation </h3>
// MAGIC 
// MAGIC 
// MAGIC In order to reliably evaluate a model we need to eliminate the effect of overfitting. That is achieved by
// MAGIC evaluating our model on data that were not used during training. For this reason we split the data into
// MAGIC training and validation sets.

// COMMAND ----------

val Array(split20, split80) = df.randomSplit(Array(0.20, 0.80), seed = 1800009193L)
val validationSet = split20.cache()
val trainingSet = split80.cache()

// COMMAND ----------

// MAGIC %md
// MAGIC <h2> Lets wrap the XGBoost estimator into a pipeline stage</h2>

// COMMAND ----------

import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.{Param, ParamMap}
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.sql.types._
import org.apache.spark.ml.util._

/*
*  A stage intended to be called after the VectorAssembler. It renames the custom target variable name to "label"
*  so that the DF can be accepted in later estimator stages. It also cleans up the old feature columns since presumable
*  they have already been assembled by the VectorAssembler. Currently not working but only god knows why.
*/

class ColumnRenamer(override val uid: String) extends Transformer with DefaultParamsWritable {
  
  def this() = this(Identifiable.randomUID("colRenamer"))
  
  // Copy constructor, potentially with new params
  def copy(extra: ParamMap): ColumnRenamer = defaultCopy(extra)
  
  final val inputCol = new Param[String](this, "inputCol", "The old name used for the target variable")

  def setInputCol(oldName: String): this.type = set(inputCol, oldName)

  override def transformSchema(schema: StructType): StructType = {
    //schema.add(StructField("label", DoubleType))(Set("features", "labels"))
    //StructType(schema.dropWhile(field => field.name != "features")).add(StructField("label", DoubleType))
    StructType(schema.fields :+ new StructField("label", DoubleType, false))
  }
  
  override def transform(df: Dataset[_]): DataFrame = {
    transformSchema(df.schema, logging = true)
    df.select($(inputCol), "features").withColumnRenamed($(inputCol), "label")
  }
}

// COMMAND ----------

display(trainingSet)

// COMMAND ----------

import ml.dmlc.xgboost4j.scala.spark.XGBoostEstimator
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel, ParamGridBuilder}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.VectorAssembler

class ColumnDropper(override val uid: String) extends Transformer {
  
  def this() = this(Identifiable.randomUID("ColumnDropper"))
  
  // Copy constructor, potentially with new params
  def copy(extra: ParamMap): ColumnDropper = this
  
  final val droppedCols = new Param[Array[String]](this, "droppedCols", "Columns to be dropped")

  def setdroppedCols(cols: Array[String]): this.type = set(droppedCols, cols)


  override def transformSchema(schema: StructType): StructType = {
    StructType(schema.fields filterNot {$(droppedCols) contains _.name})
  }
  
  override def transform(df: Dataset[_]): DataFrame = {
    transformSchema(df.schema, logging = true)
    df.drop($(droppedCols): _*)
  }
}

val tempBadColumns = Array("taxdelinquencyflag")
val tempDropper = new ColumnDropper("tempDropper")
  .setdroppedCols(tempBadColum)

val features: Array[String] = trainingSet.columns.filter(_ != "label")
val assembler = new VectorAssembler()
  .setInputCols(features)
  .setOutputCol("features")

// Drop unused columns 
val dropper = new ColumnDropper("colDropper")
  .setdroppedCols(features)

// Estimator Stage using XGBoost
// Some default parameters found online. These seem to produce a rather good model.
val median = trainingSet.stat.approxQuantile("label", Array(0.5), 0)(0)
val paramMap = List(
  "eta" -> 0.037,
  "max_depth" -> 5,
  "subsample" -> 0.80,
  "objective" -> "reg:linear",
  "eval_metric" -> "mae",
  "lambda" -> 0.8,
  "alpha" -> 0.4,
  "base_score" -> median).toMap
val xgboostEstimator = new XGBoostEstimator(paramMap)

// Combine the stages into the final pipeline       
val pipeline = new Pipeline()
      .setStages(Array(assembler, xgboostEstimator))

val evaluator = new RegressionEvaluator()
  .setLabelCol("label")
  .setPredictionCol("prediction")
  .setMetricName("mae")

val paramGrid = new ParamGridBuilder()
      .addGrid(xgboostEstimator.alpha, Array(0.01, 0.1, 0.2, 0.3))
      .addGrid(xgboostEstimator.maxDepth, Array(5, 6, 7))
      .addGrid(xgboostEstimator.eta, Array(0.01, 0.1, 0.6))
      .build()

val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(4)


val cvModel = cv.fit(trainingSet)

// COMMAND ----------

import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.sql.functions.lit

val raw_features = "/FileStore/tables/bg1pnui51503397757215/train_features.csv"

val raw = spark.read
        .format("csv")
        .option("header", "true")
        .option("mode", "DROPMALFORMED")
        .option("inferSchema", true)
        .csv(raw_features)

// Read and proprocess the raw input file
val acceptedTypes = Array(IntegerType, DoubleType, BooleanType, LongType)
val badFields = raw.schema.fields filterNot {acceptedTypes contains _.dataType} map {_.name}

val df = raw.drop(badFields: _*).na.fill(-1)

// Lets create our input for each of the 6 timestamps required. Same months in different years will be treated the same as our current model does not take the year into account
val oct = df.withColumn("Month", lit(10))
val nov = df.withColumn("Month", lit(11))
val dec = df.withColumn("Month", lit(12))

cvModel.transform(oct)

// COMMAND ----------

// MAGIC %md #### Evaluate Model
// MAGIC 
// MAGIC You can evaluate the XGBoost model using Evaluators from MLlib.

// COMMAND ----------

val predictions = cvModel.transform(validationSet)
val mae = evaluator.evaluate(predictions)

print("grid search tuning achieves: " + mae)

// COMMAND ----------

display(validationSet)

// COMMAND ----------

// MAGIC %md
// MAGIC ## Lets use our trained model to make a submission

// COMMAND ----------

import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.sql.functions.lit

val raw_features = "/FileStore/tables/bg1pnui51503397757215/train_features.csv"

val raw = spark.read
        .format("csv")
        .option("header", "true")
        .option("mode", "DROPMALFORMED")
        .option("inferSchema", true)
        .csv(raw_features)

// Read and proprocess the raw input file
val acceptedTypes = Array(IntegerType, DoubleType, BooleanType, LongType)
val badFields = indexed.schema.fields filterNot {acceptedTypes contains _.dataType} map {_.name}

val df = indexed.drop(badFields: _*).na.fill(-1)

// Lets create our input for each of the 6 timestamps required. Same months in different years will be treated the same as our current model does not take the year into account
val oct = df.withColumn("Month", lit(10))
val nov = df.withColumn("Month", lit(11))
val dec = df.withColumn("Month", lit(12))

//val predictions = cvModel.transform(oct)

// COMMAND ----------

trainingSet.printSchema

// COMMAND ----------


