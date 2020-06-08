package com.gupao.bigdata.spark.demo

import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

object LRDemo {

  case class Obs(clas: Double, thickness: Double, size: Double, shape: Double, madh: Double, epsize: Double,
                 bnuc: Double, bchrom: Double, nNuc: Double, mit: Double)

  def parseRDD(rdd: RDD[String]): RDD[Array[Double]] = {
    rdd.map(_.split(",")).filter(_(6) != "?").map(_.drop(1)).map(_.map(_.toDouble))
  }

  def parseObs(line: Array[Double]): Obs = {
    Obs (
      if (line(9) == 4.0) 1 else 0, line(0), line(1), line(2), line(3), line(4), line(5), line(6), line(7), line(8)
    )
  }

  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("Spark MLLib Example")
      .master("local[4]")
      .getOrCreate()

    import spark.implicits._

    val rdd = spark.sparkContext.textFile("D:\\Code\\breast-cancer-wisconsin.data")
    val obsRDD = parseRDD(rdd).map(parseObs)
    val obsDF = obsRDD.toDF().cache()
    obsDF.show()

    val featureCols = Array("thickness", "size", "shape", "madh", "epsize", "bnuc", "bchrom", "nNuc", "mit")
    val assembler = new VectorAssembler().setInputCols(featureCols).setOutputCol("features")
    val df2 = assembler.transform(obsDF)
    df2.show()

    val labelIndexer = new StringIndexer().setInputCol("clas").setOutputCol("label")
    val df3 = labelIndexer.fit(df2).transform(df2)
    df3.show()

    val splitSeed = System.currentTimeMillis()
    val Array(trainingData, testData) = df3.randomSplit(Array(0.7, 0.3), splitSeed)

    val lr = new LogisticRegression().setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.8)
    val model = lr.fit(trainingData)
    println(s"Coefficients: ${model.coefficients} Intercept: ${model.intercept}")

    val predict = model.transform(testData)
    predict.select("label", "prediction").show(50)

    val evaluator = new BinaryClassificationEvaluator().setLabelCol("label")
    val accuracy = evaluator.evaluate(predict)
    println(s"Accuracy: ${accuracy}")
    Console.readInt()

  }
}
