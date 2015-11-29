import org.apache.spark.mllib.classification.{LogisticRegressionModel, LogisticRegressionWithLBFGS}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.mllib.util.MLUtils

/**
  * Created by b3j90 on 29/11/15.
  */
class LogisticRegression {

  val conf = new SparkConf().setAppName("Simple Application")
  val sc = new SparkContext(conf)

  val data = MLUtils.loadLibSVMFile(sc, "/Users/b3j90/Documents/Developer/spark-1.5.2-bin-hadoop2.6/data/mllib/sample_libsvm_data.txt")

  val splits = data.randomSplit(Array(0.6,0.4), seed=11L)
  val training = splits(0).cache()
  val test = splits(1)

  val model = new LogisticRegressionWithLBFGS()
    .setNumClasses(10).run(training)

  val predictionAndLabels = test.map {
    case LabeledPoint(label, features) =>
      val prediction = model.predict(features)
      (prediction, label)
  }

  val metrics = new MulticlassMetrics(predictionAndLabels)
  val precision = metrics.precision
  println("Precision = " + precision)

  model.save(sc, "logisticModel")
  val sameModel = LogisticRegressionModel.load(sc, "logisticModel")

}
