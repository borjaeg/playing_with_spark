import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.classification.{SVMModel, SVMWithSGD}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.util.MLUtils

object LinearModels {

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("Simple Application")
    val sc = new SparkContext(conf)

    val data = MLUtils.loadLibSVMFile(sc, "/Users/b3j90/Documents/Developer/spark-1.5.2-bin-hadoop2.6/data/mllib/sample_libsvm_data.txt")
    val splits = data.randomSplit(Array(0.6,0.4), seed=11L)
    val training = splits(0).cache()
    val test = splits(1)

    val numIterations = 100
    // It performs by default L2 regularization
    val model = SVMWithSGD.train(training, numIterations)

    model.clearThreshold()

    val scoreAndLabels = test.map { point =>
      val score = model.predict(point.features)
      (score, point.label)
    }

    val metrics = new BinaryClassificationMetrics(scoreAndLabels)
    val auROC = metrics.areaUnderROC()

    println("Area under ROC = " + auROC)

    //model.save(sc, "SVMmodel")
    //val sameModel = SVModel.load(sc, "SVMModel")

  }
}
