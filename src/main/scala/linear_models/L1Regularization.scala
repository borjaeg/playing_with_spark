package linear_models

import org.apache.spark.mllib.classification.SVMWithSGD
import org.apache.spark.mllib.optimization.L1Updater
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.{SparkConf, SparkContext}

class L1Regularization {

  val svmAlg = new SVMWithSGD()
  svmAlg.optimizer.setNumIterations(200).
    setRegParam(0.1).setUpdater(new L1Updater)

  val conf = new SparkConf().setAppName("Simple Application")
  val sc = new SparkContext(conf)

  val data = MLUtils.loadLibSVMFile(sc, "/Users/b3j90/Documents/Developer/spark-1.5.2-bin-hadoop2.6/data/mllib/sample_libsvm_data.txt")
  val splits = data.randomSplit(Array(0.6,0.4), seed=11L)
  val training = splits(0).cache()
  val test = splits(1)

  val modelL1 = svmAlg.run(training)
}
