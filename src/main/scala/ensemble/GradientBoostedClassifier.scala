package ensemble

import org.apache.spark.mllib.tree.GradientBoostedTrees
import org.apache.spark.mllib.tree.configuration.BoostingStrategy
import org.apache.spark.mllib.tree.model.GradientBoostedTreesModel
import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.mllib.util.MLUtils

/**
  * Created by b3j90 on 29/11/15.
  */
class GradientBoostedClassifier {

  val conf = new SparkConf().setAppName("Random Forest Classifier")
  val sc = new SparkContext(conf)

  // Load and parse the data file.
  val data = MLUtils.loadLibSVMFile(sc, "data/mllib/sample_libsvm_data.txt")
  // Split the data into training and test sets (30% held out for testing)
  val splits = data.randomSplit(Array(0.7, 0.3))
  val (trainingData, testData) = (splits(0), splits(1))

  // Train a GradientBoostedTrees model.
  //  The defaultParams for Classification use LogLoss by default.
  val boostingStrategy = BoostingStrategy.defaultParams("Classification")
  boostingStrategy.setNumIterations(3) // Note: Use more iterations in practice.
  boostingStrategy.treeStrategy.setNumClasses(2)
  boostingStrategy.treeStrategy.setMaxDepth(5)
  //  Empty categoricalFeaturesInfo indicates all features are continuous.
  boostingStrategy.treeStrategy.setCategoricalFeaturesInfo(Map[Int, Int]())

  val model = GradientBoostedTrees.train(trainingData, boostingStrategy)

  // Evaluate model on test instances and compute test error
  val labelAndPreds = testData.map { point =>
    val prediction = model.predict(point.features)
    (point.label, prediction)
  }
  val testErr = labelAndPreds.filter(r => r._1 != r._2).count.toDouble / testData.count()
  println("Test Error = " + testErr)
  println("Learned classification GBT model:\n" + model.toDebugString)

  // Save and load model
  model.save(sc, "GradientBoostedClassifierPath")
  val sameModel = GradientBoostedTreesModel.load(sc, "GradientBoostedClassifierPath")
}
