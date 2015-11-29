package ensemble

import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.tree.model.RandomForestModel
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.{SparkContext, SparkConf}

/**
  * Created by b3j90 on 29/11/15.
  */
class RandomForestClassifier {

  val conf = new SparkConf().setAppName("Random Forest Classifier")
  val sc = new SparkContext(conf)

  val data = MLUtils.loadLibSVMFile(sc,"data/mllib/sample_libsvm_data.txt")
  val splits = data.randomSplit(Array(0.7,0.3), seed = 1L)
  val (trainingData, testData) = (splits(0), splits(1))


  val numClasses = 2
  //  Empty categoricalFeaturesInfo indicates all features are continuous.
  val categoricalFeaturesInfo = Map[Int, Int]()
  val numTrees = 3 // Use more in practice
  val featureSubsetStrategy = "auto"
  val impurity = "gini"
  val maxDepth = 4
  val maxBins = 32

  val model = RandomForest.trainClassifier(trainingData, numClasses, categoricalFeaturesInfo,
  numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins)

  val labelsAndPreds = testData map { instance =>
    val prediction = model.predict(instance.features)
    (instance.label, prediction)
  }

  val testErr = labelsAndPreds.filter(x => x._1 == x._2).count.toDouble/testData.count
  println("Test Error = " + testErr)
  println("Learned classification forest model:\n" + model.toDebugString)

  // Save and load model
  model.save(sc, "RandomForestModelPath")
  val sameModel = RandomForestModel.load(sc, "RandomForestModelPath")


}
