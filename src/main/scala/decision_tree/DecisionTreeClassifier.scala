package decision_tree

import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.model.DecisionTreeModel
import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.mllib.util.MLUtils


class DecisionTreeClassifier {

  val conf = new SparkConf().setAppName("Decision Tree Classification App")
  val sc = new SparkContext(conf)

  val data = MLUtils.loadLibSVMFile(sc, "data/mllib/sample_libsvm_data.txt")
  val splits = data.randomSplit(Array(0.7,0.3))
  val (trainingData, testData) = (splits(0), splits(1))

  // Train a Decision Tree Model
  val numClasses = 2
  val categoricalFeaturesInfo = Map[Int, Int]()
  val impurity = "gini"
  val maxDepth = 5
  val maxBins = 32

  val model = DecisionTree.trainClassifier(trainingData, numClasses, categoricalFeaturesInfo,impurity,
  maxDepth, maxBins)

  // Evaluate model on test instances
  val labelAndPreds = testData.map { point =>
    val prediction = model.predict(point.features)
    (point.label, prediction)
  }

  val testErr = labelAndPreds.filter(x => x._1 == x._2).count.toDouble / testData.count()
  println("Test Error = " + testErr)
  println("Learned classification tree model: \n" + model.toDebugString)


  // Save Model
  model.save(sc, "DecisionTreeModelPath")
  val sameModel = DecisionTreeModel.load(sc, "DecisionTreeModelPath")
}
