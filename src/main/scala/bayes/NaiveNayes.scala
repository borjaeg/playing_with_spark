package bayes


import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.mllib.classification.{NaiveBayes, NaiveBayesModel}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint

/**
  * Created by b3j90 on 29/11/15.
  */
class NaiveNayesClassifier {

  val conf = new SparkConf().setAppName("Naive Bayes App")
  val sc = new SparkContext(conf)

  val data = sc.textFile("data/mllib/sample_naive_bayes_data.txt")
  val parsedData = data.map { line =>
    val parts = line.split(',')
    LabeledPoint(parts(0).toDouble, Vectors.dense(parts(1).split(' ').map(_.toDouble)))
  }
    // Divide amont training set and test set

  val splits = parsedData.randomSplit(Array(0.6,0.4), seed = 11L)
  // Since the training data is only used once, it is not necessary to cache it
  val training = splits(0)
  val test = splits(1)

  val model = NaiveBayes.train(training, lambda = 1.0, modelType= "multinomial")
  val predictionAndLabel = test.map { p =>
    (model.predict(p.features), p.label)
  }
  val accuracy = 1.0 * predictionAndLabel.filter(x => x._1 == x._2).count() / test.count()
  model.save(sc, "naiveBayesModel")
  val sameModel = NaiveBayesModel.load(sc, "naiveBayesModel")
}
