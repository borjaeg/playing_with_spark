package linear_models

import org.apache.spark.SparkConf
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.{LabeledPoint, StreamingLinearRegressionWithSGD}
import org.apache.spark.streaming.StreamingContext

/**
  * Created by b3j90 on 29/11/15.
  */
class StreamingLinearRegression {

  // Anytime a text file is placed in /training/data/dir the model will update
  // Anytime a text file is placed in /testing/data/dir the model will show predictions

  val conf = new SparkConf().setAppName("Simple Application")
  val ssc = new StreamingContext()
  // Each line should be a data point formatted as (y,[x1,x2,x3])
  val trainingData = ssc.textFileStream("/training/data/dir").map(LabeledPoint.parse).cache()
  val testData = ssc.textFileStream("/testing/data/dir").map(LabeledPoint.parse)

  val numFeatures = 3
  val model = new StreamingLinearRegressionWithSGD()
    .setInitialWeights(Vectors.zeros(numFeatures))

  model.trainOn(trainingData)
  model.predictOnValues(testData.map(lp => (lp.label, lp.features))).print()

  ssc.start()
  ssc.awaitTermination()

}
