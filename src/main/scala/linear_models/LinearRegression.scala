package linear_models

import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.{LabeledPoint, LinearRegressionModel, LinearRegressionWithSGD}
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by b3j90 on 29/11/15.
  */

class LinearRegression {

  class LogisticRegression {

    val conf = new SparkConf().setAppName("Simple Application")
    val sc = new SparkContext(conf)
    val data =sc.textFile("path/to/file")
    val parsedData = data.map { line =>
      val parts = line.split(",")
      LabeledPoint(parts(0).toDouble, Vectors.dense(parts(1).split(' ').map(_.toDouble)))
    }.cache()

    // Building the Model
    val numIterations = 100
    val model = LinearRegressionWithSGD.train(parsedData, numIterations)

    // Evaluate model on training example and compute training error

    val valuesAndPreds = parsedData.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }

    val MSE = valuesAndPreds.map{
      case (v,p) => math.pow((v-p),2)
    }.mean()

    println("training Mean Squared Error = " + MSE)

    model.save(sc, "LinearRegreesion")
    val sameModel = LinearRegressionModel.load(sc, "LinearRegreesion")

    //RidgeRegressionWithSGD and LassoWithSGD can be used in a similar fashion as LinearRegressionWithSGD.

  }

}
