import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.{SparkContext, SparkConf}

object Clustering {

    def main(args: Array[String]) {
      val logFile = "/Users/b3j90/Documents/Developer/spark-1.5.2-bin-hadoop2.6/README.md"
      val K = 3
      val maxIteration = 500
      val runs = 5
      val conf = new SparkConf().setAppName("Simple Clustering Application")
      val sc = new SparkContext(conf)
      val data = sc.textFile(logFile).map {
        line => Vectors.dense(line.split(',').map(_.toDouble))
      }.cache()

      val clusters: KMeansModel = KMeans.train(data, K, maxIteration, runs)
      val vectorsAndClusterIdx = data.map{ point =>
        val prediction = clusters.predict(point)
        (point.toString, prediction)
      }

    }

}
