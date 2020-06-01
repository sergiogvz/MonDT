
import skeel.classification.decision_trees.{RDM, RDMT, REMT}
import skeel.utils._

/**
  * Example of the execution of Monotonic Decision Trees
  */
object exampleMonDT extends App{


  val train = new MonKeelDataset("./data/artiset-tra.dat")
  val nClasses = train.numClass
  val test = new MonKeelDataset("./data/artiset-tst.dat")


  println("| Algorithm | Accuracy | MAE | NMI |")
  println("| -- | -- | -- | -- |")

  for(alg <-
        (Array(new RDMT(RDM.Gini),new RDMT(RDM.Shannon), new RDMT(RDM.Pessimistic), new REMT(0))
          zip Array("RDMT_Gini", "RDMT_Shannon","RDMT_Pessimistic","REMT"))
  ){

    val dt = alg._1
    val name = alg._2

    dt.fit(train.X, train.y)
    val testPred = dt.predict(test.X)

    val tstScore = new MonotonicScore(test.X, test.y zip testPred, nClasses)

    println("| " + name + " | " + tstScore.accuracy + " | " + tstScore.MAE +  " | " + tstScore.NMI1 + " |")

  }
  
}