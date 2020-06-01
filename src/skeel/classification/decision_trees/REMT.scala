package skeel.classification.decision_trees

/**
  * Rank Entropy-Based Decision Trees for Monotonic Classification
  *
  * Hu, Q., Che, X., Zhang, L., Zhang, D., Guo, M., & Yu, D. (2011). Rank entropy-based decision trees for monotonic classification. 
  * IEEE Transactions on Knowledge and Data Engineering, 24(11), 2052-2064.
  *
  * @author sergiogvz
  */
class REMT(epsilon:Double=0.01, minSamplesLeaf:Int=2, prune:Boolean=false) extends MonDT(minSamplesLeaf, prune) {


  protected var cutRMI:Double = -1

  def stopCriterionCut():Boolean = {
    cutRMI < epsilon
  }

  def splittingRule():(Int,Double) = {

    val attRMI = for(att <- 0 until nAttributes) yield{
      val possibleCuts = orderedX(att).map(Xtrain(_)(att)).distinct.dropRight(1)

      if(possibleCuts.nonEmpty)
        possibleCuts.zip(possibleCuts.map(rmi(att,_))).maxBy(_._2)
      else (-1.0,-1.0)
    }


    val cutRmiAtt = attRMI.zipWithIndex.maxBy(_._1._2)

    cutRMI = cutRmiAtt._1._2

    (cutRmiAtt._2,cutRmiAtt._1._1)
  }



  def generateTree():MonDT = {
    new REMT(epsilon, minSamplesLeaf,prune)
  }

  private def rmi(att:Int, cutValue:Double):Double = {

    val splitVector = for (i <- Xtrain.indices) yield
      if (Xtrain(i)(att) <= cutValue) 1
      else 2


    var countAtt, countClass, countAttClass:Int = 0

    var rmi = 0.0

    for(i <- Xtrain.indices) {
      countAtt=0
      countClass=0
      countAttClass=0
      for (j <- Xtrain.indices) {
        val ilsj = splitVector(i) <= splitVector(j)

        val ilsjClass = Ytrain(i) <= Ytrain(j)

        if (ilsj) countAtt += 1
        if (ilsjClass) countClass += 1
        if (ilsj && ilsjClass) countAttClass+=1

      }


      rmi += Math.log((countAtt*countClass) / (Xtrain.length*countAttClass).toDouble)
    }

    -rmi/Xtrain.length
  }



}
