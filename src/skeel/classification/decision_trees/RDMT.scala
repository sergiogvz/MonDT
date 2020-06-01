package skeel.classification.decision_trees


/**
  * Rank Discrimination Measure Tree.
  *
  * Marsala, C., & Petturiti, D. (2015). Rank discrimination measures for enforcing monotonicity in decision tree induction. Information Sciences, 291, 143-171.
  *
  * @author sergiogvz
  */
class RDMT(measure:RDM.Value = RDM.Gini ,measureThreshold:Double=0.0, minSamplesLeaf:Int=2, prune:Boolean=false) extends MonDT(minSamplesLeaf, prune) {

  protected var cutRDM:Double = -1


  def stopCriterionCut():Boolean = {
    cutRDM < measureThreshold
  }

  def splittingRule():(Int,Double) = {

    val attRMI = for(att <- 0 until nAttributes) yield{
      val possibleCuts = orderedX(att).map(Xtrain(_)(att)).distinct.dropRight(1)
      //println("Att "+ att +":" + possibleCuts.mkString(" "))
      if(possibleCuts.nonEmpty)
        possibleCuts.zip(possibleCuts.map{ cutValue =>
          measure match {
            case RDM.Gini => gini(att, cutValue)
            case RDM.Shannon => shannon(att, cutValue)
            case RDM.Pessimistic => pessimistic(att, cutValue)
          }
        }).minBy(_._2)

      else (-1.0,-1.0)
    }

    val cutRmiAtt = attRMI.zipWithIndex.minBy(_._1._2)

    cutRDM = cutRmiAtt._1._2

    (cutRmiAtt._2,cutRmiAtt._1._1)
  }



  def generateTree():MonDT = {
    new RDMT(measure,measureThreshold, minSamplesLeaf,prune)
  }

  private def shannon(att:Int, cutValue:Double):Double = {

    val splitVector = for (i <- Xtrain.indices) yield
      if (Xtrain(i)(att) <= cutValue) 1
      else 2


    var countAtt, countAttClass:Int = 0

    var rsdm = 0.0

    for(i <- Xtrain.indices) {
      countAtt=0
      //countClass=0
      countAttClass=0
      for (j <- Xtrain.indices) {
        val ilsj = splitVector(i) <= splitVector(j)

        val ilsjClass = Ytrain(i) <= Ytrain(j)

        if (ilsj) countAtt += 1
        //if (ilsjClass) countClass += 1
        if (ilsj && ilsjClass) countAttClass+=1

      }

      rsdm += Math.log(countAttClass / countAtt.toDouble) / Math.log(2)
    }

    -rsdm/Xtrain.length
  }

  private def gini(att:Int, cutValue:Double):Double = {

    val splitVector = for (i <- Xtrain.indices) yield
      if (Xtrain(i)(att) <= cutValue) 1
      else 2


    var countAtt, countAttClass: Int = 0

    var rmi = 0.0

    for (i <- Xtrain.indices) {
      countAtt = 0
      //countClass = 0
      countAttClass = 0
      for (j <- Xtrain.indices) {
        val ilsj = splitVector(i) <= splitVector(j)

        val ilsjClass = Ytrain(i) <= Ytrain(j)

        if (ilsj) countAtt += 1
        //if (ilsjClass) countClass += 1
        if (ilsj && ilsjClass) countAttClass += 1
      }

      rmi += 1 - (countAttClass / countAtt.toDouble)
    }

    rmi / Xtrain.length
  }

  private def pessimistic(att:Int, cutValue:Double):Double = {

    val splitVector = for (i <- Xtrain.indices) yield
      if (Xtrain(i)(att) <= cutValue) 0
      else 1


    var countAtt, countAttClass:Int = 0



    var minCountClass = Array.fill(2)(Double.MaxValue)

    val attCounts = for(i <- Xtrain.indices) yield {
      countAtt=0
      countAttClass=0
      for (j <- Xtrain.indices) {
        val ilsj = splitVector(i) <= splitVector(j)

        val ilsjClass = Ytrain(i) <= Ytrain(j)

        if (ilsj) countAtt += 1
        if (ilsj && ilsjClass) countAttClass+=1

      }

      if(splitVector(i) == 0 && countAttClass < minCountClass(0)) minCountClass(0) = countAttClass
      if(splitVector(i) == 1 && countAttClass < minCountClass(1)) minCountClass(1) = countAttClass

      countAtt
    }


    var rmi = 0.0

    for(i <- Xtrain.indices){
      val term = minCountClass(splitVector(i))/attCounts(i)

      rmi += (Math.log(term)/Math.log(2))/term
    }

    -rmi/Xtrain.length
  }

}

object RDM extends Enumeration {
  val Gini, Shannon, Pessimistic = Value
}
