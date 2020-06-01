package skeel.classification.decision_trees

import skeel.classification.Classifier
import collection.SortedSet

/**
  * Baseline decision tree. Every decision tree must extend this class.
  *
  * @author sergiogvz
  */
abstract class MonDT(minSamplesLeaf:Int, prune:Boolean) extends Classifier{

  protected var isLeft: Boolean = true
  protected var leftTree: MonDT = _
  protected var rightTree: MonDT = _

  protected var Xtrain:Array[Array[Double]] = _
  protected var Ytrain:Array[Int] = _

  /**
    * Pair defining the cut of the binary tree by (Attribute Index, Value of the cut)
    */
  protected var cut:(Int,Double) = _

  /**
    * Index in the orderedX of the last sample within left set
    */
  protected var cutSample:Int = -1

  protected var nSamplePerClass: Array[Int] = _

  protected var orderedX: Array[Array[Int]]= _



  override def fit(X: Array[Array[Double]], y:Array[Int], nClassesP:Int = -1){

    // Sorting the dataset according to each attribute
    val orderedXP = for ( attX <- X(0).indices.toArray ) yield {
      X.map(_(attX)).zipWithIndex.sortBy(_._1).unzip._2
    }

    fit(X, y, orderedXP, nClassesP,true)
  }

  private def fit(X: Array[Array[Double]], y:Array[Int], orderedXP:Array[Array[Int]], nClassesP:Int,  isLeftP: Boolean){
    //println(X.length)
    super.fit(X,y,nClassesP)
    Xtrain=X
    Ytrain=y
    orderedX=orderedXP
    //orderedY=orderedYP
    isLeft=isLeftP

    if (!stopCriterion){
      cut = splittingRule()
      if (!stopCriterionCut){



        cutSample = orderedX(cut._1).map(Xtrain(_)(cut._1)).lastIndexWhere(_<=cut._2)
        //Splitting the data according to the cut point
        val (left,right) = orderedX(cut._1).splitAt(cutSample+1)


        val Xleft = left.map(Xtrain(_))
        val Yleft = left.map(Ytrain(_))

        val leftMap = left.zipWithIndex.toMap

        val leftOrderedX:Array[Array[Int]] = orderedX.map(_.intersect(left)).map(_.map(leftMap(_)))


        val Xright = right.map(Xtrain(_))
        val Yright = right.map(Ytrain(_))

        val rightMap = right.zipWithIndex.toMap

        val rightOrderedX:Array[Array[Int]] = orderedX.map(_.intersect(right)).map(_.map(rightMap(_)))

        leftTree = generateTree()
        leftTree.fit(Xleft,Yleft,leftOrderedX,nClasses, true)

        rightTree = generateTree()
        rightTree.fit(Xright,Yright,rightOrderedX,nClasses, false)

      }else cut = null


    }

  }

  def predictProba(x:Array[Double]):Array[Double] = {
    if (cut==null){
      if (nSamplePerClass == null) {
        nSamplePerClass = Array.fill(nClasses)(0)
        for (i <- Ytrain) nSamplePerClass(i) += 1
      }

      nSamplePerClass.map(_ / Ytrain.length.toDouble)
    }else if(x(cut._1) <= cut._2) leftTree.predictProba(x)
    else rightTree.predictProba(x)
  }

  override def predict(x:Array[Double]):Int = {
    val probs = predictProba(x)
    val med = getMedianFromPMF(probs)
    med
  }

  private def getMedianFromPMF(x: Array[Double]): Int = {
    var lowerC = -1
    var upperC = -1
    var i = 0

    var cumulative = 0d

    while( i < x.length && lowerC == -1){
      cumulative+=x(i)
      if (cumulative>=0.5) lowerC=i
      i+=1
    }

    i = x.length-1
    cumulative = 0d
    while( i >= 0 && upperC == -1){
      cumulative+=x(i)
      if (cumulative>=0.5) upperC=i
      i-=1
    }

    val meds = Array(lowerC,upperC)

    if((meds(1)-meds(0))%2!=0)
      if(isLeft) meds(0)-=1
      else meds(1)+=1

    (meds(0)+(meds(1)-meds(0))/2d).toInt
  }



  def stopCriterion:Boolean = {
     sameClassNode || (Xtrain.length < 2*minSamplesLeaf)
  }

  def stopCriterionCut():Boolean

  def splittingRule():(Int,Double)

  def generateTree():MonDT


  private def sameClassNode():Boolean = {
    val clas = Ytrain(0)

    var sameClass = true

    var i = 1

    while (sameClass && i < Ytrain.length){
      sameClass = clas == Ytrain(i)
      i+=1
    }

    sameClass
  }




}