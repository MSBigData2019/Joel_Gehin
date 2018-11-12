package com.sparkProject

import org.apache.spark.SparkConf
// import org.apache.spark.sql.SparkSession

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, SaveMode, SparkSession}
import org.apache.spark.ml.feature._
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder, TrainValidationSplit}

object Trainer {

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAll(Map(
      "spark.scheduler.mode" -> "FIFO",
      "spark.speculation" -> "false",
      "spark.reducer.maxSizeInFlight" -> "48m",
      "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
      "spark.kryoserializer.buffer.max" -> "1g",
      "spark.shuffle.file.buffer" -> "32k",
      "spark.default.parallelism" -> "12",
      "spark.sql.shuffle.partitions" -> "12",
      "spark.driver.maxResultSize" -> "2g",
      "spark.debug.maxToStringFields" -> "100"
    ))

    val spark = SparkSession
      .builder
      .config(conf)
      .appName("TP_spark")
      .getOrCreate()


    /*******************************************************************************
      *
      *       TP 3
      *
      *       - lire le fichier sauvegarder précédemment
      *       - construire les Stages du pipeline, puis les assembler
      *       - trouver les meilleurs hyperparamètres pour l'entraînement du pipeline avec une grid-search
      *       - Sauvegarder le pipeline entraîné
      *
      *       if problems with unimported modules => sbt plugins update
      *
      ********************************************************************************/

    // 1. CHARGEMENT DU DATAFRAME
    // --------------------------

    val df: DataFrame = spark
      .read
      .option("header", true)        // Use first line of all files as header
      .option("inferSchema", "true") // Try to infer the data types of each column
      .parquet("/Users/joelgehin/spark-2.2.0-bin-hadoop2.7/prepared_trainingset")

    //println(s"Total number of rows: ${df.count}")
    //println(s"Number of columns ${df.columns.length}")

    //df.show()
    //df.printSchema()


    // 2 UTILISATION DES DONNEES TEXTUELLES
    // ------------------------------------

    // 2.a STAGE 1 = SEPARATION DES TEXTES EN MOTS

    val tokenizer = new RegexTokenizer()
      .setPattern("\\W+")
      .setGaps(true)
      .setInputCol("text")
      .setOutputCol("tokens")


    // 2.b STAGE 2 = RETRAIT DES STOPWORDS

    val remover = new StopWordsRemover()
      .setInputCol(tokenizer.getOutputCol)
      .setOutputCol("clean")


    // 2.c STAGE 3 = PARTIE TF DE TF IDF

    val CV = new CountVectorizer()
      .setInputCol(remover.getOutputCol)
      .setOutputCol("countvect")
      .setMinDF(1)


    // 2.d STAGE 4 = ECRITURE PARTIE TF

    val IDF = new IDF()
      .setInputCol(CV.getOutputCol)
      .setOutputCol("tfidf")


    // 3 CONVERSION DES CATEGORIES EN DONNEES NUMERIQUES
    // -------------------------------------------------

    // 3.e STAGE 5 = CONVERSION NUMERIQUE DE COUNTRY2

    val country_indexer = new StringIndexer()
      .setInputCol("country2")
      .setOutputCol("countryIndex")


    // 3.f STAGE 6 = CONVERSION NUMERIQUE DE CURRENCY2

    val currency_indexer = new StringIndexer()
      .setInputCol("currency2")
      .setOutputCol("currencyIndex")

    // 3.g STAGE 7 = ONEHOTENCODER COUNTRY

    val country_encoder = new OneHotEncoder()
      .setInputCol(country_indexer.getOutputCol)
      .setOutputCol("country_onehot")


    // 3.g STAGE 8 = ONEHOTENCODER CURRENCY

    val currency_encoder = new OneHotEncoder()
      .setInputCol(currency_indexer.getOutputCol)
      .setOutputCol("currency_onehot")


    // 4 MISE EN FORME DES DONNEES POUR SPARK ML
    // -----------------------------------------

    // 4.h STAGE 9 = REGROUPEMENT DES DONNEES POUR SPARK ML

    val assembler = new VectorAssembler()
      .setInputCols(Array("tfidf", "days_campaign", "hours_prepa", "goal", "country_onehot", "currency_onehot"))
      .setOutputCol("features")

    // 4.i STAGE 10 = MODELE DE CLASSIFICATION

    val lr = new LogisticRegression()
      .setElasticNetParam(0.0)
      .setFitIntercept(true)
      .setFeaturesCol("features")
      .setLabelCol("final_status")
      .setStandardization(true)
      .setPredictionCol("predictions")
      .setRawPredictionCol("raw_predictions")
      .setThresholds(Array(0.7, 0.3))
      .setTol(1.0e-6)
      .setMaxIter(300)


    // 4.j PIPELINE D ASSEMBLAGE DES 10 STAGES

    val pipeline = new Pipeline()
      .setStages(Array(tokenizer, remover, CV, IDF, country_indexer,
        currency_indexer, country_encoder,currency_encoder, assembler, lr))



    // 5 ENTRAINEMENT ET TUNING DU MODELE
    // ----------------------------------

    // 5.k SPLIT TRAIN/TEST 90%/10%

    val Array(df_training, df_test) = df.randomSplit(Array(0.9,0.1))


    // 5.l PREPARATION DU GRID SEARCH ET LANCEMENT SUR LE TRAIN

    //Création d'une grille de valeurs à tester pour les hyper-paramètres
    val paramGrid = new ParamGridBuilder()
      .addGrid(CV.minDF, Array(55.0, 75.0, 95.0))
      .addGrid(lr.regParam, Array(1.0e-8, 1.0e-6,1.0e-4,1.0e-2))
      .build()

    //Utilisation de f1-score pour comparer les différents modèles en chaque point de la grille
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("final_status")
      .setPredictionCol("predictions")
      .setMetricName("f1")

    // Préparation du grid-search

    val GridSearchTrainSplit = new TrainValidationSplit()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setTrainRatio(0.7)

    //lancer la grid-search sur le dataset “training” préparé précédemment.
    val CVModel = GridSearchTrainSplit.fit(df_training)

    //val new_df_training = CVModel.transform(df_training)




    // 5.m APPLICATION DU MEILLEUR MODELE AUX DONNEES TEST ET SCORES

    val df_WithPredictions = CVModel.transform(df_test)

    val f1_score = evaluator.evaluate(df_WithPredictions).toString

    println("f1 score:  %s".format(f1_score))




    // 5.n AFFICHAGE DES PREDICTIONS

    df_WithPredictions.groupBy("final_status", "predictions").count.show()



    // SAUVEGARDE DU MODELE

    CVModel.write.overwrite().save(path = "/Users/joelgehin/spark-2.2.0-bin-hadoop2.7/prepared_trainingset/data/Model")


  }
}
