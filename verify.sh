#!/bin/bash

classifiers=("trees.SimpleCart" "bayes.BayesianLogisticRegression" "bayes.BayesNet")

java -cp weka-dev.jar weka.filters.unsupervised.attribute.RemoveByName \
    -E "same.*" -i $1 -o "filtered-$1"

for c in "${classifiers[@]}"
do

    echo "**************************************************************"
    echo "USING CLASSIFIER:" $c
    echo "**************************************************************"
    java -cp weka.jar weka.classifiers.$c \
	 -T "filtered-$1" -c 1 -o -v -i -l $c.model

done

echo "---------------------------------"
echo "   WITH ADITIONAL ATTRIBUTES"
echo "---------------------------------"

for c in "${classifiers[@]}"
do
    echo "************************************************************"
    echo "USING CLASSIFIER:" $c
    echo "************************************************************"
    java -cp weka.jar weka.classifiers.$c \
	 -T $1 -c 1 -o -v -i -l $c.ext.model
done

exit


