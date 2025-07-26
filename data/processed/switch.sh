for i in X_train_update.csv X_test_update.csv
do
  echo $i
  mv $i ${i}_tmp
  mv ${i}_ $i
  mv ${i}_tmp ${i}_
done
\ls -l
