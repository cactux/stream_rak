# à placer dans data/images et exécuter depuis là.
rm -f image_test_cat_undersampled/*/*.jpg
mkdir image_test_cat_undersampled

cat list_test | while read file
do
  fichier=$(echo $file | sed -e "s|.*/||")
  cat=$(echo $file | sed -e "s/^..//" -e "s|/.*||")
  echo $fichier - $cat
  mkdir -p image_test_cat_undersampled/$cat
  cp -p image_train/$fichier image_test_cat_undersampled/$cat/
done

