wget -c http://visualqa.org/data/mscoco/vqa/Annotations_Train_mscoco.zip --proxy on
wget -c http://visualqa.org/data/mscoco/vqa/Questions_Train_mscoco.zip --proxy on
wget -c http://msvocds.blob.core.windows.net/coco2014/train2014.zip -- proxy on

wget -c http://visualqa.org/data/mscoco/vqa/Annotations_Val_mscoco.zip --proxy on
wget -c http://visualqa.org/data/mscoco/vqa/Questions_Val_mscoco.zip --proxy on
wget -c http://msvocds.blob.core.windows.net/coco2014/val2014.zip --proxy on

unzip Annotations_Train_mscoco.zip
unzip Questions_Train_mscoco.zip
unzip train2014.zip

unzip Annotations_Val_mscoco.zip
unzip Questions_Val_mscoco.zip
unzip val2014.zip