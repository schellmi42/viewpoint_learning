folder='log_DLDL'
mkdir $folder
log='--logFolder '$folder'/'
numPts='--maxNumPts 4096 '
epochs='--maxEpoch 1501 '
f='--f ModelNet40_simplified '
DO='--useDropOut '
VQ='--VQ 4,5,7,8 '
LR='--initLearningRate 0.001 --learningDecayRate 200 --learningDecayFactor 0.75 '
aug='--augment --rotation_axis 012 '
sr='--smallrotations '
bs='--batchSize 8 '
loss='--trackVQLoss '
t='--label_threshold 0.01 '
restore='--restore '
fp='--fix_path '
ne='--no_eval '

args1=$epochs$numPts$f$VQ$LR$bs$t$loss$fp$aug

echo [START]
python viewpoint_learning/code/train_with_DLDL.py $args1 $log'airplane' --cates 'airplane' --gpu 0 
# python viewpoint_learning/code/train_with_DLDL.py $args1 $log'bench' --cates 'bench' --gpu 1 &
# python viewpoint_learning/code/train_with_DLDL.py $args1 $log'bottle' --cates 'bottle' --gpu 2 &
# python viewpoint_learning/code/train_with_DLDL.py $args1 $log'car' --cates 'car' --gpu 3 &
# python viewpoint_learning/code/train_with_DLDL.py $args1 $log'chair' --cates 'chair' --gpu 4 &
# python viewpoint_learning/code/train_with_DLDL.py $args1 $log'sofa' --cates 'sofa' --gpu 5 &
# python viewpoint_learning/code/train_with_DLDL.py $args1 $log'table' --cates 'table' --gpu 6 &
# python viewpoint_learning/code/train_with_DLDL.py $args1 $log'toilet' --cates 'toilet' --gpu 7
# wait
echo [done]
