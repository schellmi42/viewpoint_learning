folder='log_ML'
mkdir $folder

numPts='--maxNumPts 4096 '
epochs='--maxEpoch 3001 '
logFolder='--logFolder '$folder' '
f='--f ModelNet40_simplified '
DO='--useDropOut '
VQ='--VQ 4,5,7,8 '
LR='--initLearningRate 0.001 --learningDecayRate 100 --learningDecayFactor 0.75 '
aug='--augment --rotation_axis 012 '
bs='--batchSize 8 '
loss='--trackVQLoss '
t='--label_threshold 0.01 '

args=$epochs$numPts$logFolder$f$DO$VQ$LR$aug$bsi$t$loss
echo [START]
python viewpoint_learning/code/train_with_sigmoid.py $args --affix 'VQs' --cates 'airplane' --gpu 0
# python viewpoint_learning/code/train_with_sigmoid.py $args --affix 'VQs' --cates 'car' --gpu 1 &
# python viewpoint_learning/code/train_with_sigmoid.py $args --affix 'VQs' --cates 'chair' --gpu 2 &
# python viewpoint_learning/code/train_with_sigmoid.py $args --affix 'VQs' --cates 'bench' --gpu 3 &
# python viewpoint_learning/code/train_with_sigmoid.py $args --affix 'VQs' --cates 'bottle' --gpu 4 &
# python viewpoint_learning/code/train_with_sigmoid.py $args --affix 'VQs' --cates 'table' --gpu 5 &
# python viewpoint_learning/code/train_with_sigmoid.py $args --affix 'VQs' --cates 'toilet' --gpu 6 &
# python viewpoint_learning/code/train_with_sigmoid.py $args --affix 'VQs' --cates 'sofa' --gpu 7 
echo [done]
