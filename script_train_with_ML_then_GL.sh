folder='log_ML-GL'
mkdir $folder
log='--logFolder '$folder'/'
numPts='--maxNumPts 4096 '
epochs='--maxEpoch 1501 '
f='--f ModelNet40_simplified '
DO='--useDropOut '
VQ='--VQ combo '
LR='--initLearningRate 0.001 --learningDecayRate 200 --learningDecayFactor 0.75 '
aug='--augment --rotation_axis 012 '
sr='--smallrotations '
bs='--batchSize 8 '
loss='--trackVQLoss '
t='--label_threshold 0.01 '
restore='--restore '
fp='--fix_path '
ne='--no_eval '

# dummy call to initialize folders before multiprocessing
args1=$epochs$numPts$f$DO$VQ$LR$aug$bs$t$loss$fp$ne
python viewpoint_learning/code/train_with_ML.py $f $VQ --maxEpoch 0 $ne $log'dummy' --cates 'airplane' --gpu 0  
 
args2=$epochs$numPts$f$DO$VQ$LR$aug$bs$loss$fp$restore
echo [START]
python viewpoint_learning/code/train_with_ML.py $args1 $log'airplane' --cates 'airplane' --gpu 0 
# python viewpoint_learning/code/train_with_ML.py $args1 $log'bench' --cates 'bench' --gpu 1 &
# python viewpoint_learning/code/train_with_ML.py $args1 $log'bottle' --cates 'bottle' --gpu 2 &
# python viewpoint_learning/code/train_with_ML.py $args1 $log'car' --cates 'car' --gpu 3 &
# python viewpoint_learning/code/train_with_ML.py $args1 $log'chair' --cates 'chair' --gpu 4 &
# python viewpoint_learning/code/train_with_ML.py $args1 $log'sofa' --cates 'sofa' --gpu 5 &
# python viewpoint_learning/code/train_with_ML.py $args1 $log'table' --cates 'table' --gpu 6 &
# python viewpoint_learning/code/train_with_ML.py $args1 $log'toilet' --cates 'toilet' --gpu 7
wait
echo [MIDDLE]
python viewpoint_learning/code/train_with_GL.py $args2 $log'airplane' --cates 'airplane' --gpu 0
# python viewpoint_learning/code/train_with_GL.py $args2 $log'bench' --cates 'bench' --gpu 1 &
# python viewpoint_learning/code/train_with_GL.py $args2 $log'bottle' --cates 'bottle' --gpu 2 &
# python viewpoint_learning/code/train_with_GL.py $args2 $log'car' --cates 'car' --gpu 3 &
# python viewpoint_learning/code/train_with_GL.py $args2 $log'chair' --cates 'chair' --gpu 4 &
# python viewpoint_learning/code/train_with_GL.py $args2 $log'sofa' --cates 'sofa' --gpu 5 &
# python viewpoint_learning/code/train_with_GL.py $args2 $log'table' --cates 'table' --gpu 6 &
# python viewpoint_learning/code/train_with_GL.py $args2 $log'toilet' --cates 'toilet' --gpu 7 
wait
echo [ENDED]
