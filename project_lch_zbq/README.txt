Requirments：
pytorch-cuda:11.7
torchvision
timm
cvzone	pip install cvzone
cv2

directories & files:
best_model: store the best model for each model or training methods
cnn5_model: store the model for every train for CNN4 and CNN5
KD_model: store the model for every train for KD on CNN5
resnet_model: store the model for every train for ResNet34
data: asl_alphabet_train is the training dataset, asl_alphabet_test is the raw data collected by us, and asl_alphabet_test_prem is the best testset.
performance: the video of the application is in it
model: the customized models; the CNN5 is in "CNN5.py" and CNN4 is in "CNN5_modify"
utils: "helpers.py" defined some helpers, especially the "load_checkpoint" function; "test_process.py" is to process our test dataset.
train_resnet.py: train the ResNet34
train_cnn5.py: train the CNN5 from script
train_cnn4.py: train the CNN4 from script
train_KD.py: train the CNN5 with knowledge distillation.
test_model.py: you can run the test in this script without to retrain the model.
video.py: generate the video for showing.

Best Model:
best_CNN4_pretrain_True_imgsize_112_bs_32.pth  (model size=78.1MB; Acc=81.61%) (Our final model)
best_ResNet34_pretrain_True_imgsize_224_bs_32.pth  (model size=181MB; Acc=93.10%)
best_KD_CNN5_ResNet34_imgsize_224_bs_32_epochs_10.pth  (model size=84.8MB; Acc=78.16%)

Run command:
Train ResNet:
python train_resnet.py --img_size=224 --model=ResNet34 --load_pretrain_model=True --epochs=10 --batch_size=32
Train CNN4:
python train_cnn4.py --img_size=112 --model=CNN4 --epochs=10 --batch_size=32
Train CNN5:
python train_cnn5.py --img_size=224 --model=CNN5 --epochs=10 --batch_size=32
Train with KD:
python train_KD.py --img_size=224 --model=CNN5 --epochs=10 --batch_size=32 --apply_knowledge_distillation=True --load_teacher_model=True
Test model performance:
python test_model --img_size=#{use the img_size correlated to the model above}# --model=#{the model you want to test; choice=[ResNet34, CNN5, CNN4]}# --use_KD=#{if the model you test use KD}#