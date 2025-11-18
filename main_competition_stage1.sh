# pip install numpy==1.26.4 scipy==1.15.3 -i https://pypi.tuna.tsinghua.edu.cn/simple
# pip install torch-summary==1.4.4 thop

DATA_PATH="../Deception_datasets/MMDD2025_features_unify"

FUSION_TYPE_list=("mult") # "mult" "transformer" "senet" "concat" 

MODAL_list=("vaf") # "v" "a" "f" "va" "af" "vf" "vaf" # v-affect+openface (behaviour), a-audio, f-face frames

TRAIN_DATASET_list=("All") #  "BagOfLies" "MU3D" "RLT" "DOLOS" "MDPE" "BoxOfLies"
TRAIN_DATASET_PATH_list=("$DATA_PATH/MU3D/MU3D_features.pkl","$DATA_PATH/RLT/RLT_features.pkl","$DATA_PATH/BagOfLies/BgOL_features.pkl") #"$DATA_PATH/MU3D/MU3D_features.pkl","$DATA_PATH/RLT/RLT_features.pkl","$DATA_PATH/BagOfLies/BgOL_features.pkl","$DATA_PATH/MMDD_stage2_features/DOLOS_train_l464_t365.pkl","$DATA_PATH/MMDD_stage2_features/MDPE_train_balanced_l493_t492.pkl"

# Hyper-parameters
LR_list=(1e-4) # 5e-5 1e-4 2e-4 3e-4 4e-4 5e-4 6e-4 7e-4 8e-4 9e-4 1e-3
BATCH_SIZE_list=(32) # 4 8 16 32 64

# "transformer" 5e-5 8
# "mult" 3e-4 8

for LR in "${LR_list[@]}"
do
for BATCH_SIZE in "${BATCH_SIZE_list[@]}"
do
for FUSION_TYPE in "${FUSION_TYPE_list[@]}"
do
TRAIN_DATASET="All"
TRAIN_DATASET_PATH=${TRAIN_DATASET_PATH_list[$i]}
for MODAL in "${MODAL_list[@]}"
do
echo "****************************************************"
CUDA_VISIBLE_DEVICES=0 python train_test_feature.py \
    --train_dataset $TRAIN_DATASET \
    --train_list $TRAIN_DATASET_PATH \
    --valid_dataset "BoxOfLies" \
    --valid_list '../Deception_datasets/MMDD2025_features_unify/BoxOfLies/BOL_test_features.pkl' \
    --test_dataset "BoxOfLies" \
    --test_list '../Deception_datasets/MMDD2025_features_unify/BoxOfLies/BOL_test_features.pkl' \
    --fusion_type $FUSION_TYPE \
    --modalities $MODAL \
    --lr $LR \
    --batchsize $BATCH_SIZE \
    --log logs1 \
    --testing \
    --adaption
done
done
done
done

# for stage 1
# --test_dataset "BoxOfLies" \
# --test_list '../Deception_datasets/MMDD2025_features/BoxOfLies/BOL_test_features.pkl' \

# for stage 2
# --test_dataset "MMDD2025_final" \
# --test_list '../Deception_datasets/MMDD2025_features/MMDD2025_final_test_features.pkl' \


# for modality in "va" "af" "vf" "vaf"
# do
#   for type in "concat" "transformer" "senet"
#   do
#     CUDA_VISIBLE_DEVICES=1 python train_test_feature.py --train_dataset "MU3D" \
#     --train_list '/pathtofeature/MU3D/MU3D_features.pkl' \
#     --test_dataset "BagOfLies" \
#     --test_list '/pathtofeature/BagOfLies/BgOL_test_features.pkl' \
#     --fusion_type $type \
#     --modalities $modality
#   done
# done