import os
import json
import shutil
from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p
from nnUNet.nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json

if __name__ == '__main__':
    print("Creating BRATS2023 dataset...")
    # Read ID lists
    with open("/work/H2020DeciderFicarra/asaporita/Progetto_AI_Medical_Imaging/datalist/train.txt", "r") as f:
        train_ids = [line.strip() for line in f if line.strip()]
    with open("/work/H2020DeciderFicarra/asaporita/Progetto_AI_Medical_Imaging/datalist/val.txt", "r") as f:
        val_ids = [line.strip() for line in f if line.strip()]
    with open("/work/H2020DeciderFicarra/asaporita/Progetto_AI_Medical_Imaging/datalist/test.txt", "r") as f:
        test_ids = [line.strip() for line in f if line.strip()]
    print(f"Train IDs: {len(train_ids)}, Val IDs: {len(val_ids)}, Test IDs: {len(test_ids)}")

    training_path = r"/work/grana_neuro/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
    dataset_path  = r"/work/H2020DeciderFicarra/asaporita/Progetto_AI_Medical_Imaging/nnUNet_raw/Dataset001_BraTS2023"
    splits = [{"train": train_ids, "val": val_ids}]

    imagesTr = join(dataset_path, "imagesTr")
    labelsTr = join(dataset_path, "labelsTr")
    imagesTs = join(dataset_path, "imagesTs")
    labelsTs = join(dataset_path, "labelsTs")
    maybe_mkdir_p(imagesTr)
    maybe_mkdir_p(labelsTr)
    maybe_mkdir_p(imagesTs)
    maybe_mkdir_p(labelsTs)
    all_ids = train_ids + val_ids 

    print("Init conversion...")
    for cid in all_ids:
        # FLAIR -> _0000
        shutil.copy(join(training_path, cid, f"{cid}-t2f.nii.gz"), join(imagesTr, f"{cid}_0000.nii.gz"))
        # T1 -> _0001
        shutil.copy(join(training_path, cid, f"{cid}-t1n.nii.gz"), join(imagesTr, f"{cid}_0001.nii.gz"))
        # T1ce -> _0002
        shutil.copy(join(training_path, cid, f"{cid}-t1c.nii.gz"), join(imagesTr, f"{cid}_0002.nii.gz"))
        # T2 -> _0003
        shutil.copy(join(training_path, cid, f"{cid}-t2w.nii.gz"), join(imagesTr, f"{cid}_0003.nii.gz"))
        # Label
        shutil.copy(join(training_path, cid, f"{cid}-seg.nii.gz"), join(labelsTr, f"{cid}.nii.gz"))
        print(f"Processed {cid}")

    for cid in test_ids:
        # FLAIR -> _0000
        shutil.copy(join(training_path, cid, f"{cid}-t2f.nii.gz"), join(imagesTs, f"{cid}_0000.nii.gz"))
        # T1 -> _0001
        shutil.copy(join(training_path, cid, f"{cid}-t1n.nii.gz"), join(imagesTs, f"{cid}_0001.nii.gz"))
        # T1ce -> _0002
        shutil.copy(join(training_path, cid, f"{cid}-t1c.nii.gz"), join(imagesTs, f"{cid}_0002.nii.gz"))
        # T2 -> _0003
        shutil.copy(join(training_path, cid, f"{cid}-t2w.nii.gz"), join(imagesTs, f"{cid}_0003.nii.gz"))
        # Label
        shutil.copy(join(training_path, cid, f"{cid}-seg.nii.gz"), join(imagesTs, f"{cid}.nii.gz"))
        print(f"Processed {cid}")

    print("Creating dataset.json...")
    generate_dataset_json(
        output_folder=dataset_path,
        channel_names={0: 'FLAIR', 1: 'T1', 2: 'T1ce', 3: 'T2'},
        labels={'background': 0, 'NCR': 1, 'ED': 2, 'ET': 3},
        file_ending='.nii.gz',
        num_training_cases=len(all_ids),
        dataset_name='BraTS2023',
    )
    print("Dataset creation completed.")

    preproc_ds = "/work/H2020DeciderFicarra/asaporita/Progetto_AI_Medical_Imaging/nnUNet_preprocessed/Dataset001_BraTS2023"
    maybe_mkdir_p(preproc_ds)
    with open(os.path.join(preproc_ds, "splits_final.json"), "w") as f:
        json.dump(splits, f, indent=4)

    print(f"Created dataset with {len(train_ids)} train, {len(val_ids)} val, {len(test_ids)} test (all placed in imagesTr/labelsTr).")
