#!/usr/bin/env bash
set -euo pipefail

if (( $# != 7 )); then
    echo "Usage: $0 XNAT_HOST XNAT_USER XNAT_PASS SESSION_ID RESOURCE_DIR FILE_EXT OUTPUT_DIR" >&2
    exit 2
fi

export XNAT_HOST=$1
export XNAT_USER=$2
export XNAT_PASS=$3
session_id=$4
resource_dir=$5
file_ext=$6
output_dir=$7

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
python3 "$script_dir/download_xnat_resource.py" download-resource \
    "$session_id" NIFTI_LOCATION .csv "$output_dir"
nifti_location_file=$(ls $output_dir/*_NIFTILOCATION.csv)
python3 "$script_dir/download_xnat_resource.py" download-scan-resource-from-csv \
  $nifti_location_file ${resource_dir} ${file_ext}  "$output_dir"
python3 "$script_dir/download_xnat_resource.py" download-scan-resource-from-csv \
  $nifti_location_file NIFTI .nii  "$output_dir"
## get the downloaded file:
mask_file=$(ls $output_dir/*${file_ext})
grayscale_file=$(ls $output_dir/*.nii)
new_mask_file=${mask_file%.nii*}_matched.nii.gz
# python3 "$script_dir/nifti_mask_size_utils.py" \
#     ${mask_file} \
#     ${grayscale_file} \
#     ${new_mask_file}
############################
# original_ct_file=$original_CT_directory_names/
# levelset_infarct_mask_file=${mask_file} ##/${infarctfilename}
# echo "levelset_infarct_mask_file:${levelset_infarct_mask_file}"
# original_ct_file=${grayscale_file}
## preprocessing infarct mask:
python3 "$script_dir/levelset2originalRF_new_flip_utilities.py" \
  levelset2originalRF_new_flip \
  "$grayscale_file" \
  "$mask_file" \
  "$output_dir"
cp "$mask_file"  ${new_mask_file}
###############################
python3 "$script_dir/download_xnat_resource.py" download-scan-resource-folder \
  $nifti_location_file DICOM .dcm "$output_dir"
# python3 /pymipl/nifti2rtss.py \
python3 "$script_dir/nifti2rtss_corrected.py" \
  "${new_mask_file}" \
  "$output_dir" \
  "${new_mask_file%.nii*}.dcm" \
  --structure_labels "${file_ext}" \
  --segmentation_intensities all


PROJECT=$(python3 "$script_dir/download_xnat_resource.py" get-project-name "$session_id") 
SESSION="$session_id" 
mask_file_basename=$(basename $mask_file)
ROI_LABEL="${mask_file_basename%.nii*}_matched"
ROI_NAME="${mask_file_basename%.nii*}_matched"
dcmfilename="${new_mask_file%.nii*}.dcm"

curl -u "$XNAT_USER:$XNAT_PASS" \
  --data-binary "@$dcmfilename" \
  -H "Content-Type: application/octet-stream" \
  -X PUT \
  "$XNAT_HOST/xapi/roi/projects/$PROJECT/sessions/$SESSION/collections/$ROI_LABEL?type=RTSTRUCT&overwrite=true"

  # name=$ROI_NAME&