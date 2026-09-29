#!/bin/bash

die(){
    echo >&2 "$@"
    echo "Usage: run.sh project session roi_label nifti2rtss_options"
    echo "Data folders expected at /input/nifti, /input/dcm & /output"
    exit 1
}

echo $@

PROJECT=$1
shift
SESSION=$1
shift
ROI_LABEL=$1
shift
OTHER_OPTIONS=$@

if [ -z $PROJECT ] || [ -z $SESSION ] || [ -z $ROI_LABEL ]; then
	die "Project, Session or ROI label not specified."
fi

NIFTI="`ls /input/nifti/*.nii`"
if [ -z $NIFTI ]; then
	die "Failed to find *.nii file in /input/nifti/"
fi

DICOM="`ls /input/dcm/*.dcm`"
if [ -z $DICOM ]; then
        die "Failed to find *.dcm in /input/dcm/"
fi

echo "Converting NIFTI mask to RTStruct using mask in /input/mask  and image files in /input/dcm"
echo "python /pymipl/nifti2rtss.py $OTHER_OPTIONS $NIFTI /input/dcm/ /output/$ROI_LABEL.dcm"

mkdir /tmp/dcm
cp /input/dcm/*.dcm /tmp/dcm/

python /pymipl/nifti2rtss.py $OTHER_OPTIONS $NIFTI /tmp/dcm/ /output/$ROI_LABEL.dcm


echo "RTStruct conversion complete"
RTSTRUCT="`ls /output/*.dcm`"
rm -rf /tmp/dcm
if [ -z $RTSTRUCT ]; then
	die "nifti2rtss.py failed to convert NIFTI to RTStruct"
fi

echo "RTStruct file: " $RTSTRUCT
echo "Uploading to XNAT at: " $XNAT_HOST
echo "Project: " $PROJECT
echo "SESSION: " $SESSION
echo "ROI_LABEL: " $ROI_LABEL

curl -u $XNAT_USER:$XNAT_PASS --data-binary "@$RTSTRUCT" -H "Content-Type: application/octet-stream" -X PUT $XNAT_HOST/xapi/roi/projects/$PROJECT/sessions/$SESSION/collections/$ROI_LABEL?type=RTSTRUCT


