'''
This script was derived from: https://github.com/wanderine/nnunetdocker/convert_to_RTSTRUCT.py v. 07.31.2020

Author: Mikhail Milchenko, mmilchenko@wustl.edu
Copyright (c) 2021, Computational Imaging Lab, Washington University School of Medicine

Redistribution and use in source and binary forms, for any purpose, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''

import random, nibabel as nib, argparse, numpy as np, os
from datetime import datetime

from skimage import measure

import pydicom
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence
from pydicom.uid import generate_uid
#from utils import write_rec_file

def concatenate_coordinates(coordinates_x, coordinates_y, coordinates_z):

    vector = np.zeros((len(coordinates_x)*3,1))

    for i in range(len(coordinates_x)):
        vector[i*3+0] = coordinates_x[i] 
        vector[i*3+1] = coordinates_y[i]
        vector[i*3+2] = coordinates_z[i]

    return vector



def sort_dcms_by_slice_pos(input_dicom_path, dcm_files):
    """Sort DICOM image slices along the acquisition slice normal."""
    dcmss = []

    for dcm in dcm_files:
        dcm_path = dcm if os.path.isabs(dcm) else os.path.join(input_dicom_path, dcm)
        ds = pydicom.dcmread(dcm_path, stop_before_pixels=True)

        if 'ImagePositionPatient' not in ds or 'ImageOrientationPatient' not in ds:
            continue

        iop = np.asarray(ds.ImageOrientationPatient, dtype=float)
        row_dir = iop[:3]
        col_dir = iop[3:]
        normal = np.cross(row_dir, col_dir)
        normal_norm = np.linalg.norm(normal)
        if normal_norm == 0:
            continue
        normal = normal / normal_norm

        ipp = np.asarray(ds.ImagePositionPatient, dtype=float)
        slice_position = float(np.dot(ipp, normal))

        dcmss.append(dict(
            file=dcm,
            dataset=ds,
            z=float(ipp[2]),            # retained for compatibility/debugging
            slice_position=slice_position,
            normal=normal,
        ))

    if len(dcmss) < 1:
        raise ValueError(f'No valid image-slice DICOM files found in {input_dicom_path}')

    return sorted(dcmss, key=lambda dcms: dcms['slice_position'])

def create_rtss_dataset(dicoms_sorted,structure_label,series_number=None,series_description=None):
    rf=dicoms_sorted[0]['dataset']

    SOP_class_UID='1.2.840.10008.5.1.4.1.1.481.3'
    SOP_inst_UID,ser_inst_UID=generate_uid(),generate_uid()
    dt0=datetime.min
    date0,time0=dt0.strftime("%Y%m%d"),dt0.strftime("%H%M%S")
    dt=datetime.now()
    date,time=dt.strftime("%Y%m%d"),dt.strftime("%H%M%S")

    meta=Dataset()
    meta.FileMetaInformationGroupLength = 198
    meta.FileMetaInformationVersion = bytes('01', 'utf-8') # '\x00\x01'
    meta.MediaStorageSOPClassUID = SOP_class_UID
    meta.MediaStorageSOPInstanceUID = SOP_inst_UID
    meta.TransferSyntaxUID = '1.2.840.10008.1.2'
    meta.ImplementationClassUID = '1.2.40.0.13.1.1.1'
    meta.ImplementationVersionName = u'1.0'

    r=Dataset()
    r.Manufacturer,r.StructureSetLabel,r.file_meta=u'NRG',structure_label,meta
    r.OperatorsName=u'nifti2rtss'
    r.is_implicit_VR,r.is_little_endian=True,True
    r.SpecificCharacterSet = 'ISO_IR 100'
    r.InstanceCreationDate = date
    r.InstanceCreationTime = time
    r.SOPClassUID=SOP_class_UID
    r.SOPInstanceUID=SOP_inst_UID
    r.InstanceNumber='1'
    if series_number:
        r.SeriesNumber=series_number
    else:
        r.SeriesNumber=str(rf.SeriesNumber)+'01'

    r.StudyDate=rf.StudyDate if 'StudyDate' in rf else date0
    r.StudyTime=rf.StudyTime if 'StudyTime' in rf else time0
    
    '''
    r.AccessionNumber=rf.AccessionNumber if 'AccessionNumber' in rf else None
    r.StudyDescription,r.StudyInstanceUID,r.StudyID=rf.StudyDescription,rf.StudyInstanceUID,rf.StudyID

    r.PatientName,r.PatientID=rf.PatientName,rf.PatientID
    r.PatientBirthDate=''
    r.PatientSex,r.ReferringPhysicianName=rf.PatientSex,rf.ReferringPhysicianName
    '''
    copy_fields=[
        'AccessionNumber',
        'StudyDescription', 'StudyInstanceUID', 'StudyID',
        'PatientName', 'PatientID',
        'PatientSex', 'PatientBirthDate', 'ReferringPhysicianName'
    ]
    for field in copy_fields:
        if field in rf: r.add(rf[field])
    
    r.Modality='RTSTRUCT'
    r.SeriesInstanceUID=ser_inst_UID
    r.SeriesDescription=series_description if series_description else u'RTSS generated by nifti2rtss'
    r.SeriesDate,r.SeriesTime=date,time
    r.StructureSetDate,r.StructureSetTime=date,time
    
    #1. referenced frame of reference sequence
    referenced_frame_of_ref_seq=Sequence()
    r.ReferencedFrameOfReferenceSequence=referenced_frame_of_ref_seq

    #2. Referenced frame of reference #1
    referenced_frame_of_ref1=Dataset()
    referenced_frame_of_ref1.FrameOfReferenceUID=rf.FrameOfReferenceUID
    
    #3. RT referenced study sequence
    rt_referenced_study_seq=Sequence()
    referenced_frame_of_ref1.RTReferencedStudySequence=rt_referenced_study_seq
    

    #4. RT referenced study sequence, study #1
    rt_referenced_study1=Dataset()
    rt_referenced_study1.ReferencedSOPClassUID=SOP_class_UID
    rt_referenced_study1.ReferencedSOPInstanceUID=rf.StudyInstanceUID
    

    #5. RT referenced series sequence
    rt_referenced_series_seq=Sequence()
    rt_referenced_study1.RTReferencedSeriesSequence=rt_referenced_series_seq

    #6. RT referenced series 1
    rt_referenced_series1=Dataset()
    rt_referenced_series1.SeriesInstanceUID=rf.SeriesInstanceUID

    #7. Contour image sequence
    contour_image_sequence=Sequence()
    rt_referenced_series1.ContourImageSequence=contour_image_sequence
 
    #Loop over all DICOM images
    for dcms in dicoms_sorted:  #range(1,numberOfDicomImages+1):
        dstemp = dcms['dataset']
        # Contour Image Sequence: Contour Image
        contour_image = Dataset()
        contour_image.ReferencedSOPClassUID = dstemp.SOPClassUID
        contour_image.ReferencedSOPInstanceUID = dstemp.SOPInstanceUID
        contour_image_sequence.append(contour_image)
   
    #append all sequences
    rt_referenced_series_seq.append(rt_referenced_series1)
    #print('rt_referenced_series_seq', rt_referenced_series_seq)
    rt_referenced_study_seq.append(rt_referenced_study1)
    referenced_frame_of_ref_seq.append(referenced_frame_of_ref1)
    
    #8. Structure set ROI sequence
    structure_set_roi_sequence=Sequence()
    r.StructureSetROISequence=structure_set_roi_sequence

    # Structure set ROI #1
    #structure_set_roi1=Dataset(); ssr1=structure_set_roi1
    #ssr1.ROINumber,ROIName,ROIDescription="1","na","na"
    #ssr1.ROIGenerationAlgorithm='AUTOMATIC'
    #structure_set_roi_sequence.append(ssr1)

    return r
    
def get_valid_dicom_files(input_dicom_path):
    valid_files = []
    required_tags = [
        'SOPClassUID',
        'SOPInstanceUID',
        'PixelSpacing',
        'Rows',
        'Columns',
        'ImagePositionPatient',
        'ImageOrientationPatient',
    ]

    for root, _, files in os.walk(input_dicom_path):
        for f in files:
            path = os.path.join(root, f)
            try:
                # fast check: do not load pixel data
                ds = pydicom.dcmread(path, stop_before_pixels=True, force=False)
                if not all(tag in ds for tag in required_tags):
                    continue
                valid_files.append(path)
            except Exception:
                pass

    if len(valid_files) < 1:
        raise ValueError(f'No valid image-slice DICOM files found in {input_dicom_path}')

    return valid_files
    
def convert(input_nifti_path: str, input_dicom_path: str, output_dicom_path: str,
            structure_labels, segmentation_intensities, structure_set_label,
            poly_approx_tol, min_poly_pts, series_number, series_description):

    if not structure_labels:
        raise ValueError('At least one structure label must be supplied.')
    if segmentation_intensities == 'all':
        if len(structure_labels) != 1:
            raise ValueError("'all' segmentation intensities requires exactly one structure label.")
        numberOfROIs = 1
    else:
        if len(structure_labels) != len(segmentation_intensities):
            raise ValueError('The number of structure labels must match the number of segmentation intensities.')
        numberOfROIs = len(structure_labels)

    if structure_set_label is None:
        structure_set_label = structure_labels[0] if numberOfROIs == 1 else 'MULTI_ROI'

    tol=poly_approx_tol
    
    #---------------
    # First DICOM part
    #---------------

    # Get number of DICOM files in DICOM path
    dicomFiles = get_valid_dicom_files(input_dicom_path)

    numberOfDicomImages = len(dicomFiles)
    # Load template DICOM file header (first file)
    dicomsSorted=sort_dcms_by_slice_pos(input_dicom_path,dicomFiles)

    ds = dicomsSorted[0]['dataset']
    #ds.dir()
    #return
    #print(ds)
    
    rowPixelSize = float(ds.PixelSpacing[0])
    colPixelSize = float(ds.PixelSpacing[1])

    xyPixelSize = 0.5 * (rowPixelSize + colPixelSize)
    poly_approx_tol /= xyPixelSize

    # DICOM slice geometry in patient LPS coordinates.
    iop = np.asarray(ds.ImageOrientationPatient, dtype=float)
    row_direction = iop[:3]
    col_direction = iop[3:]
    slice_normal = np.cross(row_direction, col_direction)
    slice_normal /= np.linalg.norm(slice_normal)

    dicom_slice_positions = np.asarray(
        [float(np.dot(np.asarray(item['dataset'].ImagePositionPatient, dtype=float), slice_normal))
         for item in dicomsSorted],
        dtype=float,
    )

    print('DICOM pixel spacing (row, col):', rowPixelSize, colPixelSize)
    print('DICOM slice normal:', slice_normal)
    print('Number of DICOM slices:', numberOfDicomImages)

    #---------------
    # NIFTI part
    #---------------

    # IMPORTANT:
    # Do not apply an unconditional flip/reorientation here.  The NIfTI affine
    # already defines how voxel indices map into physical RAS coordinates.
    nii = nib.load(input_nifti_path)
    volume = nii.get_fdata().astype(float)

    print('NIfTI file dimensions:', volume.shape)
    print('NIfTI orientation:', nib.aff2axcodes(nii.affine))
    print('NIfTI affine:\n', nii.affine)

    if len(volume.shape) == 4:
        volume = volume[..., 0]
        print('Assuming the first channel of the input NIfTI is the segmentation mask.')
    elif len(volume.shape) == 3:
        print('Segmentation mask is 3D.')
    else:
        raise ValueError(f'Unsupported NIfTI dimension: {volume.shape}')

    if segmentation_intensities == 'all':
        roi_volumes = [volume]
    else:
        roi_volumes = [volume == intensity for intensity in segmentation_intensities]

    # Store contours by DICOM slice, not by NIfTI slice.  This prevents a
    # mismatch when the NIfTI slice order is reversed relative to the DICOMs.
    AllCoordinates = []

    # NIfTI affine maps voxel indices -> RAS. RTSTRUCT requires DICOM LPS.
    ras_to_lps = np.diag([-1.0, -1.0, 1.0, 1.0])
    nifti_to_lps = ras_to_lps @ nii.affine

    # Loop over ROIs and NIfTI slices, extract contours, transform each contour
    # point to patient LPS coordinates, and attach it to the nearest DICOM slice.
    for roi_volume in roi_volumes:
        AllCoordinatesThisROI = [[] for _ in range(numberOfDicomImages)]

        for nifti_slice in range(volume.shape[2]):
            image = roi_volume[:, :, nifti_slice]

            # skimage returns points as (axis0, axis1), which correspond directly
            # to NIfTI voxel indices (i, j) for volume[:, :, k].
            contours = measure.find_contours(image, 0.5)

            for contour in contours:
                cont1 = measure.approximate_polygon(contour, poly_approx_tol)
                nCoordinates = len(cont1)
                if nCoordinates < min_poly_pts:
                    continue

                # Build homogeneous NIfTI voxel coordinates [i, j, k, 1].
                vox = np.column_stack((
                    cont1[:, 0],
                    cont1[:, 1],
                    np.full(nCoordinates, nifti_slice, dtype=float),
                    np.ones(nCoordinates, dtype=float),
                ))

                # Convert voxel coordinates directly to DICOM patient LPS mm.
                contour_lps = (nifti_to_lps @ vox.T).T[:, :3]

                # Match this contour to the nearest DICOM image plane.
                contour_plane_position = float(np.median(contour_lps @ slice_normal))
                dicom_slice_index = int(np.argmin(np.abs(dicom_slice_positions - contour_plane_position)))
                target_plane_position = dicom_slice_positions[dicom_slice_index]

                # CLOSED_PLANAR contours should lie exactly on the referenced
                # DICOM image plane. Project tiny affine/rounding differences
                # onto that plane along the slice normal.
                offsets = target_plane_position - (contour_lps @ slice_normal)
                contour_lps = contour_lps + offsets[:, None] * slice_normal[None, :]

                coordinates = concatenate_coordinates(*contour_lps.T)
                coordinates = np.squeeze(coordinates)

                AllCoordinatesThisROI[dicom_slice_index].append(coordinates)

        AllCoordinates.append(AllCoordinatesThisROI)

    #---------------
    # Second DICOM part (RTstruct)
    #---------------
    # StructureSetLabel describes the whole set, while each ROI has its own name.
    rtds=create_rtss_dataset(dicomsSorted,structure_set_label,series_number,series_description)

    # Structure Set ROI Sequence
    structure_set_roi_sequence = rtds.StructureSetROISequence
    rtds.StructureSetLabel = structure_set_label

    print('Number of ROIs:',numberOfROIs)
    # Loop over ROIs
    for ROI in range(1,numberOfROIs+1):
        # Structure Set ROI Sequence: Structure Set ROI
        structure_set_roi = Dataset()
        structure_set_roi.ROINumber = str(ROI)
        structure_set_roi.ReferencedFrameOfReferenceUID = ds.FrameOfReferenceUID 
        structure_set_roi.ROIName = structure_labels[ROI-1]
        structure_set_roi.ROIGenerationAlgorithm = 'AUTOMATIC'
        structure_set_roi_sequence.append(structure_set_roi)
	
    # ROI Contour Sequence
    roi_contour_sequence = Sequence()
    rtds.ROIContourSequence = roi_contour_sequence

    # Loop over ROI contour sequences
    for ROI in range(1,numberOfROIs+1):

        # ROI Contour Sequence: ROI Contour 1
        roi_contour = Dataset()
        roi_contour.ROIDisplayColor = [0, 230, 0]

        # Contour Sequence
        contour_sequence = Sequence()
        roi_contour.ContourSequence = contour_sequence
        cnumber=0

        # Loop over slices in volume (ROI)
        for slice in range(numberOfDicomImages):

            # Should Contour Sequence be inside this loop?
            #roi_contour.ContourSequence = contour_sequence

            # Loop over contour sequences in this slice
            numberOfContoursInThisSlice = len(AllCoordinates[ROI-1][slice])
            if numberOfContoursInThisSlice < 1: continue

            # Contour Image Sequence
            contour_image_sequence = Sequence()
            contour_image1 = Dataset()
            contour_image1.ReferencedSOPClassUID = ds.SOPClassUID
            contour_image1.ReferencedSOPInstanceUID=dicomsSorted[slice]['dataset'].SOPInstanceUID
            contour_image_sequence.append(contour_image1) #one image per contour    

            for c in range(numberOfContoursInThisSlice):

                currentCoordinates = AllCoordinates[ROI-1][slice][c]
                
                # Contour Sequence: Contour 1
                contour = Dataset()
                contour.ContourImageSequence = contour_image_sequence
                
                cnumber+=1
                contour.ContourGeometricType = 'CLOSED_PLANAR'                
                contour.NumberOfContourPoints = len(currentCoordinates)/3
                contour.ContourData = currentCoordinates.tolist()
                contour_sequence.append(contour)
                #DEBUG print('contourData',currentCoordinates.tolist())

        roi_contour.ReferencedROINumber = ROI
        roi_contour_sequence.append(roi_contour)


    # RT ROI Observations Sequence
    rtroi_observations_sequence = Sequence()
    rtds.RTROIObservationsSequence = rtroi_observations_sequence

    # Loop over ROI observations
    for ROI in range(1,numberOfROIs+1):
        # RT ROI Observations Sequence: RT ROI Observations 1
        rtroi_observations = Dataset()
        rtroi_observations.ObservationNumber = str(ROI)
        rtroi_observations.ReferencedROINumber = str(ROI)
        rtroi_observations.RTROIInterpretedType = 'ORGAN'
        rtroi_observations.ROIObservationLabel = structure_labels[ROI-1]
        rtroi_observations.ROIInterpreter = ''
        rtroi_observations_sequence.append(rtroi_observations)

    rtds.ApprovalStatus='UNAPPROVED'
    rtds.file_meta.FileMetaInformationVersion = b'\x00\x01'
    
    RTDCM_name = output_dicom_path
    #ds.is_implicit_VR,ds.is_little_endian=True,True
    print('is_implicit_VR=',ds.is_implicit_VR,'is_little_endian=',ds.is_little_endian)
    pydicom.filewriter.dcmwrite(RTDCM_name,rtds,write_like_original=False)
    #rtds.save_as(RTDCM_name)
    print('RTSTRUCT saved as %s'%RTDCM_name)
    
def parse_structure_labels(value):
    labels = [label.strip() for label in value.split(',')]
    if not labels or any(not label for label in labels):
        raise argparse.ArgumentTypeError('structure labels must be a comma-separated list of non-empty labels')
    return labels


def parse_segmentation_intensities(value):
    if value.strip().lower() == 'all':
        return 'all'
    try:
        intensities = [int(item.strip()) for item in value.split(',')]
    except ValueError:
        raise argparse.ArgumentTypeError(
            "segmentation intensities must be 'all' or a comma-separated list of integers")
    if not intensities:
        raise argparse.ArgumentTypeError('at least one segmentation intensity must be supplied')
    return intensities


def get_parser():
    """
    Parse input arguments.
    """
    parser = argparse.ArgumentParser(description='Convert nifti images to RTSTRUCT file')

    # Positional arguments.
    parser.add_argument("input_nifti", help="Path to input NIFTI image")
    parser.add_argument("input_dicom", help="Path to input DICOM images")
    parser.add_argument("output_dicom", help="Path to output DICOM image")
    parser.add_argument("--structure_labels", metavar="<label,...>", type=parse_structure_labels,
                        default=["ROI1"], help='ordered, comma-separated structure labels [ROI1]')
    parser.add_argument("--segmentation_intensities", metavar="<all|int,...>",
                        type=parse_segmentation_intensities, default='all',
                        help="'all' for one binary mask or ordered, comma-separated label intensities [all]")
    parser.add_argument("--structure_set_label", metavar="<string>", default=None,
                        help="structure set label [the ROI label for single ROI; MULTI_ROI for multiple]")
    parser.add_argument("--series_description",metavar="<string>",type=str,default=None,help='series description for RTSTRUCT [None]')
    parser.add_argument("--series_number", metavar="<string>", type=str, default=None, help='Segmentation series number [None]')
    parser.add_argument("--tolerance",metavar="<float>", type=float, default=1,help="polygon approximation tolerance (mm) [1]")
    parser.add_argument("--min_poly_pts", metavar="<int>",type=int,default=3,help="minimum number of points in polygon [3]")

    return parser.parse_args()

if __name__ == "__main__":
    p = get_parser()
    print(p)
    convert(p.input_nifti, p.input_dicom, p.output_dicom, p.structure_labels,
            p.segmentation_intensities, p.structure_set_label, p.tolerance,
            p.min_poly_pts, p.series_number, p.series_description)
    #write_rec_file(p.output_dicom,infiles=[p.input_dicom,p.input_nifti])
