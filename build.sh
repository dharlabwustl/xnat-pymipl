# cp Dockerfile.base Dockerfile && \
./command2label.py xnat/command.json  >> Dockerfile && \
docker build -t sharmaatul11/nifti2dicom .
##docker build -t xnat/pymip:dev .
