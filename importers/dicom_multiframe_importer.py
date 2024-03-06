from common.importer import Importer
from django import forms
from annotationweb.models import ImageSequence, Dataset, Subject, ImageMetadata
import os
from os.path import join, basename
import glob
from highdicom.io import ImageFileReader

class DicomMultiframeImporterForm(forms.Form):
    path = forms.CharField(label='Data path', max_length=1000)

    # TODO validate path

    def __init__(self, data=None):
        super().__init__(data)


class DicomMultiframeImporter(Importer):
    """
    Data should be sorted in the following way in the root folder:
    Subject 1/
        Sequence 1
        Sequence 2
    Subject 2/
        ...

    This importer will create a subject for each subject folder and an image sequence for each subfolder.
    """

    name = 'Dicom multiframe importer'
    dataset = None

    def get_form(self, data=None):
        return DicomMultiframeImporterForm(data)

    def import_data(self, form):
        if self.dataset is None:
            raise Exception('Dataset must be given to importer')

        path = form.cleaned_data['path']
        # Go through each subfolder and create a subject for each
        for file in os.listdir(path):
            subject_dir = join(path, file)
            #if not os.path.isdir(subject_dir):
            #    continue

            try:
                # Check if subject exists in this dataset first
                subject = Subject.objects.get(name=file, dataset=self.dataset)
            except Subject.DoesNotExist:
                # Create new subject
                subject = Subject()
                subject.name = file
                subject.dataset = self.dataset
                subject.save()

            for file2 in os.listdir(subject_dir):
                image_sequence_path = join(subject_dir, file2)

                filename_format = join(image_sequence_path, 'frame_#')
                try:
                    # Check to see if sequence exist
                    image_sequence = ImageSequence.objects.get(format=filename_format, subject=subject)
                except ImageSequence.DoesNotExist:
                    # Create new image sequence
                    image_sequence = ImageSequence()
                    image_sequence.format = filename_format
                    image_sequence.subject = subject
                    with ImageFileReader(image_sequence_path) as image:
                        image_sequence.nr_of_frames = image.number_of_frames
                    image_sequence.save()


        return True, path
