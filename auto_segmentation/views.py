import os

from django.shortcuts import render, redirect
from django.contrib import messages
from django.http import JsonResponse, HttpResponseRedirect
from annotationweb.models import Task, KeyFrameAnnotation, ImageSequence
import common.task
import json
from spline_segmentation.models import *
from django.db import transaction

from mdi.importers import MetaImageImporter  #TODO: remove mdi dependancy since it is not open source
import numpy as np
import tensorflow as tf
import cv2
from scipy.ndimage import center_of_mass
from scipy.interpolate import splev, splprep


def segment_next_image(request, task_id):
    return segment_image(request, task_id, None)


def segment_image(request, task_id, image_id):
    try:
        context = common.task.setup_task_context(request, task_id, Task.AUTO_SEGMENTATION, image_id)
        context['javascript_files'] = ['auto_segmentation/segmentation.js']

        # Check if image is already segmented, if so get data and pass to template
        try:
            annotations = KeyFrameAnnotation.objects.filter(image_annotation__task_id=task_id, image_annotation__image_id=image_id)
            control_points = ControlPoint.objects.filter(image__in=annotations).order_by('index')
            context['control_points'] = control_points
            context['target_frames'] = annotations
        except KeyFrameAnnotation.DoesNotExist:
            pass

        return render(request, 'auto_segmentation/segment_image.html', context)
    except common.task.NoMoreImages:
        messages.info(request, 'This task is finished, no more images to segment.')
        return redirect('index')
    except RuntimeError as e:
        messages.error(request, str(e))
        return HttpResponseRedirect(request.META.get('HTTP_REFERER'))


def save_segmentation(request):
    error_messages = ''
    image_id = json.loads(request.POST['image_id'])
    control_points = json.loads(request.POST['control_points'])
    target_frame_types = json.loads(request.POST['target_frame_types'])
    n_labels = int(request.POST['n_labels'])

    try:
        # Use atomic transaction here so if something crashes the annotations are restored..
        with transaction.atomic():
            annotations = common.task.save_annotation(request)

            # Save segmentation
            # Save control points
            for annotation in annotations:
                frame_nr = str(annotation.frame_nr)

                # Set frame metadata
                annotation.frame_metadata = target_frame_types[frame_nr]
                annotation.save()

                for object in control_points[frame_nr]:
                    nr_of_control_points = len(control_points[frame_nr][object]['control_points'])
                    if nr_of_control_points < 1:
                        continue
                    for point in range(nr_of_control_points):
                        control_point = ControlPoint()
                        control_point.image = annotation
                        control_point.x = float(control_points[frame_nr][object]['control_points'][point]['x'])
                        control_point.y = float(control_points[frame_nr][object]['control_points'][point]['y'])
                        control_point.index = point
                        control_point.object = int(object)
                        control_point.label = Label.objects.get(id=int(control_points[frame_nr][object]['label']['id']))
                        control_point.uncertain = bool(control_points[frame_nr][object]['control_points'][point]['uncertain'])
                        control_point.save()

            response = {
                'success': 'true',
                'message': 'Annotation saved',
            }
    except Exception as e:
        response = {
            'success': 'false',
            'message': str(e),
        }
        raise e

    return JsonResponse(response)


def show_segmentation(request, task_id, image_id):
    pass

def inference(request):
    try:
        # Use atomic transaction here so if something crashes the annotations are restored..
        with transaction.atomic():
            print(tf.config.list_physical_devices())
            image_id = int(request.POST["image_id"])
            task_id = int(request.POST["task_id"])
            n_labels = int(request.POST["n_labels"])
            labels = json.loads(request.POST["labels"])
            control_points = json.loads(request.POST['control_points'])
            sequence_path = ImageSequence.objects.get(id=image_id).format.strip('US-2D_#.mhd')

            # Load image sequence using MetaImageImporter
            importer = MetaImageImporter(filepath=sequence_path, frame_tag='US-2D_*')
            data = (importer.data[:][..., None] / 255.0).astype(np.float32).transpose(0, 2, 1, 3)
            # Reshape to 256x256
            data_resize = np.zeros((data.shape[0], 256, 256, 1))
            for i in range(data.shape[0]):
                data_resize[i, ..., 0] = cv2.resize(data[i], dsize=(256, 256))

            # Load saved_model object
            with open(Task.objects.get(id=task_id).network_config_path) as f:
                config = json.load(f)['config']
            model = tf.keras.models.load_model(config['model_path'], compile=False)
            # Assert that number of labels matches model outputs
            assert len(config['n_control_points']) == len(config['output_channels'])
            assert len(config['output_channels']) == n_labels
            assert (np.array(config['output_channels']) > 0).all()
            assert (np.array(config['n_control_points']) > 0).all()
            # Predict, choose output channels slice, threshold
            pred = model.predict(data_resize)[..., np.array(config['output_channels'])] > float(config['threshold'])
            # Reshape back to original shape
            pred_resize = np.zeros((*data.shape[:-1], pred.shape[-1]))
            for i in range(pred.shape[0]):
                for j in range(pred.shape[-1]):
                    pred_resize[i, ..., j] = cv2.resize(pred[i, ..., j].astype('uint8')*255, dsize=data[0,...,0].shape[::-1])
            # Create controlpoint object
            pts_tables = estimate_spline(pred_resize.astype('float32')/255, config['n_control_points'])
            for frame in range(data.shape[0]):
                if str(frame) in control_points.keys():
                    continue
                instance = {}
                for i in range(n_labels):
                    instance[str(i+1)] = dict()

                for j in range(pred.shape[-1]):
                    pts = pts_tables[j][frame]

                    instance[str(j + 1)]['label'] = labels[j]
                    instance[str(j + 1)]['control_points'] = []
                    if np.isnan(np.array(pts)).any():
                        instance.pop(str(j + 1))  # Remove object if control points contain nans
                    else:
                        for pt in pts:
                            instance[str(j + 1)]['control_points'].append(
                                {'x': pt[0], 'y': pt[1], 'label_id': labels[j]['id'], 'label': j, 'uncertain': 'true'})

                    # set control points
                    if instance:
                        control_points[str(frame)] = instance

            response = {
                'success': 'true',
                'message': 'Control points inferred',
                'control_points': json.dumps(dict(sorted(control_points.items(), key=lambda x: int(x[0]))))
            }
    except Exception as e:
        response = {
            'success': 'false',
            'message': str(e),
            'control_points': 'none'
        }
        raise e

    return JsonResponse(response)

def estimate_spline(pred, n_control_points):

    assert len(n_control_points) == pred.shape[-1]
    nframes = pred.shape[0]

    pts_table_list = []

    for channel, npts in enumerate(n_control_points):
        pts_table = np.zeros((nframes, npts, 2))

        for i in range(nframes):
            if npts == 1:
                pts_table[i, 0, :] = np.array(center_of_mass(pred[i, ..., channel])[::-1])
            else:
                pts_table[i, :, :] = mask2controlpoints(pred[i, ..., channel], npts)
        pts_table_list.append(pts_table)

    return pts_table_list

def mask2controlpoints(mask_out, npts):
    # TODO: fix issue where first and last points overlap

    if not mask_out.any():
        return [np.nan, np.nan]

    # Cast type
    mask_out = (mask_out/mask_out.max()*255).astype(np.uint8)

    # Find contours
    # image, contours, hierarchy = cv2.findContours(mask_out, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours, hierarchy = cv2.findContours(mask_out, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # Choose contour with largest area
    sort = sorted(contours, key=cv2.contourArea, reverse=True)
    c_out = np.squeeze(sort[0])

    # Fit spline and smooth
    dec = 1
    x, y = c_out.T
    # Decimate
    x = x[0::dec]
    y = y[0::dec]
    y[-1] = y[0]  # Avoid runtimewarning...
    x[-1] = x[0]
    # Spline rep.
    tck, u = splprep([x, y], u=None, k=3, s=0)
    # resample along spline
    u_new = np.linspace(u.min(), u.max(), npts+1)
    x_new, y_new = splev(u_new, tck, der=0)
    c_smooth = np.concatenate((x_new[:, np.newaxis], y_new[:, np.newaxis]), axis=1)
    c_smooth = c_smooth[:-1]
    return c_smooth