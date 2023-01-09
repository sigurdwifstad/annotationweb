from django.shortcuts import render, redirect
from django.contrib import messages
from django.http import JsonResponse, HttpResponseRedirect
from annotationweb.models import Task, KeyFrameAnnotation, ImageSequence
import common.task
import json
from spline_segmentation.models import *
from django.db import transaction

from mdi.importers import MetaImageImporter
import numpy as np
import tensorflow as tf
import cv2
from scipy.ndimage import center_of_mass
from scipy.interpolate import splev, splprep
from mlmia.dataloader import Resize, ResizeSegmentation


def segment_next_image(request, task_id):
    return segment_image(request, task_id, None)


def segment_image(request, task_id, image_id):
    try:
        context = common.task.setup_task_context(request, task_id, Task.SPLINE_LINE_POINT, image_id)
        context['javascript_files'] = ['spline_line_point/segmentation.js']

        # Check if image is already segmented, if so get data and pass to template
        try:
            annotations = KeyFrameAnnotation.objects.filter(image_annotation__task_id=task_id, image_annotation__image_id=image_id)
            control_points = ControlPoint.objects.filter(image__in=annotations).order_by('index')
            context['control_points'] = control_points
            context['target_frames'] = annotations
        except KeyFrameAnnotation.DoesNotExist:
            pass

        return render(request, 'spline_line_point/segment_image.html', context)
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
            image_id = json.loads(request.POST["image_id"])
            n_labels = int(request.POST["n_labels"])
            control_points = json.loads(request.POST['control_points'])
            sequence_path = ImageSequence.objects.get(id=image_id).format.strip('US-2D_#.mhd')

            # Load image sequence using MetaImageImporter
            importer = MetaImageImporter(filepath=sequence_path, frame_tag='US-2D_*')
            data = (importer.data[:][..., None] / 255.0).astype(np.float32).transpose(0, 2, 1, 3)
            # Reshape to 256x256
            data_resize = np.zeros((data.shape[0], 256, 256, 1))
            resizer = Resize(256, 256)
            for i in range(data.shape[0]):
                data_resize[i] = resizer.transform(data[i])

            # Load saved_model object
            model_path = "C:/Users/sigurdvw/Models/ethiopia_pseudo_unet"
            model = tf.keras.models.load_model(model_path, compile=False)
            # Assert that number of labels matches model outputs
            assert model.output_shape[-1] == n_labels + 1  # Plus one due to background
            # Predict (and threshold at 0.0)
            pred = model.predict(data_resize)[..., 1:] > 0.0
            # Reshape back to original shape
            resizer_seg = ResizeSegmentation(data.shape[1], data.shape[2])
            pred_resize = np.zeros((*data.shape[:-1], pred.shape[-1]))
            for i in range(pred.shape[0]):
                for j in range(pred.shape[-1]):
                    pred_resize[i, ..., j] = resizer_seg.transform(pred[i, ..., j])
            # Create controlpoint object
            # TODO: get labels from segmentation.js
            labels = [{'id': 5, 'red': 255, 'green': 0, 'blue': 0, 'parent_id': 0},
                      {'id': 6, 'red': 0, 'green': 0, 'blue': 255, 'parent_id': 0},
                      {'id': 7, 'red': 0, 'green': 255, 'blue': 0, 'parent_id': 0},
                      {'id': 8, 'red': 253, 'green': 208, 'blue': 23, 'parent_id': 0}]
            pts_tables = estimate_valve_spline(pred_resize)
            for frame in range(data.shape[0]):
                if str(frame) in control_points.keys():
                    continue
                instance = {'1': dict(), '2': dict(), '3': dict(), '4': dict()}

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

def estimate_valve_spline(pred):

    npts = 7
    N_classes = pred.shape[-1]
    N = pred.shape[0]

    annulus_left = np.zeros((pred.shape[0], 1, 2))
    annulus_right = np.zeros((pred.shape[0], 1, 2))
    leaflet_left = np.zeros((pred.shape[0], npts, 2))
    leaflet_right = np.zeros((pred.shape[0], npts, 2))

    if N_classes == 5:
        pred = pred[...,1:]  # Remove background
        N_classes -= 1
    elif N_classes != 4:
        raise AssertionError('Unsupported number of classes')

    for i in range(N):
        annulus_left[i, 0, :] = np.array(center_of_mass(pred[i, ..., 2])[::-1])
        annulus_right[i, 0, :] = np.array(center_of_mass(pred[i, ..., 3])[::-1])
        leaflet_left[i, :, :] = mask2controlpoints(pred[i, ..., 0], npts)
        leaflet_right[i, :, :] = mask2controlpoints(pred[i, ..., 1], npts)

    return leaflet_left, leaflet_right, annulus_left, annulus_right

def mask2controlpoints(mask_out, npts):
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
    u_new = np.linspace(u.min(), u.max(), npts)
    x_new, y_new = splev(u_new, tck, der=0)
    c_smooth = np.concatenate((x_new[:, np.newaxis], y_new[:, np.newaxis]), axis=1)

    return c_smooth