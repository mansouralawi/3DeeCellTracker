from __future__ import division, absolute_import, print_function, unicode_literals
from __future__ import print_function, unicode_literals, absolute_import, division

import functools
import numbers
import warnings

import numpy as np
import scipy.ndimage as ndi
from csbdeep.utils import _raise
from csbdeep.utils.tf import keras_import
from stardist.models import StarDist3D
from stardist.nms import _ind_prob_thresh

K = keras_import('backend')
Sequence = keras_import('utils', 'Sequence')
Adam = keras_import('optimizers', 'Adam')
ReduceLROnPlateau, TensorBoard = keras_import('callbacks', 'ReduceLROnPlateau', 'TensorBoard')


class StarDist3DCustom(StarDist3D):

    def _predict_instances_generator(self, img, axes=None, normalizer=None,
                                     sparse=True,
                                     prob_thresh=None, nms_thresh=None,
                                     scale=None,
                                     n_tiles=None, show_tile_progress=True,
                                     verbose=False,
                                     return_labels=True,
                                     predict_kwargs=None, nms_kwargs=None,
                                     overlap_label=None, return_predict=False):
        """Predict instance segmentation from input image.

        Parameters
        ----------
        img : :class:`numpy.ndarray`
            Input image
        axes : str or None
            Axes of the input ``img``.
            ``None`` denotes that axes of img are the same as denoted in the config.
        normalizer : :class:`csbdeep.data.Normalizer` or None
            (Optional) normalization of input image before prediction.
            Note that the default (``None``) assumes ``img`` to be already normalized.
        sparse: bool
            If true, aggregate probabilities/distances sparsely during tiled
            prediction to save memory (recommended).
        prob_thresh : float or None
            Consider only object candidates from pixels with predicted object probability
            above this threshold (also see `optimize_thresholds`).
        nms_thresh : float or None
            Perform non-maximum suppression that considers two objects to be the same
            when their area/surface overlap exceeds this threshold (also see `optimize_thresholds`).
        scale: None or float or iterable
            Scale the input image internally by this factor and rescale the output accordingly.
            All spatial axes (X,Y,Z) will be scaled if a scalar value is provided.
            Alternatively, multiple scale values (compatible with input `axes`) can be used
            for more fine-grained control (scale values for non-spatial axes must be 1).
        n_tiles : iterable or None
            Out of memory (OOM) errors can occur if the input image is too large.
            To avoid this problem, the input image is broken up into (overlapping) tiles
            that are processed independently and re-assembled.
            This parameter denotes a tuple of the number of tiles for every image axis (see ``axes``).
            ``None`` denotes that no tiling should be used.
        show_tile_progress: bool
            Whether to show progress during tiled prediction.
        verbose: bool
            Whether to print some info messages.
        return_labels: bool
            Whether to create a label image, otherwise return None in its place.
        predict_kwargs: dict
            Keyword arguments for ``predict`` function of Keras model.
        nms_kwargs: dict
            Keyword arguments for non-maximum suppression.
        overlap_label: scalar or None
            if not None, label the regions where polygons overlap with that value
        return_predict: bool
            Also return the outputs of :func:`predict` (in a separate tuple)
            If True, implies sparse = False

        Returns
        -------
        (:class:`numpy.ndarray`, dict), (optional: return tuple of :func:`predict`)
            Returns a tuple of the label instances image and also
            a dictionary with the details (coordinates, etc.) of all remaining polygons/polyhedra.

        """
        if predict_kwargs is None:
            predict_kwargs = {}
        if nms_kwargs is None:
            nms_kwargs = {}

        if return_predict and sparse:
            sparse = False
            warnings.warn("Setting sparse to False because return_predict is True")

        nms_kwargs.setdefault("verbose", verbose)

        _axes         = self._normalize_axes(img, axes)
        _axes_net     = self.config.axes
        _permute_axes = self._make_permute_axes(_axes, _axes_net)
        _shape_inst   = tuple(s for s,a in zip(_permute_axes(img).shape, _axes_net) if a != 'C')

        if scale is not None:
            if isinstance(scale, numbers.Number):
                scale = tuple(scale if a in 'XYZ' else 1 for a in _axes)
            scale = tuple(scale)
            len(scale) == len(_axes) or _raise(ValueError(f"scale {scale} must be of length {len(_axes)}, i.e. one value for each of the axes {_axes}"))
            for s,a in zip(scale,_axes):
                s > 0 or _raise(ValueError("scale values must be greater than 0"))
                (s in (1,None) or a in 'XYZ') or warnings.warn(f"replacing scale value {s} for non-spatial axis {a} with 1")
            scale = tuple(s if a in 'XYZ' else 1 for s,a in zip(scale,_axes))
            verbose and print(f"scaling image by factors {scale} for axes {_axes}")
            img = ndi.zoom(img, scale, order=1)

        yield 'predict'  # indicate that prediction is starting
        res = None
        if sparse:
            for res in self._predict_sparse_generator(img, axes=axes, normalizer=normalizer, n_tiles=n_tiles,
                                                      prob_thresh=prob_thresh, show_tile_progress=show_tile_progress, **predict_kwargs):
                if res is None:
                    yield 'tile'  # yield 'tile' each time a tile has been processed
        else:
            for res in self._predict_generator(img, axes=axes, normalizer=normalizer, n_tiles=n_tiles,
                                               show_tile_progress=show_tile_progress, **predict_kwargs):
                if res is None:
                    yield 'tile'  # yield 'tile' each time a tile has been processed
            res = tuple(res) + (None,)

        if self._is_multiclass():
            prob, dist, prob_class, points, prob_map = res
        else:
            prob, dist, points, prob_map = res
            prob_class = None

        yield 'nms'  # indicate that non-maximum suppression is starting
        res_instances = self._instances_from_prediction(_shape_inst, prob, dist,
                                                        points=points,
                                                        prob_class=prob_class,
                                                        prob_thresh=prob_thresh,
                                                        nms_thresh=nms_thresh,
                                                        scale=(None if scale is None else dict(zip(_axes,scale))),
                                                        return_labels=return_labels,
                                                        overlap_label=overlap_label,
                                                        **nms_kwargs)

        # last "yield" is the actual output that would have been "return"ed if this was a regular function
        if return_predict:
            yield res_instances, tuple(res[:-1]), prob_map
        else:
            yield res_instances, prob_map

    @functools.wraps(_predict_instances_generator)
    def predict_instances(self, *args, **kwargs):
        # the reason why the actual computation happens as a generator function
        # (in '_predict_instances_generator') is that the generator is called
        # from the stardist napari plugin, which has its benefits regarding
        # control flow and progress display. however, typical use cases should
        # almost always use this function ('predict_instances'), and shouldn't
        # even notice (thanks to @functools.wraps) that it wraps the generator
        # function. note that similar reasoning applies to 'predict' and
        # 'predict_sparse'.

        # return last "yield"ed value of generator
        r = None
        for r in self._predict_instances_generator(*args, **kwargs):
            pass
        return r

    def _predict_sparse_generator(self, img, prob_thresh=None, axes=None, normalizer=None, n_tiles=None, show_tile_progress=True, b=2, **predict_kwargs):
        """ Sparse version of model.predict()
        Returns
        -------
        (prob, dist, [prob_class], points)   flat list of probs, dists, (optional prob_class) and points
        """
        if prob_thresh is None: prob_thresh = self.thresholds.prob

        x, axes, axes_net, axes_net_div_by, _permute_axes, resizer, n_tiles, grid, grid_dict, channel, predict_direct, tiling_setup = \
            self._predict_setup(img, axes, normalizer, n_tiles, show_tile_progress, predict_kwargs)

        def _prep(prob, dist):
            prob = np.take(prob,0,axis=channel)
            dist = np.moveaxis(dist,channel,-1)
            dist = np.maximum(1e-3, dist)
            return prob, dist

        proba, dista, pointsa, prob_class = [],[],[], []

        if np.prod(n_tiles) > 1:
            raise NotImplementedError("The prediction function has not been realized when np.prod(n_tiles)>1")
            tile_generator, output_shape, create_empty_output = tiling_setup()

            sh = list(output_shape)
            sh[channel] = 1;

            proba, dista, pointsa, prob_classa = [], [], [], []

            for tile, s_src, s_dst in tile_generator:

                results_tile = predict_direct(tile)

                # account for grid
                s_src = [slice(s.start//grid_dict.get(a,1),s.stop//grid_dict.get(a,1)) for s,a in zip(s_src,axes_net)]
                s_dst = [slice(s.start//grid_dict.get(a,1),s.stop//grid_dict.get(a,1)) for s,a in zip(s_dst,axes_net)]
                s_src[channel] = slice(None)
                s_dst[channel] = slice(None)
                s_src, s_dst = tuple(s_src), tuple(s_dst)

                prob_tile, dist_tile = results_tile[:2]
                prob_tile, dist_tile = _prep(prob_tile[s_src], dist_tile[s_src])

                bs = list((b if s.start==0 else -1, b if s.stop==_sh else -1) for s,_sh in zip(s_dst, sh))
                bs.pop(channel)
                inds   = _ind_prob_thresh(prob_tile, prob_thresh, b=bs)
                proba.extend(prob_tile[inds].copy())
                dista.extend(dist_tile[inds].copy())
                _points = np.stack(np.where(inds), axis=1)
                offset = list(s.start for i,s in enumerate(s_dst))
                offset.pop(channel)
                _points = _points + np.array(offset).reshape((1,len(offset)))
                _points = _points * np.array(self.config.grid).reshape((1,len(self.config.grid)))
                pointsa.extend(_points)

                if self._is_multiclass():
                    p = results_tile[2][s_src].copy()
                    p = np.moveaxis(p,channel,-1)
                    prob_classa.extend(p[inds])
                yield  # yield None after each processed tile

        else:
            # predict_direct -> prob, dist, [prob_class if multi_class]
            results = predict_direct(x)
            prob, dist = results[:2]
            prob, dist = _prep(prob, dist)
            inds   = _ind_prob_thresh(prob, prob_thresh, b=b)
            proba = prob[inds].copy()
            dista = dist[inds].copy()
            _points = np.stack(np.where(inds), axis=1)
            pointsa = (_points * np.array(self.config.grid).reshape((1,len(self.config.grid))))

            if self._is_multiclass():
                p = np.moveaxis(results[2],channel,-1)
                prob_classa = p[inds].copy()

        proba = np.asarray(proba)
        dista = np.asarray(dista).reshape((-1,self.config.n_rays))
        pointsa = np.asarray(pointsa).reshape((-1,self.config.n_dim))

        prob_map = resizer.after(prob[:, :, :, None], self.config.axes)[..., 0]

        idx = resizer.filter_points(x.ndim, pointsa, axes_net)
        proba = proba[idx]
        dista = dista[idx]
        pointsa = pointsa[idx]

        # last "yield" is the actual output that would have been "return"ed if this was a regular function
        if self._is_multiclass():
            prob_classa = np.asarray(prob_classa).reshape((-1,self.config.n_classes+1))
            prob_classa = prob_classa[idx]
            yield proba, dista, prob_classa, pointsa, prob_map
        else:
            prob_classa = None
            yield proba, dista, pointsa, prob_map

    def predict_instances_big(self, img, axes, block_size, min_overlap, context=None,
                              labels_out=None, labels_out_dtype=np.int32, show_progress=True, **kwargs):
        """Predict instance segmentation from very large input images.

        Intended to be used when `predict_instances` cannot be used due to memory limitations.
        This function will break the input image into blocks and process them individually
        via `predict_instances` and assemble all the partial results. If used as intended, the result
        should be the same as if `predict_instances` was used directly on the whole image.

        **Important**: The crucial assumption is that all predicted object instances are smaller than
                       the provided `min_overlap`. Also, it must hold that: min_overlap + 2*context < block_size.

        Example
        -------
        >>> img.shape
        (20000, 20000)
        >>> labels, polys = model.predict_instances_big(img, axes='YX', block_size=4096,
                                                        min_overlap=128, context=128, n_tiles=(4,4))

        Parameters
        ----------
        img: :class:`numpy.ndarray` or similar
            Input image
        axes: str
            Axes of the input ``img`` (such as 'YX', 'ZYX', 'YXC', etc.)
        block_size: int or iterable of int
            Process input image in blocks of the provided shape.
            (If a scalar value is given, it is used for all spatial image dimensions.)
        min_overlap: int or iterable of int
            Amount of guaranteed overlap between blocks.
            (If a scalar value is given, it is used for all spatial image dimensions.)
        context: int or iterable of int, or None
            Amount of image context on all sides of a block, which is discarded.
            If None, uses an automatic estimate that should work in many cases.
            (If a scalar value is given, it is used for all spatial image dimensions.)
        labels_out: :class:`numpy.ndarray` or similar, or None, or False
            numpy array or similar (must be of correct shape) to which the label image is written.
            If None, will allocate a numpy array of the correct shape and data type ``labels_out_dtype``.
            If False, will not write the label image (useful if only the dictionary is needed).
        labels_out_dtype: str or dtype
            Data type of returned label image if ``labels_out=None`` (has no effect otherwise).
        show_progress: bool
            Show progress bar for block processing.
        kwargs: dict
            Keyword arguments for ``predict_instances``.

        Returns
        -------
        (:class:`numpy.ndarray` or False, dict)
            Returns the label image and a dictionary with the details (coordinates, etc.) of the polygons/polyhedra.

        """
        from stardist.stardist.big import _grid_divisible, BlockND, OBJECT_KEYS#, repaint_labels
        from stardist.stardist.matching import relabel_sequential
        from csbdeep.utils import _raise, backend_channels_last, axes_check_and_normalize, axes_dict, load_json, save_json
        from tqdm import tqdm


        n = img.ndim
        axes = axes_check_and_normalize(axes, length=n)
        grid = self._axes_div_by(axes)
        axes_out = self._axes_out.replace('C','')
        shape_dict = dict(zip(axes,img.shape))
        shape_out = tuple(shape_dict[a] for a in axes_out)

        if context is None:
            context = self._axes_tile_overlap(axes)

        if np.isscalar(block_size):  block_size  = n*[block_size]
        if np.isscalar(min_overlap): min_overlap = n*[min_overlap]
        if np.isscalar(context):     context     = n*[context]
        block_size, min_overlap, context = list(block_size), list(min_overlap), list(context)
        assert n == len(block_size) == len(min_overlap) == len(context)

        if 'C' in axes:
            # single block for channel axis
            i = axes_dict(axes)['C']
            block_size[i] = img.shape[i]
            min_overlap[i] = context[i] = 0

        block_size  = tuple(_grid_divisible(g, v, name='block_size',  verbose=False) for v,g,a in zip(block_size, grid,axes))
        min_overlap = tuple(_grid_divisible(g, v, name='min_overlap', verbose=False) for v,g,a in zip(min_overlap,grid,axes))
        context     = tuple(_grid_divisible(g, v, name='context',     verbose=False) for v,g,a in zip(context,    grid,axes))

        # print(f"input: shape {img.shape} with axes {axes}")
        print(f'effective: block_size={block_size}, min_overlap={min_overlap}, context={context}', flush=True)

        for a,c,o in zip(axes,context,self._axes_tile_overlap(axes)):
            if c < o:
                print(f"{a}: context of {c} is small, recommended to use at least {o}", flush=True)

        # create block cover
        blocks = BlockND.cover(img.shape, axes, block_size, min_overlap, context, grid)

        if np.isscalar(labels_out) and bool(labels_out) is False:
            labels_out = None
        else:
            if labels_out is None:
                labels_out = np.zeros(shape_out, dtype=labels_out_dtype)
            else:
                labels_out.shape == shape_out or _raise(ValueError(f"'labels_out' must have shape {shape_out} (axes {axes_out})."))

        # Initialize probability map with half the shape on the first dimension
        grid_reciprocal = tuple(1 / g for g in self.config.grid) 
        prob_map_out = np.zeros(
        tuple(int(img.shape[i] * grid_reciprocal[i]) for i in range(3)),  
        dtype=np.float32)

        polys_all = {}
        # problem_ids = []
        label_offset = 1

        kwargs_override = dict(axes=axes, overlap_label=None, return_labels=True, return_predict=False)
        if show_progress:
            kwargs_override['show_tile_progress'] = False # disable progress for predict_instances
        for k,v in kwargs_override.items():
            if k in kwargs: print(f"changing '{k}' from {kwargs[k]} to {v}", flush=True)
            kwargs[k] = v
        blocks = tqdm(blocks, disable=(not show_progress))
        # actual computation
        for block in blocks:
            (labels, polys), details  = self.predict_instances(block.read(img, axes=axes), **kwargs)
            labels = block.crop_context(labels, axes=axes_out)
            labels, polys = block.filter_objects(labels, polys, axes=axes_out)
            # TODO: relabel_sequential is not very memory-efficient (will allocate memory proportional to label_offset)
            # this should not change the order of labels
            labels = relabel_sequential(labels, label_offset)[0]

            prob_block_cropped = block.crop_context(details, axes=axes_out,scale_factors=tuple(1 / x for x in self.config.grid))
            block.write(prob_map_out, prob_block_cropped, axes=axes_out,scale_factors=tuple(1 / x for x in self.config.grid))


            if labels_out is not None:
                block.write(labels_out, labels, axes=axes_out)

            for k,v in polys.items():
                polys_all.setdefault(k,[]).append(v)

            label_offset += len(polys['prob'])
            del labels

        polys_all = {k: (np.concatenate(v) if k in OBJECT_KEYS else v[0]) for k,v in polys_all.items()}

        # if labels_out is not None and len(problem_ids) > 0:
        #     # if show_progress:
        #     #     blocks.write('')
        #     # print(f"Found {len(problem_ids)} objects that violate the 'min_overlap' assumption.", file=sys.stderr, flush=True)
        #     repaint_labels(labels_out, problem_ids, polys_all, show_progress=False)

        return labels_out, polys_all, prob_map_out#, tuple(problem_ids)