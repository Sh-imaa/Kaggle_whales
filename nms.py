# improved on http://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/

# non-maximum supression

# import the necessary packages
import numpy as np
 
# Malisiewicz et al.
def non_max_suppression_fast(boxes, overlapThresh, by_pred=True, max_bd=True):
	# if there are no boxes, return an empty list
	if len(boxes) == 0:
		return []
 
	# if the bounding boxes integers, convert them to floats --
	# this is important since we'll be doing a bunch of divisions
	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")
 
	# initialize the list of picked indexes	
	pick = []
 
	# grab the coordinates of the bounding boxes
	x1 = boxes[:,0]
	y1 = boxes[:,1]
	x2 = boxes[:,2]
	y2 = boxes[:,3]

	# compute the area of the bounding boxes and sort the bounding
	# boxes by the bottom-right y-coordinate of the bounding box
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = np.argsort(y2)

	# keep looping while some indexes still remain in the indexes
	# list
	while len(idxs) > 0:
		# grab the last index in the indexes list and add the
		# index value to the list of picked indexes
		i = idxs[-1]
 
		# find the largest (x, y) coordinates for the start of
		# the bounding box and the smallest (x, y) coordinates
		# for the end of the bounding box
		xx1 = np.maximum(x1[i], x1[idxs[:-1]])
		yy1 = np.maximum(y1[i], y1[idxs[:-1]])
		xx2 = np.minimum(x2[i], x2[idxs[:-1]])
		yy2 = np.minimum(y2[i], y2[idxs[:-1]])
 
		# compute the width and height of the bounding box
		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)
 
		# compute the ratio of overlap
		overlap = (w * h) / area[idxs[:-1]]
 
		# delete all indexes from the index list that have overlap > overlapThresh
		to_be_deleted_idxs = np.concatenate(([len(idxs)-1], np.where(overlap > overlapThresh)[0]))
		if by_pred:
			to_be_deleted = boxes[idxs[to_be_deleted_idxs]]
			pred_sort_idxs = np.argsort(to_be_deleted[:, 4])
			best_pred_idx = idxs[to_be_deleted_idxs[pred_sort_idxs[-1]]]
			pick.append(best_pred_idx)
		elif not max_bd:
			avg_x1 = int(np.average(boxes[idxs[to_be_deleted_idxs]][:,0]))
			avg_y1 = int(np.average(boxes[idxs[to_be_deleted_idxs]][:,1]))
			avg_x2 = int(np.average(boxes[idxs[to_be_deleted_idxs]][:,2]))
			avg_y2 = int(np.average(boxes[idxs[to_be_deleted_idxs]][:,3]))
			pick.append((avg_x1, avg_y1, avg_x2, avg_y2))
		else:
			min_x1 = int(np.min(boxes[idxs[to_be_deleted_idxs]][:,0]))
			min_y1 = int(np.min(boxes[idxs[to_be_deleted_idxs]][:,1]))
			max_x2 = int(np.max(boxes[idxs[to_be_deleted_idxs]][:,2]))
			max_y2 = int(np.max(boxes[idxs[to_be_deleted_idxs]][:,3]))
			pick.append((min_x1, min_y1, max_x2, max_y2))

		idxs = np.delete(idxs, to_be_deleted_idxs)
 
	# return only the bounding boxes that were picked using the
	# integer data type
	if by_pred:
		return boxes[pick].astype("int")
	else:
		return pick


