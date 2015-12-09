import numpy as np
import scipy.io as sio
import cv2

from os.path import join
from os import listdir

from math import pi

def generateAllChannels(directory, saving_loc):
	for image in listdir(directory):
		image_ext = image[-4:]
		if (image_ext != '.jpg' and image_ext != '.png'):
			continue
		generateChannels(directory, image, saving_loc)

def generateChannels(directory, image, saving_loc):
	channels = [] # list of array
	img = cv2.imread(join(directory, image))
	colored = True if len(img.shape) > 2 else False
	if not colored: 
		return
	img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	if colored:
		img_luv = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
	x_der = cv2.Sobel(img_gray, cv2.CV_32F, 0, 1)
	y_der = cv2.Sobel(img_gray, cv2.CV_32F, 1, 0)
	mag, angle = cv2.cartToPolar(x_der, y_der)

	no_rows = mag.shape[0]
	no_cols = mag.shape[1]

	hist = []
	for i in range(6):
		hist.append(np.zeros((no_rows, no_cols)))

	for row in range(no_rows):
		for col in range(no_cols):
			ang = angle[row, col] * 180 / pi
			if ang >= 180:
				ang -= 180
			elif ang < 0:
				ang += 180

			ind = int(ang/30)

			hist[ind][row, col] = mag[row, col]

	for i in range(3):
		channels.append(img_luv[:,:,i])
	channels.append(mag)
	channels += hist

	# saving
	image_name = image[:-3] + "mat"
	sio.savemat(join(saving_loc, image_name), mdict={'channels': np.array(channels)})
	return np.array(channels)

def generate_model(grad_img, saving_path, size=(6,6)):
	img = cv2.imread(grad_img)
	h, w = img.shape[:2]
	x_step , y_step = size[0], size[1]
	for i in range(x_step, w, x_step):
		cv2.line(img, (i,0), (i,h), (255,0,0))
	for i in range(y_step, h, y_step):
		cv2.line(img, (0,i), (w,i), (255,0,0))

	head = (3*6,1*6), (7*6,4*6)
	upper_body = (2*6,4*6), (8*6,11*6)
	lower_body = (3*6,11*6), (7*6,18*6)

	cv2.rectangle(img, head[0], head[1], (0,255,255))
	cv2.rectangle(img, upper_body[0], upper_body[1], (255,0,255))
	cv2.rectangle(img, lower_body[0], lower_body[1], (255,255,0))
	cv2.imshow('model', img)
	cv2.waitKey(0)

	#  bg, head, upper_body, lower_body = 0, 1, 2, 3
	model = np.zeros((h,w))
	model[head[0][1]:head[1][1], head[0][0]: head[1][0]] = 1
	model[upper_body[0][1]:upper_body[1][1], upper_body[0][0]: upper_body[1][0]] = 2
	model[lower_body[0][1]:lower_body[1][1], lower_body[0][0]: lower_body[1][0]] = 3

	np.save(saving_path, model)

def generate_templates(model, template_path, position_path, size=(6,6)):
	templates =  {}
	t_pos = {}
	h_m, w_m = model.shape[:2]
	min_h, max_h = 1, 4
	min_w, max_w = 2, 3
	x_step , y_step = size[0], size[1]
	for i in range(0, w_m, x_step):
		for j in range(0, h_m, y_step):
			for w in range(min_w, max_w+1):
				if (i+w*x_step) >= w_m:
					continue
				for h in range(min_h, max_h+1):
					if (j+h*y_step) >= h_m:
						continue
					x1, y1, x2, y2 = i, j, (i+w*x_step), (j+h*y_step)
					template = np.zeros((h*y_step, w*x_step))
					model_subset = model[y1:y2, x1:x2]
					unique = np.unique(model_subset)
					pos = np.array([[x1, y1], [x2,y2]])
					if len(unique) == 2:
						number_of_elements_n1 = template[model_subset == unique[0]].shape[0]
						number_of_elements_p1 = template[model_subset == unique[1]].shape[0]
						template[model_subset == unique[0]] = -1.0/number_of_elements_n1
						template[model_subset == unique[1]] = 1.0/number_of_elements_p1
						t_name = 'template_%s_%s_%s_%s'%(i,j,w,h)
						templates[t_name] = template
						t_pos[t_name] = pos

					elif len(unique) == 3:
						number_of_elements_n1 = template[model_subset == unique[1]].shape[0]
						number_of_elements_p1 = template[model_subset == unique[0]].shape[0]
						template[model_subset == unique[0]] = 1.0/number_of_elements_p1
						template[model_subset == unique[1]] = -1.0/number_of_elements_n1
						t_name = 'template_%s_%s_%s_%s_tri1'%(i,j,w,h)
						templates[t_name] = template
						t_pos[t_name] = pos

						template = np.ones((h*y_step, w*x_step))
						number_of_elements_n1 = template[model_subset == unique[2]].shape[0]
						number_of_elements_p1 = template[model_subset == unique[0]].shape[0]
						template[model_subset == unique[2]] = -1.0/number_of_elements_n1
						template[model_subset == unique[0]] = 1.0/number_of_elements_p1
						t_name = 'template_%s_%s_%s_%s_tri2'%(i,j,w,h)
						templates[t_name] = template
						t_pos[t_name] = pos

						template = np.ones((h*y_step, w*x_step))
						number_of_elements_n1 = template[model_subset == unique[2]].shape[0]
						number_of_elements_p1 = template[model_subset == unique[0]].shape[0]
						template[model_subset == unique[1]] = 1.0/number_of_elements_p1
						template[model_subset == unique[2]] = -1.0/number_of_elements_n1
						t_name = 'template_%s_%s_%s_%s_tri3'%(i,j,w,h)
						templates[t_name] = template
						t_pos[t_name] = pos

	sio.savemat(template_path, templates)
	sio.savemat(position_path, t_pos)

def compute_all_features(channels_folder, templates, t_pos, saving_loc):
	features = []
	for channels in listdir(channels_folder):
		if channels[-4:] != '.mat':
			continue
		img_channels = sio.loadmat(join(channels_folder, channels))['channels']
		features.append(compute_features(img_channels, templates, t_pos))
	np.save(saving_loc, np.asarray(features))

def compute_features(channels, templates, t_pos):
	features = []
	for ch in channels:
		for template_name, template in templates.iteritems():
			if not isinstance(template, np.ndarray):
				continue
			pos = t_pos[template_name] 
			x1, y1, x2, y2 = pos[0,0], pos[0,1], pos[1,0], pos[1,1]
			ch_subset = ch[y1:y2, x1:x2]
			feature = np.sum(ch_subset*template)
			features.append(feature)
	return features

def resize_all(directory_source, directory_dist, model_size):
	for image in listdir(directory_source):
		image_ext = image[-4:]
		if (image_ext != '.jpg' and image_ext != '.png'):
			continue
		img_path = join(directory_source, image)
		img = cv2.imread(img_path)
		img_resized = cv2.resize(img, (model_size[1], model_size[0]))
		saving_path = join(directory_dist, image)
		cv2.imwrite(saving_path, img_resized)

def main():
	model_path = '/home/retailyze/features/CF/model.npy'
	template_path = '/home/retailyze/features/CF/templates.mat'
	position_path = '/home/retailyze/features/CF/t_position.mat'
	# print 'generating model'
	# generate_model('/home/retailyze/Downloads/avg_grad1.jpg', model_path)
	model = np.load(model_path)
	# print 'generating templates'
	# generate_templates(model, template_path, position_path )
	templates = sio.loadmat(template_path)
	t_pos = sio.loadmat(position_path)


	channels_path_pos = '/home/retailyze/Downloads/INRIAPerson/body_model/channels/cropped/'
	channels_path_neg = '/home/retailyze/Downloads/INRIAPerson/body_model/channels/neg/'
	# images_path_pos = '/home/retailyze/Downloads/INRIAPerson/NormalizedWithPadding/96X160H96_train/Train/pos/'
	images_path_pos = '/home/retailyze/Downloads/INRIAPerson/Train/cropped/gaussianR1pos/'
	images_path_neg = '/home/retailyze/Downloads/INRIAPerson/extraNeg/cropped/gaussianR1neg/'
	resized_path_pos = '/home/retailyze/Downloads/INRIAPerson/body_model/resized/cropped/'
	resized_path_neg = '/home/retailyze/Downloads/INRIAPerson/body_model/resized/neg/'
	feature_path_pos = join(channels_path_pos, 'features_pos.npy')
	feature_path_neg = join(channels_path_neg, 'features_neg.npy')

	pos = True
	neg = False

	if pos:
		print 'changing images size'
		resize_all(images_path_pos, resized_path_pos, model.shape)
		# # generate all images channels and save them
		print 'generating channels'
		generateAllChannels(resized_path_pos, channels_path_pos)
		# compute features
		print 'computing features'
		compute_all_features(channels_path_pos, templates, t_pos, feature_path_pos)

	if neg:
		# print 'changing images size'
		# resize_all(images_path_neg, resized_path_neg, model.shape)
		# # generate all images channels and save them
		# print 'generating channels'
		# generateAllChannels(resized_path_neg, channels_path_neg)
		# compute features
		print 'computing features'
		compute_all_features(channels_path_neg, templates, t_pos, feature_path_neg)



if __name__ == '__main__':
	main()
