# test scene with background sub
import cv2
import scipy.io as sio
from sklearn.externals import joblib
from skimage.transform import pyramid_gaussian, pyramid_expand, pyramid_reduce
from generate_features import *
from math import pi

def generateIntegralChannels(image):
	channels = [] # list of array
	img = image
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
			ang1 = ang
			if ang >= 180 and ang < 360:
				ang -= 180
			elif ang < 0:
				ang += 180
			elif ang >=  360:
				ang = 0

			ind = int(ang/30)


			try:
				hist[ind][row, col] = mag[row, col]
			except:
				print mag.shape
				print (np.asarray(hist)).shape
				print row, col, ind, ang, ang1

	for i in range(3):
		channels.append(cv2.integral(img_luv[:,:,i]))
	channels.append(cv2.integral(mag))
	for i in range(len(hist)):
		hist[i] = cv2.integral(hist[i])
	channels += hist
	return channels

def predict_per_scene(img):

	# /home/retailyze/Downloads/INRIAPerson/Test/pos/crop001607.png
	# img = cv2.imread("/home/retailyze/CV/Videos/Passers/test_frame2.png")

	h, w, ch = img.shape

	img = cv2.GaussianBlur(img,(1,1),0)

	pyramids = 10
	step_size_h = 20
	step_size_w = 20
	window_size = (160,96)
	model = joblib.load("/home/retailyze/Downloads/INRIAPerson/svm/extra.pkl")

	# read features
	feature_dir = "/home/retailyze/Downloads/INRIAPerson/features/"
	icf_features_no = sio.loadmat(join(feature_dir, "features_icf.mat"))["icf"]
	

	images = [img]
	# upscale
	# upscaled = img
	# for i in range(1):
	# 	# upscaled = pyramid_expand(upscaled, upscale=1.1)
	# 	upscaled = cv2.pyrUp(upscaled)
	# 	images.append(upscaled)
	# downscale
	downscaled = img
	for i in range(pyramids):
		# downscaled = pyramid_reduce(downscaled, downscale=1.1)
		# downscaled = cv2.pyrDown(downscaled)
		# images.append(downscaled)
		new_size =  int(downscaled.shape[1]/1.2), int(downscaled.shape[0]/1.2)
		downscaled = cv2.resize(downscaled, new_size)
		images.append(downscaled)

	for (i,image) in enumerate(images[3:]):
		print "new pyramid-%s"%i
		# image = image.astype(np.float32)
		channels = generateIntegralChannels(image)
		print "integral channels generated"
			# for x in range(0, w-window_size[1], step_size_w):
			# for y in range(0, h-window_size[0], step_size_h):
		for x in range(0, w-window_size[1], step_size_w):
			for y in range(0, h/5, step_size_h):
				try:
					test_existance = image[y+window_size[0], x+window_size[1]]
				except:
					continue
				window_channels = np.asarray(channels)[:, y:y+window_size[0], x:x+window_size[1]]
				features_icf = evaluate_all_features(window_channels, "icf", icf_features_no)
				prediction = model.predict(features_icf)
				dist = model.decision_function(features_icf)
				if dist > 0:
					print "found human candidate", dist
					# cv2.rectangle(image, (x,y), (x+window_size[1],y+window_size[0]), (0,255,0))
					center = (x+window_size[1]/2, y+window_size[0]/2)
					cv2.circle(image, center, 3, (0,225,0))
				else:
					print dist
					# cv2.rectangle(image, (x,y), (x+window_size[1],y+window_size[0]), (255,0,0))
				cv2.imshow("Scene%s"%i, image)
				cv2.waitKey(1000)

def main():
	video_path = '/home/retailyze/CV/annotation/videos/17/20Min.avi'
	# video_path = "/home/retailyze/CV/Videos/Passers/Tue Aug 25 20_32_46 2015out.16.avi"
 	cap = cv2.VideoCapture(video_path)
 	print "FPS: %s"%cap.get(cv2.CAP_PROP_FPS)
 	fgbg = cv2.createBackgroundSubtractorMOG2(1000,100, True)
 	count = 0
 	starting_thread = 10 #123 # 282 #237 #1586
 	window_size = (160,96)

 	model = joblib.load("/home/retailyze/Downloads/INRIAPerson/NormalizedWithPadding/svm/not_cropped_svm.pkl")

	# read features
	feature_dir = "/home/retailyze/Downloads/INRIAPerson/features/"
	icf_features_no = sio.loadmat(join(feature_dir, "icf_96_160.mat"))["icf"]
	while(True):
		boundingBoxes = []
		ret, frame = cap.read()
		frame = cv2.GaussianBlur(frame,(21,21),0)
		frame = cv2.resize(frame, (int(frame.shape[1]/1.2), int(frame.shape[0]/1.2)))
		frame = cv2.resize(frame, (int(frame.shape[1]/1.2), int(frame.shape[0]/1.2)))
		frame = cv2.resize(frame, (int(frame.shape[1]/1.2), int(frame.shape[0]/1.2)))
		frame = cv2.resize(frame, (int(frame.shape[1]/1.2), int(frame.shape[0]/1.2)))
		frame = cv2.resize(frame, (int(frame.shape[1]/1.2), int(frame.shape[0]/1.2)))
		frame = cv2.resize(frame, (int(frame.shape[1]/1.2), int(frame.shape[0]/1.2)))
		frame = np.lib.pad(frame, ((80,80), (80,80), (0,0)), mode='median')

		w, h = (frame.shape[0], frame.shape[1])
		fgmask = fgbg.apply(frame)
		count += 1
		ret, fgmask = cv2.threshold(fgmask, 127, 255, cv2.THRESH_BINARY)
		kernel = np.ones((20,20), np.uint8)
		if(count<starting_thread):
			continue
		if (count%10 != 0):
			continue
		for i in range(20):
			fgmask = cv2.dilate(fgmask, kernel, iterations=1)
			fgmask = cv2.erode(fgmask, kernel, iterations=1)
		im2, contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
		for contour in contours[1:]:
			xc,yc,wc,hc = cv2.boundingRect(contour)
			area = wc*hc
			if(area<int(0.005*w*h) or area>int(0.4*w*h)):
				continue
			xc,yc,wc,hc = (int(xc*0.8), int(yc*0.45), int(wc*1.6), int(hc*1.6))
			if(wc < window_size[1] or hc < window_size[0]):
				continue
			cv2.rectangle(frame, (xc,yc), (xc+wc,yc+hc), (255,255,0))
			cv2.putText(frame, str(area), (xc,yc), cv2.FONT_HERSHEY_SIMPLEX, .7, (255,0,255))
			img = frame[ yc:yc+hc, xc:xc+wc]
			channels = generateIntegralChannels(img)
			for i in range(xc, xc+wc-window_size[1], 10):
				for j in range(yc, yc+hc-window_size[0], 10):
					if((xc+wc - i) < window_size[1] or (yc+hc - j)< window_size[0]):
						continue
					try:
						test_existance = frame[j+window_size[0],i+window_size[1]]
					except:
						continue
					window_channels = np.asarray(channels)[:, j-yc:j-yc+window_size[0], i-xc:i-xc+window_size[1]]
					features_icf = evaluate_all_features(window_channels, "icf", icf_features_no)
					prediction = model.predict(features_icf)
					dist = model.decision_function(features_icf)
					if dist > 0:
						print "found human candidate", dist
						cv2.rectangle(frame, (i,j), (i+window_size[1], j+window_size[0]), (0,255,0))
						center = (i+window_size[1]/2, j+window_size[0]/2)
						cv2.circle(frame, center, 3, (0,225,0))
					else:
						print dist
						cv2.rectangle(frame, (i,j), (i+window_size[1], j+window_size[0]), (255,0,0))
					cv2.imshow("frame", frame)
					cv2.waitKey(100)

			
		# print count
		# cv2.imshow("frame", frame)
		# cv2.imwrite("/home/retailyze/CV/Videos/Passers/test_frame2.png", frame)
		# return
		# cv2.waitKey(0)

	cap.release()
	cv2.destroyAllWindows()

if __name__ == "__main__":
	main()