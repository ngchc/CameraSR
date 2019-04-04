import os
import numpy as np
import tensorflow as tf
from PIL import Image


def compute_psnr(im1, im2):
	if im1.shape != im2.shape:
		raise Exception('the shapes of two images are not equal')
	rmse = np.sqrt(((np.asfarray(im1) - np.asfarray(im2)) ** 2).mean())
	psnr = 20 * np.log10(255.0 / rmse)
	return psnr


class Net(object):
	def __init__(self, data, reuse=False):
		self.data = data
		self.reuse = reuse
	
	def build_net(self, summary=False):
		with tf.variable_scope('vdsr20', reuse=self.reuse):
			outputs = tf.layers.conv2d(self.data, 64, 3, padding='same', name='conv1', activation=tf.nn.relu, use_bias=True, reuse=self.reuse,
									   kernel_initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False))
			for layers in range(2, 19 + 1):
				outputs = tf.layers.conv2d(outputs, 64, 3, padding='same', name='conv%d' % layers, activation=tf.nn.relu, use_bias=True, reuse=self.reuse,
										   kernel_initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False))
			outputs = self.data + tf.layers.conv2d(outputs, 1, 3, padding='same', name='conv20', reuse=self.reuse,
								  				   kernel_initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False))
			self.sr = outputs


def main():
	test_folder = './testset'
	
	with tf.device('/gpu:0'):
		imn = tf.placeholder('float', [1, None, None, 1])
	
	# recreate the network
	net = Net(imn, reuse=False)
	with tf.device('/gpu:0'):
		net.build_net()
	output = net.sr
	
	# create a session for running operations in the graph
	config = tf.ConfigProto(allow_soft_placement=True)
	config.gpu_options.allow_growth = True
	sess = tf.Session(config=config)
	
	# restore weights
	saver = tf.train.Saver()
	saver.restore(sess, os.path.join('./model', 'model.ckpt'))

	sum_psnr = 0
	for i in [1, 16, 48, 60, 98]:
		im_hr = np.array(Image.open(os.path.join(test_folder, '%03dH.PNG' % i)))
		im_lr = np.array(Image.open(os.path.join(test_folder, '%03dL.PNG' % i)))

		im_lr = im_lr.astype(np.float32) / 255.0
		im_lr = np.expand_dims(im_lr, axis=0)
		im_lr = np.expand_dims(im_lr, axis=3)

		im_sr = sess.run(output, feed_dict={imn: im_lr})
		im_sr = np.squeeze(im_sr) * 255.0
		
		im_sr = np.maximum(im_sr, 0)
		im_sr = np.minimum(im_sr, 255)
		Image.fromarray(np.asarray(im_sr, dtype=np.uint8)).save(os.path.join('./results', '%03dR.PNG' % i))

		psnr = compute_psnr(im_hr, np.asarray(im_sr, dtype=np.uint8))
		sum_psnr += psnr
		print('%d: %.2f dB' % (i, psnr))
	print('avg: %.2f dB' % (sum_psnr / 5))


if __name__ == '__main__':
	main()
