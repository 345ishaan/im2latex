import numpy as np
import os
from PIL import Image
import tensorflow as tf
import h5py
import cnn
from ops import Embedding,Attention_Enc,Attention_Dec
from data_gen import *
import matplotlib.pyplot as plt




class Model(object):

	def __init__(self,phase,vocab_list_path=None,data_base_dir=None,data_path=None,formula_map=None,test_data_map=None,train_data_map=None,validate_data_map=None,cnn_pretrain_path=None,batch_size=20,image=None):

		self.map_v={}
		self.map_i={}

		self.map_v['GO'] = 1
		self.map_v['EOS'] = 2
		self.map_v['PAD'] = 0
		self.map_v['UNKNOWN'] = 3

		self.map_i[0] = 'PAD'
		self.map_i[1] = 'GO'
		self.map_i[2] = 'EOS'
		self.map_i[3] = 'UNKNOWN'

		self.embedding_dim = 80
		self.target_vocab_size = 502
		self.ENC_MAX_WIDTH = 50
		self.ENC_MAX_HEIGHT = 20
		self.ENC_OP_DIM = 512
		self.batch_size = batch_size
		self.ENC_DIM = 256
		self.DEC_DIM = 512
		self.MAX_CT_VEC_LENGTH = self.ENC_MAX_HEIGHT*self.ENC_MAX_WIDTH

		self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
		self.image = image

		self.img_ip = tf.placeholder(shape=(None,None,None,1),dtype=tf.float32,name='data')
		self.conv_op = tf.placeholder(shape=(None,None,None,512),dtype=tf.float32,name='convop')
		self.decoder_input = tf.placeholder(shape=(None,None),dtype=tf.int32,name='seq')

		self.embedding_seqs = Embedding('embedding',self.target_vocab_size,self.embedding_dim,self.decoder_input)

		self.conv_op = cnn.CNN_Net(self.img_ip)
		print "Initialised CNN"
		self.encoder_output = Attention_Enc('Attention_Enc',self.conv_op,self.ENC_DIM,self.ENC_MAX_WIDTH,self.ENC_MAX_HEIGHT,self.ENC_OP_DIM,self.batch_size)
		print "Initialised Encoder"
		self.decoder_output ,self.logits= Attention_Dec('Attention_Dec',self.encoder_output,self.embedding_seqs,self.DEC_DIM,self.MAX_CT_VEC_LENGTH,self.ENC_OP_DIM,self.batch_size,self.embedding_dim,self.target_vocab_size)
		print "Initialised Decoder"

		self.w_fp = h5py.File(cnn_pretrain_path)
		cnn_weight_dict={}
		cnn_weight_dict['conv1/w:0'] = self.w_fp['conv1/conv1.W:0'][...]
		cnn_weight_dict['conv2/w:0'] = self.w_fp['conv2/conv2.W:0'][...]
		cnn_weight_dict['conv3/w:0'] = self.w_fp['conv3/conv3.W:0'][...]
		cnn_weight_dict['BatchNorm/beta:0'] = self.w_fp['conv3//beta:0'][...]
		cnn_weight_dict['BatchNorm/moving_mean:0'] = self.w_fp['conv3//moving_mean:0'][...]
		cnn_weight_dict['BatchNorm/moving_variance:0'] = self.w_fp['conv3//moving_variance:0'][...]

		cnn_weight_dict['conv4/w:0'] = self.w_fp['conv4/conv4.W:0'][...]
		cnn_weight_dict['conv5/w:0'] = self.w_fp['conv5/conv5.W:0'][...]
		cnn_weight_dict['BatchNorm_1/beta:0'] = self.w_fp['conv5//beta:0'][...]
		cnn_weight_dict['BatchNorm_1/moving_mean:0'] = self.w_fp['conv5//moving_mean:0'][...]
		cnn_weight_dict['BatchNorm_1/moving_variance:0'] = self.w_fp['conv5//moving_variance:0'][...]

		cnn_weight_dict['conv6/w:0'] = self.w_fp['conv6/conv6.W:0'][...]
		cnn_weight_dict['BatchNorm_2/beta:0'] = self.w_fp['conv6//beta:0'][...]
		cnn_weight_dict['BatchNorm_2/moving_mean:0'] = self.w_fp['conv6//moving_mean:0'][...]
		cnn_weight_dict['BatchNorm_2/moving_variance:0'] = self.w_fp['conv6//moving_variance:0'][...]

		cnn_weight_dict['embedding_1:0']=self.w_fp['Embedding/Embedding:0'][...]
		cnn_weight_dict['Attention_Enchidden_fw_1:0'] = self.w_fp['AttLSTM.Enc_.init.h0_1:0'][...]
		cnn_weight_dict['Attention_Enchidden_bw_1:0'] = self.w_fp['AttLSTM.Enc_init.h0_2:0'][...]
		cnn_weight_dict['BiRNN/FW/Attention_Enc.BiLSTMEncoder_fw.Gates.W:0'] = self.w_fp['scan/while/BiRNN/FW/FW/while/AttLSTM.BiLSTMEncoder_fw.Gates/AttLSTM.BiLSTMEncoder_fw.Gates.W:0'][...]
		cnn_weight_dict['BiRNN/FW/Attention_Enc.BiLSTMEncoder_fw.Gates.b:0'] = self.w_fp['scan/while/BiRNN/FW/FW/while/AttLSTM.BiLSTMEncoder_fw.Gates/AttLSTM.BiLSTMEncoder_fw.Gates.b:0'][...]

		cnn_weight_dict['BiRNN/BW/Attention_Enc.BiLSTMEncoder_bw.Gates.W:0'] = self.w_fp['scan/while/BiRNN/BW/BW/while/AttLSTM.BiLSTMEncoder_bw.Gates/AttLSTM.BiLSTMEncoder_bw.Gates.W:0'][...]
		cnn_weight_dict['BiRNN/BW/Attention_Enc.BiLSTMEncoder_bw.Gates.b:0'] = self.w_fp['scan/while/BiRNN/BW/BW/while/AttLSTM.BiLSTMEncoder_bw.Gates/AttLSTM.BiLSTMEncoder_bw.Gates.b:0'][...]
		cnn_weight_dict['Attention_Dechidden_dec_1:0'] = self.w_fp['AttLSTM.Decoder.init.h0:0'][...]
		cnn_weight_dict['RNN/Attention_Dec.AttentionCell.Gates.W:0'] = self.w_fp['RNN/while/AttLSTM.AttentionCell.Gates/AttLSTM.AttentionCell.Gates.W:0'][...]
		cnn_weight_dict['RNN/Attention_Dec.AttentionCell.Gates.b:0'] = self.w_fp['RNN/while/AttLSTM.AttentionCell.Gates/AttLSTM.AttentionCell.Gates.b:0'][...]

		cnn_weight_dict['RNN/Attention_Dec.AttentionCell.target_t.W:0'] = self.w_fp['RNN/while/AttLSTM.AttentionCell.target_t/AttLSTM.AttentionCell.target_t.W:0'][...]
		cnn_weight_dict['RNN/Attention_Dec.AttentionCell.output_t.W:0'] = self.w_fp['RNN/while/AttLSTM.AttentionCell.output_t/AttLSTM.AttentionCell.output_t.W:0'][...]
		cnn_weight_dict['logits.W:0'] = self.w_fp['MLP.1/MLP.1.W:0'][...]
		cnn_weight_dict['logits.b:0'] = self.w_fp['MLP.1/MLP.1.b:0'][...]


		for param in tf.all_variables():
			print "{},{}".format(param.get_shape(),param.name)
			if 'conv' in param.name or 'Batch' in param.name :
				self.sess.run(param.assign(cnn_weight_dict[param.name]))
			else:
				self.sess.run(param.assign(cnn_weight_dict[param.name]))

	def predict_hand(self,set='test',batch_size=1,visualize=True):
		imgs=[]
		for i in range(batch_size):
			imgs.append(np.asarray(Image.open(self.image).convert('YCbCr'))[:,:,0][:,:,None])

		imgs = np.asarray(imgs,dtype=np.float32)

		inp_seqs = np.zeros((batch_size,160)).astype('int32')
		print imgs.shape

		inp_seqs[:,0] = np.load('working_on_best_model/model/properties.npy').tolist()['char_to_idx']['#START']

		for i in xrange(1,160):
			input_feed={}
			input_feed[self.img_ip.name] = imgs
			input_feed[self.decoder_input.name] = inp_seqs[:,:i]
			output_feed = [ self.logits,self.conv_op]
			op = self.sess.run(output_feed,input_feed)
			prediction = tf.to_int32(tf.argmax( op[0], 2))
			prediction_num = np.array(prediction.eval(session=self.sess))
			inp_seqs[:,i] = prediction_num[:,i-1]
		np.save('working_on_best_model/model/pred_imgs_hw.npy',imgs)
		np.save('working_on_best_model/model/pred_latex_hw.npy',inp_seqs)

		#np.save('/home/abhitrip/Downloads/website/working_on_best_model/model/pred_imgs_hw.npy',imgs)
		#np.save('/home/abhitrip/Downloads/website/working_on_best_model/model/pred_latex_hw.npy',inp_seqs)








def test_image(img):

	obj = Model(
		'test',
		cnn_pretrain_path='working_on_best_model/model/rweights.h5',
		batch_size=1,
		image=img
		)

	print img
	obj.predict_hand()

	import numpy as np
	import re
	from IPython.display import display, Math, Latex, Image

	imgs = np.load('working_on_best_model/model/pred_imgs_hw.npy')
	preds = np.load('working_on_best_model/model/pred_latex_hw.npy')
	properties = np.load('/home/abhitrip/Downloads/website/working_on_best_model/model/properties.npy').tolist()
	displayPreds = lambda Y: display(Math(Y.split('#END')[0]))
	idx_to_chars = lambda Y: ' '.join(map(lambda x: properties['idx_to_char'][x],Y))
	#displayIdxs = lambda Y: display(Math(''.join(map(lambda x: properties['idx_to_char'][x],Y))))


	preds_chars = idx_to_chars(preds[0,1:]).replace('$','')

	output =  preds_chars.split('#END')[0]
	return output

if __name__=='__main__':
	op = test_image('working_on_best_model/model/1be6abc0ff.png')
	print op






