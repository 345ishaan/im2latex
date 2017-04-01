import numpy as np
import os
import sys
from PIL import Image
import tensorflow as tf
import h5py
import cnn
from ops import Embedding,Attention_Enc,Attention_Dec,set_weights
from data_gen import *


class Model(object):

	def __init__(self,phase,load_model=False,
		forward_only=False,
		vocab_list_path=None,
		data_base_dir=None,
		model_dir=None,
		data_path=None,
		formula_map=None,
		test_data_map=None,
		train_data_map=None,
		validate_data_map=None,
		cnn_pretrain_path=None,
		batch_size=20,
		num_epoch=30):
		
		self.forward_only = forward_only
		self.model_dir = model_dir
		self.num_epoch = 30
		self.phase=phase
		

		self.w_fp = h5py.File(cnn_pretrain_path)
		vocab = open(data_base_dir+'/'+vocab_list_path).readlines()
		self.map_v = {x.split('\n')[0]:i+4 for i,x in enumerate(vocab)}
		self.map_v['GO'] = 1
		self.map_v['EOS'] = 2
		self.map_v['PAD'] = 0
		self.map_v['UNKNOWN'] = 3

		self.map_i={}		
		for key in self.map_v:
			self.map_i[self.map_v[key]] = key
		self.map_i[0] = 'PAD'
		self.map_i[1] = 'GO'
		self.map_i[2] = 'EOS'
		self.map_i[3] = 'UNKNOWN'

		if self.phase == 'train':
			self.train_gen = DataGen(data_base_dir, data_path,formula_map,train_data_map,self.map_v,self.map_i)
			self.val_gen   = DataGen(data_base_dir, data_path,formula_map,validate_data_map,self.map_v,self.map_i,evaluate=True)
			#self.buckets = self.train_gen.bucket_specs
			#self.val_buckets = self.val_gen.bucket_specs
		elif self.phase == 'test':
			self.test_gen =  DataGen(data_base_dir,data_path,formula_map,test_data_map,self.map_v,self.map_i,evaluate=True)
			#self.test_buckets = self.test_gen.bucket_specs
			#self.bucket_init = self.test_buckets
			self.write_pred = open('pred.txt','w')
			self.write_ground = open('ground.txt','w')

		self.embedding_dim = 80
		self.target_vocab_size = len(self.map_v.keys())
		self.ENC_MAX_WIDTH = 50
		self.ENC_MAX_HEIGHT = 20
		self.ENC_OP_DIM = 512
		self.batch_size = batch_size
		self.ENC_DIM = 256
		self.DEC_DIM = 512
		self.MAX_CT_VEC_LENGTH = self.ENC_MAX_HEIGHT*self.ENC_MAX_WIDTH

		self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
		

		self.img_ip = tf.placeholder(shape=(None,None,None,1),dtype=tf.float32,name='data')
		self.conv_op = tf.placeholder(shape=(None,None,None,512),dtype=tf.float32,name='convop')
		self.decoder_input = tf.placeholder(shape=(None,None),dtype=tf.int32,name='seq')
		self.target_computeloss = tf.placeholder(tf.float32, [batch_size, None,None], name='target_computeloss')

		self.embedding_seqs = Embedding('embedding',self.target_vocab_size,self.embedding_dim,self.decoder_input)
		print self.embedding_seqs.get_shape()

		self.conv_op = cnn.CNN_Net(self.img_ip)
		print "Initialised CNN"
		self.encoder_output = Attention_Enc('Attention_Enc',self.conv_op,self.ENC_DIM,self.ENC_MAX_WIDTH,self.ENC_MAX_HEIGHT,self.ENC_OP_DIM,self.batch_size)
		print "Initialised Encoder"
		self.decoder_output ,self.logits= Attention_Dec('Attention_Dec',self.encoder_output,self.embedding_seqs,self.DEC_DIM,self.MAX_CT_VEC_LENGTH,self.ENC_OP_DIM,self.batch_size,self.embedding_dim,self.target_vocab_size)
		print "Initialised Decoder"

		self.trainable_params=[]
		for param in tf.all_variables():
			if 'conv' in param.name or 'Batch' in param.name:
				continue
			else:
				self.trainable_params.append(param)

		if not self.forward_only:
			self.norm_prob = tf.nn.softmax(logits=self.logits)
			self.bounded_norm_prob = tf.clip_by_value(self.norm_prob,clip_value_min=0.001,clip_value_max=0.999)
			self.losses = -tf.reduce_sum(self.target_computeloss*tf.log(tf.cast(self.bounded_norm_prob,dtype=tf.float32)),2)
			self.total_loss = tf.reduce_mean(self.losses)
			self.gradients = tf.gradients(self.losses, self.trainable_params)
			self.opt = tf.train.AdamOptimizer(learning_rate=0.001,beta1=0.9,beta2=0.999,epsilon=1e-08,use_locking=False,name='Adam')
			
			self.train_step = self.opt.apply_gradients(zip(self.gradients, self.trainable_params), global_step=tf.Variable(0, trainable=False))


		params_save_dict={}
		for p in tf.all_variables():
			params_save_dict[p.name] = p

		self.saver_all = tf.train.Saver(params_save_dict)

		ckpt = tf.train.get_checkpoint_state(model_dir)

		if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path) and load_model:
			print "Restoring Model"
			self.saver_all.restore(self.sess, ckpt.model_checkpoint_path)
		else:
			print "Initializing Model"
			self.sess.run(tf.initialize_all_variables())
		

		cnn_weight_dict = set_weights(self.w_fp)

		for param in tf.all_variables():
			#print "{},{}".format(param.get_shape(),param.name)
			if 'conv' in param.name or 'Batch' in param.name :
				self.sess.run(param.assign(cnn_weight_dict[param.name]))

	def pred_model(self):
		batch_gen = self.test_gen.gen(self.batch_size)
		batch = batch_gen.next()
		input_feed={}
		input_feed[self.img_ip.name] = np.array(batch['data']).transpose(0,2,3,1)
		for i in range(150):
			inp_seqs = np.array(batch['decoder_inputs']).T[:,:i]
			input_feed[self.decoder_input.name] = inp_seqs
			output_feed = [ self.logits,self.conv_op]
			op = self.sess.run(output_feed,input_feed)
			prediction = tf.to_int32(tf.argmax( op[0], 2))
			prediction_num = np.array(prediction.eval(session=self.sess))
			for xx in range(prediction_num.shape[1]):
				self.write_pred.write(str(self.map_i[prediction_num[0][xx]])+'\t')
			self.write_pred.write('\n')
		self.write_pred.close()

	
	def launch(self):
		if self.phase == 'test':
			self.pred_model()
		elif self.phase == 'train':
			prev_val_loss = sys.maxint
			
			for epoch in range(self.num_epoch):
				print "Ep**Ep"
				iteration =1
				for batch in self.train_gen.gen(self.batch_size):
					iteration += 1
					#bucket_id = batch['bucket_id']
					img_data = np.array(batch['data']).transpose(0,2,3,1)
					decoder_inputs = np.array(batch['decoder_inputs']).T
					dec_exp = batch['decoder_expectation']
					loss_op = self.step(img_data, decoder_inputs, dec_exp,1)
					if iteration %100 == 0:
						print "Train_Iter*****\t{}".format(iteration)
						print "Train Loss\t{}\n".format(loss_op)
				count = 0.0
				for batch in self.val_gen.gen(self.batch_size):
					#bucket_id = batch['bucket_id']
					img_data = np.array(batch['data']).transpose(0,2,3,1)
					decoder_inputs = np.array(batch['decoder_inputs']).T
					dec_exp = batch['decoder_expectation']
					loss_op = self.step(img_data, decoder_inputs, dec_exp,2)
					count += 1
				if loss_op/count < prev_val_loss:
					checkpoint_path = os.path.join(self.model_dir, "translate.ckpt")
					self.saver_all.save(self.sess, checkpoint_path)

	def step(self,img_data,decoder_inputs,dec_exp,step_flag):
		if step_flag == 1:
			input_feed={}
			input_feed[self.img_ip.name] = img_data
			input_feed[self.decoder_input.name] = decoder_inputs
			input_feed[self.target_computeloss.name] = np.eye(self.target_vocab_size)[np.array(dec_exp)]
			output_feed=[self.train_step,self.total_loss]
			op = self.sess.run(output_feed,input_feed)
			return op[1]
		elif step_flag == 2:
			input_feed={}
			input_feed[self.img_ip.name] = img_data
			input_feed[self.decoder_input.name] = decoder_inputs
			input_feed[self.target_computeloss.name] = np.eye(self.target_vocab_size)[np.array(dec_exp)]
			output_feed=[self.total_loss]
			op = self.sess.run(output_feed,input_feed)
			return op[0]

obj = Model('train',load_model=False,forward_only=False,
	cnn_pretrain_path=None,
	batch_size=20,
	data_base_dir=None,
	model_dir=None,
	data_path = 'smaller_images_processed',
	formula_map='formulas.norm.lst',
	train_data_map='train_filter.lst',
	validate_data_map='validate_filter.lst',
	test_data_map='test_filter.lst',
	vocab_list_path='latex_vocab.txt')

obj.launch()














