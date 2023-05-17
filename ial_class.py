import tensorflow as tf
import os
import numpy as np
from numpy.random import default_rng
import build_unet
from tqdm.auto import tqdm


class TrainObject():
    def __init__(self,fname,root):
        self.fname = fname
        self.root = root
        self.batch_size = len(fname)
    
    def load_data(self):
        x = []
        y = []
        click_window = []
        for i in range(self.batch_size):
            x.append(np.load(os.path.join(self.root,'vol_'+self.fname[i])))
            y_t = np.load(os.path.join(self.root,'mask_tumor_'+self.fname[i]))
            y_h = np.load(os.path.join(self.root,'mask_healthy_'+self.fname[i]))
            y_b = np.ones(y_t.shape) - (y_t+y_h)
            y.append([y_t,y_h,y_b])
            c_t = np.load(os.path.join(self.root,'click_tumor_'+self.fname[i]))
            c_h = np.load(os.path.join(self.root,'click_healthy_'+self.fname[i]))
            c_b = np.load(os.path.join(self.root,'click_background_'+self.fname[i]))
            click_window.append([c_t,c_h,c_b])
            
        self.x = np.array(x)
        self.y = np.transpose(np.array(y),(0,2,3,1))
        self.click_window = np.transpose(np.array(click_window),(0,2,3,1))
        #raw slice
        self.x = tf.cast(self.x,tf.float32)
        #true channel level masks
        self.y = tf.cast(self.y,tf.float32)
        #honest clicked regions
        self.click_window = tf.cast(self.click_window,tf.float32)
        
class DiceLossROI(tf.keras.losses.Loss):
    """
    To be used when ground truth NOT available
    DiceLoss restricted to clicked regions
    Input will be dl_roi(TrainObject.click_window,TrainObject.pred)
    """
    def __init__(self, alpha = [1/3,1/3,1/3],**kwargs):
        super(DiceLossROI, self).__init__(**kwargs)
        self.smooth = 1.e-6
        self.alpha = tf.cast(alpha,tf.float32)
        
    def call(self,y_true,y_pred):
        """
        Assumes y_true is the click window (and accurately captures roi)
        y_pred is model output (to be windowed to clicked region)
        [batch_size,512,512,3] - 3 channels: tumor, healthy, background
        Returns alpha_1 dice(tumor) + alpha_2 dice(healthy) + alpha_3 dice(background)
        """
        #begin by windowing y_pred to only within click region
        
        y_pred = tf.cast(y_pred,tf.float32)
        y_true = tf.cast(y_true,tf.float32)
        y_pred = tf.math.multiply(y_true,y_pred)
        
        num = tf.reduce_sum(tf.square(y_true - y_pred),axis = (1,2))
        dem = tf.reduce_sum(tf.square(y_true), axis = (1,2)) + tf.reduce_sum(tf.square(y_pred), axis = (1,2)) + self.smooth
        temp = tf.math.divide(num,dem) #should be shape: batch_size x 3
        temp = tf.reduce_mean(temp,axis=0)
        return tf.reduce_sum(tf.math.multiply(self.alpha,temp))
class ClickDiceROI():
    def __init__(self,flavor='all',alpha=[1/3,1/3,1/3]):
        self.smooth = 1e-6
        self.flavor = flavor
        self.alpha = np.array(alpha)

    def __str__(self):
        print(f'Provides dice score inside click windows\nFlavor is either all, tumor, liver, whole liver, or background')
        print(f'alpha are weights to blend/normalize if using all channels\nie final dice = alpha.*[tumor, liver, bg] dice')
            
        
    def call(self,window,pred):
        #window = window.numpy()
        #pred = pred.numpy()
        if window.ndim != pred.ndim:
            print(f'window is {window.ndim} dim but pred is {pred.ndim} dim')
            return -1
        if window.ndim != 3:
            print(f'window.ndim: Expected 3 but got {window.ndim}')
            return -1
        if window.shape[-1] != 3:
            print(f'Incorrect image dimensions: need 512x512x3 but got {window.shape}')
        #at this point can assume image is 512x512x3
        num = 2*np.sum(window*pred,axis=(0,1))
        dem = np.sum(window,axis=(0,1)) + np.sum(pred,axis=(0,1)) + self.smooth
        dice = num/dem
        
        if self.flavor == 'all':
            return np.sum(self.alpha*dice)
        elif self.flavor == 'tumor':
            return dice[0]
        elif self.flavor == 'liver':
            return dice[1]
        elif self.flavor == 'whole liver':
            return np.sum(self.alpha*dice[:2])
        elif self.flavor == 'background':
            return dice[2]
        else:
            print(f'Need to give valid flavor, {self.flavor} not accepted')
            return -1
        

class DiceLoss(tf.keras.losses.Loss):
    """
    Expected input to.y, to.pred
    """
    def __init__(self,alpha=[1/3,1/3,1/3],**kwargs):
        super(DiceLoss,self).__init__(**kwargs)
        self.smooth = 1.e-6
        self.alpha = tf.cast(alpha,tf.float32)
        
    def __str__(self):
        print('hello\nhi')
        
    def call(self,y_true,y_pred):
        y_pred = tf.cast(y_pred,tf.float32)
        y_true = tf.cast(y_true,tf.float32)
        
        num = tf.reduce_sum(tf.square(y_true - y_pred),axis = (1,2))
        dem = tf.reduce_sum(tf.square(y_true), axis = (1,2)) + tf.reduce_sum(tf.square(y_pred), axis = (1,2)) + self.smooth
        temp = tf.math.divide(num,dem) #should be shape: batch_size x 3
        temp = tf.reduce_mean(temp,axis=0)
        return tf.reduce_sum(tf.math.multiply(self.alpha,temp))
        
        
        
class InterActiveLearning():
    def __init__(self,path_to_files,label_train_pid,unlabel_train_pid,path_to_save):
        self.path_to_files = path_to_files
        self.path_to_save = path_to_save
        
        self.train_set_label = label_train_pid
        self.num_train = len(self.train_set_label)
        self.unlabel_pool = unlabel_train_pid
        
        self.load_pid()
        
        self.rng = default_rng(seed=42)
        
        self.click_dice = ClickDiceROI()
        
    def set_dice_flavor(self,flavor='all',alpha=[1/3,1/3,1/3]):
        self.click_dice = ClickDiceROI(flavor,alpha)


        
    def load_pid(self):
        for rt, dr, files in os.walk(self.path_to_files+'val'):
            if dr == []:
                self.val_set = ['_'.join(f.split('_')[2:]) for f in files if 'mask_tumor' in f]

        for rt, dr, files in os.walk(self.path_to_files+'test'):
            if dr == []:
                self.test_set = ['_'.join(f.split('_')[2:]) for f in files if 'mask_tumor' in f]
        #this is useless to define here since it changes each AL iteration        
        #self.num_train = len(self.train_set_label)
        self.num_val = len(self.val_set)
        self.num_test = len(self.test_set)

    
    def initialize_ML(self,learning_rate=3e-5,batch_size=10,num_epochs=100,seed=42):
        input_layer = tf.keras.Input((512,512,1))
        self.unet = build_unet.build_unet2d(input_layer,num_classes=3,seed=seed)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)
        self.ml_loss = DiceLoss()
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.train_loss_history = []
        self.val_loss_history = []
        line = os.path.join(self.path_to_save,f'AL_iteration_{self.num_train}_samples')
        fldr_exists = os.path.isdir(line)
        if not fldr_exists:
            os.makedirs(line)
            print(f"Created folder: {line}\n")
        self.ckpt_path = os.path.join(line,'best_model')
        self.ckpt_dir = os.path.dirname(self.ckpt_path)
        
    def run_ML(self,save=True):
        best_val_loss = np.inf
        for e in tqdm(range(self.num_epochs)):
            print(f'\nEpoch: {e}')
            epoch_train_loss = 0
            for b in range(int(len(self.train_set_label)/self.batch_size)):
                files = self.train_set_label[b*self.batch_size:(b+1)*self.batch_size]
                to = TrainObject(files,self.path_to_files+'train')
                to.load_data()
                
                with tf.GradientTape(persistent=True) as tape:
                    pred = self.unet(to.x,training=True)
                    loss = self.ml_loss(to.y,pred)
                epoch_train_loss += loss.numpy()*len(files)/self.num_train
                grads = tape.gradient(loss,self.unet.trainable_weights)
                self.optimizer.apply_gradients(zip(grads,self.unet.trainable_weights))
                
            self.train_loss_history.append(epoch_train_loss)
            
            #Now evaluate on validation set
            epoch_val_loss = 0
            for b in range(int(self.num_val/self.batch_size)):
                files = self.val_set[b*self.batch_size:(b+1)*self.batch_size]
                to = TrainObject(files,self.path_to_files+'val')
                to.load_data()
                
                pred = self.unet(to.x,training=False)
                loss = self.ml_loss(to.y,pred)
                epoch_val_loss += loss.numpy()*len(files)/self.num_val
            print(f'Training Loss: {epoch_train_loss:0.3f}')
            print(f'Validation Loss: {epoch_val_loss:0.3f}')
            self.val_loss_history.append(epoch_val_loss)
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                if save:
                    print('Updating best model')
                    self.unet.save_weights(self.ckpt_path)
                    
        self.tlh = np.array(self.train_loss_history)
        self.vlh = np.array(self.val_loss_history)
        if save:
            np.save(self.ckpt_dir+'/train_loss_history',self.tlh)
            np.save(self.ckpt_dir+'/val_loss_history',self.vlh)
            
    def load_ML(self,path_to_best):
        input_layer = tf.keras.Input((512,512,1))
        self.unet = build_unet.build_unet2d(input_layer,num_classes=3)
        self.unet.load_weights(path_to_best)
        
    def compute_dice_uncertainty(self):
        dice_in_roi = []
        for i in tqdm(range(len(self.unlabel_pool))):
            file = [self.unlabel_pool[i]]
            to = TrainObject(file,self.path_to_files+'train')
            to.load_data()
            
            pred = self.unet(to.x,training=False)
            loss = self.click_dice.call(np.squeeze(to.click_window.numpy()),np.squeeze(pred.numpy()))
            dice_in_roi.append(-loss)
        self.uncertainty = np.array(dice_in_roi)
        
    def compute_random_uncertainty(self):
        self.uncertainty = self.rng.uniform(low=0,high=1,size=len(self.unlabel_pool))
    
    def compute_entropy_uncertainty(self):
        entropy = []
        for i in tqdm(range(len(self.unlabel_pool))):
            file = [self.unlabel_pool[i]]
            to = TrainObject(file,self.path_to_files+'train')
            to.load_data()
            
            pred = self.unet(to.x,training=False)
            entropy.append(tf.reduce_sum(-tf.math.multiply(pred,tf.math.log(pred))).numpy())
        self.uncertainty = np.array(entropy)
            
        
    def draw_most_uncertain(self,N=50):
        #https://stackoverflow.com/questions/34226400/find-the-index-of-the-k-smallest-values-of-a-numpy-array
        #argpartition with -N to get N largest values
        idx = np.argpartition(ob.uncertainty,-N)
        self.new_pids = np.array(self.unlabel_pool)[idx[-N:]]
        
    def update_pools(self):
        np.save(os.path.join(self.path_to_save,'training_set'),np.array(self.train_set_label))
        np.save(os.path.join(self.path_to_save,'sampled_pids'),np.array(self.new_pids))
        self.unlabel_pool = [p for p in ob.unlabel_pool if p not in ob.new_pids]
        self.train_set_label += list(self.new_pids)
        self.num_train = len(self.train_set_label)
    
    def test_set_evaluation(self,alpha=[1/3,1/3,1/3]):
        dice_lf = DiceLoss(alpha)
        self.test_dice = []
        for b in tqdm(range(self.num_test)):
            file = [self.test_set[b]]
            to = TrainObject(file,self.path_to_files+'test')
            to.load_data()
            
            pred = self.unet(to.x,training=False)
            loss = dice_lf(to.y,pred)
            self.test_dice.append(1-loss.numpy())
    def print_test_stats(self):
        print(f'Average dice score on test set of {self.num_test} scans: {np.mean(self.test_dice):0.3f}')
        print(f'Standard deviation: {np.std(self.test_dice):0.3f}')
        print(f'Median: {np.median(self.test_dice):0.3f}')
        print(f'Range of dice scores: [{np.min(self.test_dice):0.3f}, {np.max(self.test_dice):0.3f}]')
        