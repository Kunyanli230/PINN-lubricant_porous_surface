import numpy as np
import time
from pyDOE import lhs
import matplotlib.pyplot as plt
import pickle
import scipy.io
import random
import tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use appropriate GPU setting

# Set seeds for reproducibility
random.seed(1234)
np.random.seed(1234)
tf.set_random_seed(1234)

class PINN_Stokes_Slip:
    def __init__(self, XY_c, INLET, OUTLET, TOP, BOTTOM, uv_layers, lb, ub, Ls, mu=0.02, ExistModel=0, uvDir=''):
        """
        PINN for steady incompressible Stokes flow with a slip condition at the bottom (lubricant porous surface).
        
        Governing equations (Stokes):
            mu*(u_xx + u_yy) - p_x = 0,
            mu*(v_xx + v_yy) - p_y = 0,
            u_x + v_y = 0.
        
        Boundary conditions:
            INLET (x=0): prescribed velocity [u,v] (e.g. a parabolic profile)
            OUTLET (x=1): prescribed pressure (p=0)
            TOP (y=H): no-slip: u=v=0
            BOTTOM (y=0): slip: u - Ls*u_y = 0, v = 0.
            
        Parameters:
            XY_c   : Collocation points (Nx2 array) in the domain
            INLET  : Inlet points as [x, y, u, v]
            OUTLET : Outlet points as [x, y, p] (p is prescribed, e.g. 0)
            TOP    : Top wall points as [x, y] (u=v=0 expected)
            BOTTOM : Bottom wall points as [x, y] (for slip condition)
            uv_layers: List of layer sizes (e.g. [2,40,40,...,3]) with 3 outputs: [u,v,p]
            lb, ub : Lower and upper bounds of the domain
            Ls     : Slip length at the bottom wall
            mu     : Dynamic viscosity
            ExistModel: if 1, load existing model from uvDir.
            uvDir  : File directory for saving/loading NN weights.
        """
        self.count = 0
        self.lb = lb
        self.ub = ub
        self.mu = mu
        self.Ls = Ls
        
        # Collocation points for PDE residual
        self.x_c = XY_c[:, 0:1]
        self.y_c = XY_c[:, 1:2]
        
        # Inlet boundary
        self.x_INLET = INLET[:, 0:1]
        self.y_INLET = INLET[:, 1:2]
        self.u_INLET = INLET[:, 2:3]
        self.v_INLET = INLET[:, 3:4]
        
        # Outlet boundary
        self.x_OUTLET = OUTLET[:, 0:1]
        self.y_OUTLET = OUTLET[:, 1:2]
        self.p_OUTLET = OUTLET[:, 2:3]
        
        # Top wall (no-slip)
        self.x_TOP = TOP[:, 0:1]
        self.y_TOP = TOP[:, 1:2]
        
        # Bottom wall (slip condition)
        self.x_BOTTOM = BOTTOM[:, 0:1]
        self.y_BOTTOM = BOTTOM[:, 1:2]
        
        # NN architecture
        self.uv_layers = uv_layers
        self.loss_rec = []
        
        if ExistModel == 0:
            self.uv_weights, self.uv_biases = self.initialize_NN(uv_layers)
        else:
            print("Loading existing NN...")
            self.uv_weights, self.uv_biases = self.load_NN(uvDir, uv_layers)
            
        # Define placeholders
        self.learning_rate = tf.placeholder(tf.float32, shape=[])
        self.x_tf = tf.placeholder(tf.float32, shape=[None, self.x_c.shape[1]])
        self.y_tf = tf.placeholder(tf.float32, shape=[None, self.y_c.shape[1]])
        
        self.x_INLET_tf = tf.placeholder(tf.float32, shape=[None, self.x_INLET.shape[1]])
        self.y_INLET_tf = tf.placeholder(tf.float32, shape=[None, self.y_INLET.shape[1]])
        self.u_INLET_tf = tf.placeholder(tf.float32, shape=[None, self.u_INLET.shape[1]])
        self.v_INLET_tf = tf.placeholder(tf.float32, shape=[None, self.v_INLET.shape[1]])
        
        self.x_OUTLET_tf = tf.placeholder(tf.float32, shape=[None, self.x_OUTLET.shape[1]])
        self.y_OUTLET_tf = tf.placeholder(tf.float32, shape=[None, self.y_OUTLET.shape[1]])
        self.p_OUTLET_tf = tf.placeholder(tf.float32, shape=[None, self.p_OUTLET.shape[1]])
        
        self.x_TOP_tf = tf.placeholder(tf.float32, shape=[None, self.x_TOP.shape[1]])
        self.y_TOP_tf = tf.placeholder(tf.float32, shape=[None, self.y_TOP.shape[1]])
        
        self.x_BOTTOM_tf = tf.placeholder(tf.float32, shape=[None, self.x_BOTTOM.shape[1]])
        self.y_BOTTOM_tf = tf.placeholder(tf.float32, shape=[None, self.y_BOTTOM.shape[1]])
        
        self.x_c_tf = tf.placeholder(tf.float32, shape=[None, self.x_c.shape[1]])
        self.y_c_tf = tf.placeholder(tf.float32, shape=[None, self.y_c.shape[1]])
        
        # PINN predictions
        self.u_pred, self.v_pred, self.p_pred = self.net_uv(self.x_tf, self.y_tf)
        self.f_pred_u, self.f_pred_v, self.f_pred_c = self.net_f(self.x_c_tf, self.y_c_tf)
        
        self.u_INLET_pred, self.v_INLET_pred, _ = self.net_uv(self.x_INLET_tf, self.y_INLET_tf)
        _, _, self.p_OUTLET_pred = self.net_uv(self.x_OUTLET_tf, self.y_OUTLET_tf)
        self.u_TOP_pred, self.v_TOP_pred, _ = self.net_uv(self.x_TOP_tf, self.y_TOP_tf)
        self.u_BOTTOM_pred, self.v_BOTTOM_pred, _ = self.net_uv(self.x_BOTTOM_tf, self.y_BOTTOM_tf)
        self.u_y_BOTTOM_pred = tf.gradients(self.u_BOTTOM_pred, self.y_BOTTOM_tf)[0]
        
        # Define losses
        self.loss_f = tf.reduce_mean(tf.square(self.f_pred_u)) + \
                      tf.reduce_mean(tf.square(self.f_pred_v)) + \
                      tf.reduce_mean(tf.square(self.f_pred_c))
        self.loss_INLET = tf.reduce_mean(tf.square(self.u_INLET_pred - self.u_INLET_tf)) + \
                          tf.reduce_mean(tf.square(self.v_INLET_pred - self.v_INLET_tf))
        self.loss_OUTLET = tf.reduce_mean(tf.square(self.p_OUTLET_pred - self.p_OUTLET_tf))
        self.loss_TOP = tf.reduce_mean(tf.square(self.u_TOP_pred)) + \
                        tf.reduce_mean(tf.square(self.v_TOP_pred))
        # For slip: u - Ls*u_y = 0, and v = 0 on the bottom wall
        self.loss_BOTTOM = tf.reduce_mean(tf.square(self.u_BOTTOM_pred - self.Ls * self.u_y_BOTTOM_pred)) + \
                           tf.reduce_mean(tf.square(self.v_BOTTOM_pred))
        
        self.loss = self.loss_f + 2*(self.loss_INLET + self.loss_OUTLET + self.loss_TOP + self.loss_BOTTOM)
        
        # Optimizers: first Adam then L-BFGS-B
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
                                                                var_list=self.uv_weights + self.uv_biases,
                                                                method='L-BFGS-B',
                                                                options={'maxiter': 100000,
                                                                         'maxfun': 100000,
                                                                         'maxcor': 50,
                                                                         'maxls': 50,
                                                                         'ftol': 1*np.finfo(float).eps})
        self.optimizer_Adam = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss,
                                                          var_list=self.uv_weights + self.uv_biases)
        
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        init = tf.global_variables_initializer()
        self.sess.run(init)
        
    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(num_layers - 1):
            W = self.xavier_init([layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1, layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)
        return weights, biases
    
    def xavier_init(self, size):
        in_dim, out_dim = size[0], size[1]
        xavier_stddev = np.sqrt(2/(in_dim+out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev, dtype=tf.float32), dtype=tf.float32)
    
    def save_NN(self, fileDir):
        uv_weights = self.sess.run(self.uv_weights)
        uv_biases = self.sess.run(self.uv_biases)
        with open(fileDir, 'wb') as f:
            pickle.dump([uv_weights, uv_biases], f)
            print("Saved NN parameters successfully.")
            
    def load_NN(self, fileDir, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        with open(fileDir, 'rb') as f:
            uv_weights, uv_biases = pickle.load(f)
            assert num_layers == (len(uv_weights)+1)
            for i in range(num_layers-1):
                W = tf.Variable(uv_weights[i], dtype=tf.float32)
                b = tf.Variable(uv_biases[i], dtype=tf.float32)
                weights.append(W)
                biases.append(b)
                print("Loaded NN parameters.")
        return weights, biases
    
    def neural_net(self, X, weights, biases):
        num_layers = len(weights)+1
        H = X
        for l in range(num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y
    
    def net_uv(self, x, y):
        uvp = self.neural_net(tf.concat([x, y], 1), self.uv_weights, self.uv_biases)
        u = uvp[:,0:1]
        v = uvp[:,1:2]
        p = uvp[:,2:3]
        return u, v, p
    
    def net_f(self, x, y):
        u, v, p = self.net_uv(x, y)
        mu = self.mu
        u_x = tf.gradients(u, x)[0]
        u_y = tf.gradients(u, y)[0]
        v_x = tf.gradients(v, x)[0]
        v_y = tf.gradients(v, y)[0]
        u_xx = tf.gradients(u_x, x)[0]
        u_yy = tf.gradients(u_y, y)[0]
        v_xx = tf.gradients(v_x, x)[0]
        v_yy = tf.gradients(v_y, y)[0]
        p_x = tf.gradients(p, x)[0]
        p_y = tf.gradients(p, y)[0]
        
        # Stokes residuals
        f_u = mu*(u_xx+u_yy) - p_x
        f_v = mu*(v_xx+v_yy) - p_y
        f_c = u_x + v_y
        return f_u, f_v, f_c
    
    def callback(self, loss):
        self.count += 1
        self.loss_rec.append(loss)
        print("{} th iteration, Loss: {}".format(self.count, loss))
        
    def train(self, iter, learning_rate):
        tf_dict = {self.x_c_tf: self.x_c, self.y_c_tf: self.y_c,
                   self.x_INLET_tf: self.x_INLET, self.y_INLET_tf: self.y_INLET, 
                   self.u_INLET_tf: self.u_INLET, self.v_INLET_tf: self.v_INLET,
                   self.x_OUTLET_tf: self.x_OUTLET, self.y_OUTLET_tf: self.y_OUTLET, 
                   self.p_OUTLET_tf: self.p_OUTLET,
                   self.x_TOP_tf: self.x_TOP, self.y_TOP_tf: self.y_TOP,
                   self.x_BOTTOM_tf: self.x_BOTTOM, self.y_BOTTOM_tf: self.y_BOTTOM,
                   self.learning_rate: learning_rate}
        loss_INLET_list = []
        loss_OUTLET_list = []
        loss_TOP_list = []
        loss_BOTTOM_list = []
        loss_f_list = []
        for it in range(iter):
            self.sess.run(self.train_op_Adam, tf_dict)
            if it % 10 == 0:
                loss_value = self.sess.run(self.loss, tf_dict)
                print("Iteration {}: Loss = {}".format(it, loss_value))
            loss_INLET_list.append(self.sess.run(self.loss_INLET, tf_dict))
            loss_OUTLET_list.append(self.sess.run(self.loss_OUTLET, tf_dict))
            loss_TOP_list.append(self.sess.run(self.loss_TOP, tf_dict))
            loss_BOTTOM_list.append(self.sess.run(self.loss_BOTTOM, tf_dict))
            loss_f_list.append(self.sess.run(self.loss_f, tf_dict))
            self.loss_rec.append(self.sess.run(self.loss, tf_dict))
        return loss_INLET_list, loss_OUTLET_list, loss_TOP_list, loss_BOTTOM_list, loss_f_list, self.loss
    
    def train_bfgs(self):
        tf_dict = {self.x_c_tf: self.x_c, self.y_c_tf: self.y_c,
                   self.x_INLET_tf: self.x_INLET, self.y_INLET_tf: self.y_INLET, 
                   self.u_INLET_tf: self.u_INLET, self.v_INLET_tf: self.v_INLET,
                   self.x_OUTLET_tf: self.x_OUTLET, self.y_OUTLET_tf: self.y_OUTLET, 
                   self.p_OUTLET_tf: self.p_OUTLET,
                   self.x_TOP_tf: self.x_TOP, self.y_TOP_tf: self.y_TOP,
                   self.x_BOTTOM_tf: self.x_BOTTOM, self.y_BOTTOM_tf: self.y_BOTTOM}
        self.optimizer.minimize(self.sess, feed_dict=tf_dict, fetches=[self.loss], loss_callback=self.callback)
    
    def predict(self, x_star, y_star):
        u_star, v_star, p_star = self.sess.run([self.u_pred, self.v_pred, self.p_pred],
                                               {self.x_tf: x_star, self.y_tf: y_star})
        return u_star, v_star, p_star
    
    def getloss(self):
        tf_dict = {self.x_c_tf: self.x_c, self.y_c_tf: self.y_c,
                   self.x_INLET_tf: self.x_INLET, self.y_INLET_tf: self.y_INLET, 
                   self.u_INLET_tf: self.u_INLET, self.v_INLET_tf: self.v_INLET,
                   self.x_OUTLET_tf: self.x_OUTLET, self.y_OUTLET_tf: self.y_OUTLET, 
                   self.p_OUTLET_tf: self.p_OUTLET,
                   self.x_TOP_tf: self.x_TOP, self.y_TOP_tf: self.y_TOP,
                   self.x_BOTTOM_tf: self.x_BOTTOM, self.y_BOTTOM_tf: self.y_BOTTOM}
        loss_INLET = self.sess.run(self.loss_INLET, tf_dict)
        loss_OUTLET = self.sess.run(self.loss_OUTLET, tf_dict)
        loss_TOP = self.sess.run(self.loss_TOP, tf_dict)
        loss_BOTTOM = self.sess.run(self.loss_BOTTOM, tf_dict)
        loss_f = self.sess.run(self.loss_f, tf_dict)
        loss = self.sess.run(self.loss, tf_dict)
        return loss_INLET, loss_OUTLET, loss_TOP, loss_BOTTOM, loss_f, loss

# Simple post-processing routine to visualize the predicted fields
def postProcess(xmin, xmax, ymin, ymax, field):
    x, y, u, v, p = field
    fig, ax = plt.subplots(1,3, figsize=(15,4))
    cf = ax[0].scatter(x, y, c=u, cmap='rainbow', s=3)
    ax[0].set_title("u")
    ax[0].set_xlim([xmin,xmax])
    ax[0].set_ylim([ymin,ymax])
    fig.colorbar(cf, ax=ax[0])
    
    cf = ax[1].scatter(x, y, c=v, cmap='rainbow', s=3)
    ax[1].set_title("v")
    ax[1].set_xlim([xmin,xmax])
    ax[1].set_ylim([ymin,ymax])
    fig.colorbar(cf, ax=ax[1])
    
    cf = ax[2].scatter(x, y, c=p, cmap='rainbow', s=3)
    ax[2].set_title("p")
    ax[2].set_xlim([xmin,xmax])
    ax[2].set_ylim([ymin,ymax])
    fig.colorbar(cf, ax=ax[2])
    
    plt.savefig("stokes_slip.png", dpi=300)
    plt.close('all')

if __name__ == "__main__":
    # Domain bounds: x in [0,1], y in [0,0.5]
    lb = np.array([0, 0])
    ub = np.array([1.0, 0.5])
    
    # NN configuration: input=2, several hidden layers, output=3 (u,v,p)
    uv_layers = [2] + 8*[40] + [3]
    
    # Inlet at x=0: using a parabolic profile u = 4*U_max*y*(H-y)/H^2, v=0
    num_inlet = 201
    x_inlet = np.zeros((num_inlet, 1))
    y_inlet = np.linspace(lb[1], ub[1], num_inlet)[:, None]
    U_max = 1.0
    u_inlet = 4 * U_max * y_inlet * (ub[1]-y_inlet)/(ub[1]**2)
    v_inlet = np.zeros_like(u_inlet)
    INLET = np.concatenate((x_inlet, y_inlet, u_inlet, v_inlet), axis=1)
    
    # Outlet at x=1: p=0 (for simplicity)
    num_outlet = 201
    x_outlet = np.ones((num_outlet, 1))
    y_outlet = np.linspace(lb[1], ub[1], num_outlet)[:, None]
    p_outlet = np.zeros((num_outlet, 1))
    OUTLET = np.concatenate((x_outlet, y_outlet, p_outlet), axis=1)
    
    # Top wall at y = 0.5 (no-slip: u=v=0)
    num_top = 201
    x_top = np.linspace(lb[0], ub[0], num_top)[:, None]
    y_top = ub[1]*np.ones((num_top, 1))
    TOP = np.concatenate((x_top, y_top), axis=1)
    
    # Bottom wall at y = 0 (slip condition: u - Ls*u_y=0, v=0)
    num_bottom = 201
    x_bottom = np.linspace(lb[0], ub[0], num_bottom)[:, None]
    y_bottom = lb[1]*np.ones((num_bottom, 1))
    BOTTOM = np.concatenate((x_bottom, y_bottom), axis=1)
    
    # Generate collocation points using Latin Hypercube Sampling
    num_collo = 40000
    XY_c = lb + (ub - lb) * lhs(2, num_collo)
    
    # Visualize the distribution of collocation and boundary points
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    plt.scatter(XY_c[:,0], XY_c[:,1], marker='o', alpha=0.1, color='blue')
    plt.scatter(INLET[:,0], INLET[:,1], marker='o', alpha=0.3, color='red')
    plt.scatter(OUTLET[:,0], OUTLET[:,1], marker='o', alpha=0.3, color='orange')
    plt.scatter(TOP[:,0], TOP[:,1], marker='o', alpha=0.3, color='green')
    plt.scatter(BOTTOM[:,0], BOTTOM[:,1], marker='o', alpha=0.3, color='purple')
    plt.show()
    
    # Define slip length (example value)
    Ls = 0.05
    
    with tf.device('/device:GPU:0'):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        session = tf.Session(config=config)
        
        # Either train from scratch or load an existing model.
        # To train from scratch, set ExistModel=0.
        # model = PINN_Stokes_Slip(XY_c, INLET, OUTLET, TOP, BOTTOM, uv_layers, lb, ub, Ls, mu=0.02)
        model = PINN_Stokes_Slip(XY_c, INLET, OUTLET, TOP, BOTTOM, uv_layers, lb, ub, Ls, mu=0.02, ExistModel=1, uvDir='uvNN_stokes_slip.pickle')
        
        start_time = time.time()
        loss_INLET, loss_OUTLET, loss_TOP, loss_BOTTOM, loss_f, loss = model.train(iter=10000, learning_rate=5e-4)
        model.train_bfgs()
        print("--- %s seconds ---" % (time.time() - start_time))
        
        model.save_NN('uvNN_stokes_slip.pickle')
        with open('loss_history_stokes_slip.pickle', 'wb') as f:
            pickle.dump(model.loss_rec, f)
        
        # Prediction on a grid for visualization
        x_PINN = np.linspace(lb[0], ub[0], 251)
        y_PINN = np.linspace(lb[1], ub[1], 101)
        x_PINN, y_PINN = np.meshgrid(x_PINN, y_PINN)
        x_PINN = x_PINN.flatten()[:, None]
        y_PINN = y_PINN.flatten()[:, None]
        u_PINN, v_PINN, p_PINN = model.predict(x_PINN, y_PINN)
        field = [x_PINN, y_PINN, u_PINN, v_PINN, p_PINN]
        postProcess(lb[0], ub[0], lb[1], ub[1], field)
