from matplotlib import animation, rc
import numpy as np
import matplotlib.pyplot as plt
import math


#Universe
class Universe:
    def __init__(self, K1=0, K2=0, num_bodies=4, g=9.8, air_friction_coeff=0.5):
        
        self.K1=K1
        self.K2=K2
        if K1==0 or K2==0:
            #Define universal gravitation constant
            G=6.67408e-11 #N-m2/kg2
            #Reference quantities
            m_nd=1.989e+30 #kg #mass of the sun
            r_nd=5.326e+12 #m #distance between stars in Alpha Centauri
            v_nd=30000 #m/s #relative velocity of earth around the sun
            t_nd=79.91*365*24*3600*0.51 #s #orbital period of Alpha Centauri
            #Net constants
            self.K1=G*t_nd*m_nd/(r_nd**2*v_nd)
            self.K2=v_nd*t_nd/r_nd
            
        print("Net Constants:")
        print("K1: ",self.K1)
        print("K2: ",self.K2)
        
        self.g = g
        self.air_friction_coeff = air_friction_coeff
        self.num_bodies = num_bodies
        self.bodies = []
        self.body_count = 0
        self.time = 0
        self.frames=100
        
    def addBody(self, body):
        if self.body_count<=self.num_bodies:
            body.K1 = self.K1
            body.K2 = self.K2
            self.bodies.append(body)
            self.body_count+=1
        else:
            print('Universe full, cannot add more bodies!')
        
    
    def step(self, dt):
        for index,body in enumerate(self.bodies):
            #update positions
            body.xarr.append(body.updateX(dt))
            body.yarr.append(body.updateY(dt))
            body.zarr.append(body.updateZ(dt))
            
            #update velocities and in turn acc
            body.updateVx(dt,self.bodies,index)
            body.updateVy(dt,self.bodies,index)
            body.updateVz(dt,self.bodies,index)
            
        #step in time
        self.time = self.time + dt
        
    def plot_trajectories3D(self, trajectories=[],frames=0):
        """Trajectories is a list of list containing a triplet of lists 
           for x, y and z co-ordinates for each object for each instance in time.
        """
        if frames==0:
            frames = self.frames
        if len(trajectories)==0:
            for body in self.bodies:
                trajectories.append([body.xarr, body.yarr,body.zarr])
                
        fig = plt.figure()
        ax = plt.axes(projection='3d')

        lines=[]
        empty_xs = []
        empty_ys = []
        empty_zs = []
        for i,_ in enumerate(trajectories):
            des = ['r-','b-','g-','y-']
            line, = ax.plot([], [], [], des[i])
            lines.append(line)
            empty_xs.append([])
            empty_ys.append([])
            empty_zs.append([])

        # initialization function: plot the background of each frame
        def init():
            for line in lines:
                line.set_data([], [])
                line.set_3d_properties([])
            return lines

        # animation function. This is called sequentially
        def animate(i):
            for j,traj in enumerate(trajectories):
                if i<len(traj[0]):
                    try:
                        empty_xs[j].append(traj[0][i])
                        empty_ys[j].append(traj[1][i]) 
                        empty_zs[j].append(traj[2][i])
                        lines[j].set_data(empty_xs[j],empty_zs[j])
                        lines[j].set_3d_properties(empty_ys[j])
                    except:
                        pass

            try:
                ax.set_xlim((-1+np.min(empty_xs), 1+np.max(empty_xs)))
                ax.set_ylim((-1+np.min(empty_zs), 1+np.max(empty_zs)))
                ax.set_zlim((-1+np.min(empty_ys), 1+np.max(empty_ys)))
            except:
                pass 

            return (lines)


        # call the animator. blit=True means only re-draw the parts that have changed.
        anim = animation.FuncAnimation(fig, animate, init_func=init,
                                       frames=frames, interval=100, blit=True)

        return anim
    
    def run_simulation(self, bodies, runtime=10, dt=0.001, sampling_rate=100):
        """
        Returns an animation object and trajectories.
        Use `HTML(anim.to_html5_video())` to visualize
        """

        for body in bodies:
            self.addBody(body)

        t = 0 # initial time
        self.step(dt)

        ###### Step through ######
        while self.time <= runtime:
            self.step(dt)
            t = t + dt
        ##############################
        trajectories = []
        for body in self.bodies:
            trajectories.append([body.xarr, body.yarr,body.zarr])
        
        #Simulation default params
        total_steps = int(runtime/dt) #(10,000)
        self.frames = range(0,total_steps,sampling_rate)
        
        anim = self.plot_trajectories3D(trajectories=trajectories)
        

        return anim, trajectories