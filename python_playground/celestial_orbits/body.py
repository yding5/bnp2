import numpy as np
import math
class Body:
    def __init__(self, mass, x0, y0, z0, vx0, vy0, vz0):
        """
        Body definition
        """
        #Mass in Kg
        self.mass = mass 
        
        # current x and y coordinates of the missile
        self.x = x0
        self.y = y0
        self.z = z0 
        
        # current value of velocity components
        self.K1 = 1
        self.K2 = 1
        self.vx  = self.K2*vx0
        self.vy  = self.K2*vy0
        self.vz  = self.K2*vz0
        
        # acceleration by x and y axes
        self.ax   = 0
        self.ay   = 0
        self.az   = 0
        
        # these list will contain discrete set of coordinates
        self.xarr = [self.x]
        self.yarr = [self.y]
        self.zarr = [self.z]
        
        
        
    def updateX(self, dt):
        self.x = self.x + self.vx*dt + self.ax*dt*dt
        return self.x
    
    def updateY(self, dt):
        self.y = self.y +  self.vy*dt + self.ay*dt*dt
        return self.y
    
    def updateZ(self, dt):
        self.z = self.z +  self.vz*dt + self.az*dt*dt
        return self.z
    
    def updateAx(self, bodies, body_index):
        self.ax=0
        for index, external_body in enumerate(bodies):
            if index != body_index:
                r = (self.x - external_body.x)**2 + (self.y - external_body.y)**2 + (self.z - external_body.z)**2
                r = math.sqrt(r)
                tmp = external_body.mass / r**3
                self.ax += tmp * (external_body.x - self.x)
                
        return self.K1*self.ax
    
    def updateAy(self, bodies, body_index):
        self.ay=0
        for index, external_body in enumerate(bodies):
            if index != body_index:
                r = (self.x - external_body.x)**2 + (self.y - external_body.y)**2 + (self.z - external_body.z)**2
                r = math.sqrt(r) #norm
                tmp = external_body.mass / r**3
                self.ay += tmp * (external_body.y - self.y)
        return self.K1*self.ay
    
    def updateAz(self, bodies, body_index):
        self.az=0
        for index, external_body in enumerate(bodies):
            if index != body_index:
                r = (self.x - external_body.x)**2 + (self.y - external_body.y)**2 + (self.z - external_body.z)**2
                r = math.sqrt(r)
                tmp = external_body.mass / r**3
                self.az += tmp * (external_body.z - self.z)
        return self.K1*self.az
        
    def updateVx(self, dt , bodies, body_index):
        self.vx = self.vx + 0.5*(self.ax + self.updateAx(bodies, body_index))*dt
        return self.K2*self.vx
    
    def updateVy(self, dt , bodies, body_index):
        self.vy = self.vy + 0.5*(self.ay + self.updateAy(bodies, body_index))*dt
        return self.K2*self.vy
    
    def updateVz(self, dt , bodies, body_index):
        self.vz = self.vz + 0.5*(self.az + self.updateAz(bodies, body_index))*dt
        return self.K2*self.vz