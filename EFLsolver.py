import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

G = 6.67e-11
c = 3e8
kb = 1.38e-23
h = 6.62e-34
km_per_Mpc = 3.08e19

class w0waCDM:

    def __init__(self,H0,Om0,Ode0,w0=-1.0,wa=0.0,Tcmb0=2.72):
        self.H0 = H0
        self.Om0 = Om0
        self.Ode0 = Ode0
        self.w0 = w0
        self.wa = wa
        self.Tcmb0 = Tcmb0

        self.hubble_time = km_per_Mpc/self.H0
        self.hubble_time_gyr = self.hubble_time/31536000e9

        self.critical_density_0 = 3*(1/self.hubble_time)**2*c**2/(8*np.pi*G)
        self.Or0 = 8*np.pi**5*kb**4/(15*h**3*c**3)*self.Tcmb0**4/self.critical_density_0
        self.Ok0 = 1-self.Om0-self.Ode0-self.Or0
        self.K0 = -self.Ok0 # Unit : Hubble radius **-2

        if self.K0 == 0:
            self.k = 0
        else:
            self.k = self.K0/np.abs(self.K0)

        self.solve_a(np.linspace(0,1,100000))

        pass

    def density(self,a,Omega0,w0,wa):
        return Omega0/a**(3*(1+w0+wa))*np.exp(3*wa*(a-1))
    
    def CDM_density(self,a):
        return self.density(a,self.Om0,0,0)
    
    def radiation_density(self,a):
        return self.density(a,self.Or0,1/3,0)
    
    def curvature_density(self,a):
        return self.density(a,self.Ok0,-1/3,0)
    
    def DE_density(self,a):
        return self.density(a,self.Ode0,self.w0,self.wa)
    
    def radius(self,t):
        if self.k == 0:
            return np.inf*np.ones(np.shape(t))
        else:
            return np.sqrt(np.abs(1/self.K0))*self.a(t)
        
    def conformal_time(self,t,n_points=10000):
        t_values = np.linspace(0,t,n_points)
        return np.trapz(y=1/self.a(t_values),x=t_values,axis=0)
    
    def angular_distance(self,t):
        return self.a(t)*self.comoving_distance(t)
    
    def comoving_distance(self,t):
        return -self.conformal_time(t)
    
    def luminosity_distance(self,t):
        return self.comoving_distance(t)/self.a(t)
    
    def light_cone(self):
        if self.k>0:
            return np.clip(self.comoving_distance(self.age_past),a_min=0,a_max=np.pi*self.radius(0))
        else:
            return self.comoving_distance(self.age_past)
    
    def event_horizon(self):
        return self.comoving_distance(self.age_past)-self.comoving_distance(self.age_future)
    
    def d2adt2(self,a):
        """
        One time unit is one Hubble time (1/H0)
        """
        return -1/2*(self.CDM_density(a)+2*self.radiation_density(a)+(1+3*self.w0+3*self.wa*(1-a))*self.DE_density(a))*a
    
    def function_for_solver(self,arr,t):
        return np.array([self.d2adt2(arr[1]),arr[0]])
    
    def solve_a(self,time_array):
        X0 = np.array([1,1])
        time_array_minus = -time_array[1:]
        time_array_plus = time_array
        dadt_a_plus = odeint(self.function_for_solver,X0,time_array_plus)
        dadt_a_minus = odeint(self.function_for_solver,X0,time_array_minus)
        a_array_plus = dadt_a_plus[:,1]
        H_array_plus = dadt_a_plus[:,0]/dadt_a_plus[:,1]
        a_array_minus = dadt_a_minus[1:,1]
        H_array_minus = dadt_a_minus[1:,0]/dadt_a_minus[1:,1]

        min_a_past = 1e-4
        min_a_future = 1e-4
        self.age_past = time_array_minus[np.min(np.concatenate((np.array([len(a_array_minus)]),np.arange(len(a_array_minus))[a_array_minus<=min_a_past])))]
        self.age_future = time_array[np.min(np.concatenate((np.array([len(a_array_plus)-1]),np.arange(len(a_array_plus))[a_array_plus<=min_a_future])))]
        self.bigbang=False
        self.bigcrunch=False

        if self.age_past>=-np.max(time_array):
            refined_time_array_minus = np.flip(2-np.geomspace(1,2,500000))*self.age_past
            dadt_a_minus = odeint(self.function_for_solver,X0,refined_time_array_minus)
            a_array_minus = dadt_a_minus[1:,1]
            H_array_minus = dadt_a_minus[1:,0]/dadt_a_minus[1:,1]
            time_array_minus = refined_time_array_minus[1:]
            self.age_past = refined_time_array_minus[np.min(np.concatenate((np.array([len(a_array_minus)]),np.arange(len(a_array_minus))[a_array_minus<=min_a_past])))]
            self.bigbang=True

        if self.age_future<=np.max(time_array):
            refined_time_array_plus = np.flip(2-np.geomspace(1,2,500000))*self.age_future
            dadt_a_plus = odeint(self.function_for_solver,X0,refined_time_array_plus)
            a_array_plus = dadt_a_plus[:,1]
            H_array_plus = dadt_a_plus[:,0]/dadt_a_plus[:,1]
            time_array_plus = refined_time_array_plus
            self.bigcrunch=True
            self.age_future = refined_time_array_plus[np.min(np.concatenate((np.array([len(a_array_plus)-1]),np.arange(len(a_array_plus))[a_array_plus<=min_a_future])))]

        self.time_array = np.concatenate((np.flip(time_array_minus),time_array_plus))
        self.a_array = np.concatenate((np.flip(a_array_minus),a_array_plus))
        self.H_array = np.concatenate((np.flip(H_array_minus),H_array_plus))
        self.z_array = 1/self.a_array-1

        nnan_ids = (~np.isnan(self.a_array))*(~np.isnan(self.H_array))
        self.time_array = self.time_array[nnan_ids]
        self.a_array = self.a_array[nnan_ids]
        self.H_array = self.H_array[nnan_ids]
        self.z_array = self.z_array[nnan_ids]

        if np.all(self.H_array>0) or np.all(self.H_array<0):
            self.redshift_usable=True
        else:
            self.redshift_usable=False

        pass

    def a(self,t):
        return np.interp(t,self.time_array,self.a_array)
    
    def H(self,t):
        return self.H0*np.interp(t,self.time_array,self.H_array)
    
    def z(self,t):
        return np.interp(t,self.time_array,self.z_array)
    
    def t(self,z):
        if self.redshift_usable:
            return (self.H0<0)*np.interp(z,self.z_array,self.time_array)+(self.H0>0)*np.interp(z,np.flip(self.z_array),np.flip(self.time_array))
        else:
            raise ValueError('Redshift-time relation is not bijective for this cosmological model.')

cosmo = w0waCDM(67.6,0.311,0.689,w0=-0.84,wa=-0.42)
LCDM = w0waCDM(67.6,0.311,0.689)
time = np.linspace(cosmo.age_past,1,1000)
lcdm_time = np.linspace(LCDM.age_past,1,1000)
print(f'Light cone size : {300/cosmo.H0*cosmo.light_cone():.2f} Gpc = {3.26*300/cosmo.H0*cosmo.light_cone():.2f} Glyr')
print(f'Radius : {300/cosmo.H0*cosmo.radius(0):.2f} Gpc')
print(f'Light cone size at time limit: {300/cosmo.H0*cosmo.event_horizon():.2f} Gpc = {3.26*300/cosmo.H0*cosmo.event_horizon():.2f} Glyr')
print(f'Age : {-cosmo.hubble_time_gyr*cosmo.age_past:.2f} Gyr')

plot=True
if plot:
    plt.style.use('aesthetics.mplstyle')
    plt.plot(cosmo.hubble_time_gyr*time,cosmo.a(time),linewidth=1,label=f'$w_0 w_a$CDM')
    plt.plot(LCDM.hubble_time_gyr*lcdm_time,LCDM.a(lcdm_time),linewidth=1,label=f'$\\Lambda$CDM')
    plt.plot([0],[1],'+r',markersize=20)
    plt.xlabel("Gyr")
    plt.ylabel(f"Scale factor")
    # plt.xlim([LCDM.hubble_time_gyr*LCDM.age_past,None])
    plt.ylim([0,None])
    plt.legend()
    plt.show()