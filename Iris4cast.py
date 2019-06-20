#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 10:34:22 2019
A class based formulation of other analyses.
It is structured as:
    
                            Dataset
                            _|   |_
                           |       |
                       Analysis  Forecast
                    ________________|________________
                    |               |               |
                SubxForecast  EC45Forecast  Seas5Forecast
                
Dataset initialises the majority of the variables, handles data loading, copying
and subsetting, and  provides deseasonalising and data reduction methods.

Analysis adds a preprocessing method for era5 data, and some additional variable setup

Forecast adds an error correction method, and forecast-specific variable setup

SubxForecast, EC45Forecast, and Seas5Forecast add filetype specific data processing.
@author: josh
"""

        

import iris
import copy as cp
import datetime as dt
import iris.coord_categorisation as iccat
from iris.analysis.cartography import cosine_latitude_weights
import numpy as np
import cf_units
import os

class Dataset:
    
    def __init__(self,field,dates,leads=None):
        

        """
    	Dataset is the base class shared by all analysis and forecast data sets. It defines
    	all functions that are generic between datasets. Not normally used directly.
    
    	
    	Args:
    	* field - A string used to identify which fields to load from file.
        
        *date - a list or tuple of 2 datetime.datetime objects specifying the
                first and last datetime to include in the data
            
        *leads - used by the Forecast class only, a list or tuple of 2 floats,
                 specifying minimum and maximum lead times in days to include.
    	
    	"""

        self.field=field
        self.dates=dates
        self._d_l,self._d_u=dates
        self.leads=leads

        #Only data of the same forecast hour is currently supported.
        assert dates[0].hour==dates[1].hour
        self.hour=dates[0].hour

        #Name of the primary time coordinate
        self.T="time"
        #The expected position of the primary time coordinate in the cube
        self.t=0

        #The day of year associated with 'dates'
        self.calendar_bounds=[d.timetuple().tm_yday for d in dates]
        
        self.type=Dataset
        
        #A dictionary that can contain any number of iris CubeLists, each
        #labelled with a keyword. The load_data method generates a "data" and
        #a "clim" CubeList
        
        self.data={}
        #Used by the get_climatology method
        self.dist_means=None
        self.distribution=None
        
        #The time unit to use
        self.U=cf_units.Unit(f"Days since {cf_units.EPOCH}",\
                             calendar=cf_units.CALENDAR_GREGORIAN)

        #Constraints applied to the data at different points.
        self.constraints={
        #keep only data with a valid time coordinate
        "load":iris.Constraint(cube_func=lambda cube: cube.coords(self.T)!=[]),
        
        #keep only data that falls within the calendar_bounds
        "calendar":iris.Constraint(coord_values={"day_of_year":lambda cell:\
                self._in_calendar_bounds(cell)}),
    
        #keep only data for the right hour
        "hour":iris.Constraint(hour=self.hour),
        
        #keep only data that falls within the dates
        "data":iris.Constraint(coord_values={self.T:lambda cell:\
                self._d_l<=cell<=self._d_u}),
        
        #keep only data that falls outside the dates
        "clim":iris.Constraint(coord_values={self.T:lambda cell:\
                (self._d_l>cell)or (cell>self._d_u)})
                }
        self._setup()
             
    
    def _setup(self):
        """empty method used by derived classes."""
        pass
    
    def set_path(self,path):
        """set the path from which to load data"""
        
        if os.path.isdir(path):
            self.path=path
        else:
            raise(ValueError("Not a valid path."))

    def copy(self):
        """A method which returns a copy of the Dataset"""
        
        copy=self.type(self.field,self.dates,self.leads)
        copy.dist_means=self.dist_means
        copy.distribution=self.distribution
        copy.data=cp.deepcopy(self.data)
        return copy
         
    def add_constraints(self,constr_dict):
        
        """add a dictionary of constraints 'constr_dict' to the constraints
        attribute. Any previously defined keywords will be overwritten."""
        
        for key in constr_dict:
            self.constraints[key]=constr_dict[key]

        
    def load_data(self,strict=True):
        
        """Load data from self.path as a list of iris cubes, preprocess it, 
        and split it into two CubeLists "data" and "clim".
        """
        
        self.data=iris.load([self.path+f for f in os.listdir(self.path) if f.endswith(".nc")],
                             constraints=self.constraints["load"])
        self._clean_loaded_data()
                
        a=self.data.extract(self.constraints["data"])
        c=self.data.extract(self.constraints["clim"])
        
        if strict:
            if a is None: raise(ValueError("No data after applying constraints."))
            if c is None: raise(ValueError("No climatology data after applying constraints."))

        self.data={"data":a,"clim":c}
        
    def _clean_loaded_data(self):
        """empty method used by derived classes."""

        pass

    def _in_calendar_bounds(self,x):
        
        """Evaluates whether a real number x lies between the calendar_bounds
        of the dataset, wrapping around the end of the year if necessary."""
        
        c0,c1=self.calendar_bounds
        if c1<c0:
            ans=(x<=c1) or (x>=c0)
        else:
            ans=(x<=c1) and (x>=c0)
        return ans     
           
    def restrict_area(self,region):
        
        """A convenience method that restricts the spatial extent of the 
        Dataset to one of a few preset domains, defined by a string "region".
        """
        
        if region.lower()=="europe":
            lons=[-15,20]
            lats=[32,60]
                
        elif region.lower()=="france":
            lons=[-5,8]
            lats=[42,51]
        
        else: raise(ValueError(f"Unrecognised region {region}."))
        
        #We use this over intersection, because it works for cubelists
        area_constr=iris.Constraint(longitude=lambda x: lons[0]<=x<=lons[1],\
                                    latitude=lambda x: lats[0]<=x<=lats[1])
        for key in self.data:
            self.data[key]=self.data[key].extract(area_constr)
       
    def add_cat_coord(self,iccat_function,coordname,base_coord):
        
        """Adds a categorical coordinate to all cubes in Dataset.data, defined
        by 'iccat_function' relative to 'base_coord', and called 'coordname'.
        
        Note that the name of the new coord is defined internally by
        iccat_function; coordname serves only to graciously handle the case when
        that coordinate already exists."""
        
        for key in self.data:
            for i,entry in enumerate(self.data[key]):
                if entry.coords(coordname)==[]:
                    iccat_function(entry,base_coord)
           
    def change_units(self,unit_str=None,cf_unit=None):
        
        """Changes the units of all cubes in the Dataset to a new unit given
        either by a valid cf_units.Unit string specifier 'unit_str', or a 
        cf_units.Unit object, 'cf_unit'."""
        
        if unit_str is not None and cf_unit is not None:
            raise(ValueError("Only one unit can be provided."))
        elif unit_str is not None:
            unit=cf_units.Unit(unit_str)
        elif cf_unit is not None:
            unit=cf_unit
        else: raise(ValueError("A unit must be provided."))
        
        for key in self.data:
            for i,entry in enumerate(self.data[key]):
                entry.convert_units(unit)
                
    def change_dates(self,newdates):
        """
        Redefines the 'dates' attribute to the list of 2 datetimes 'newdates',
        reapplying the "data" and "clim" constraints to match
        
        **currently quite slow for large cubelists**
        """
        self.dates=newdates
        self._d_l,self._d_u=self.dates
        self.calendar_bounds=[d.timetuple().tm_yday for d in self.dates]
        
        CL_data=iris.cube.CubeList()
        CL_clim=iris.cube.CubeList()
    
        for key in self.data:
            a=self.data[key].extract(self.constraints["data"])
            if a != []:
                CL_data.append(a)
            a=self.data[key].extract(self.constraints["clim"])
            if a != []:
                CL_clim.append(a)
        
        CL_data=iris.cube.CubeList([c for C in CL_data for c in C])
        CL_clim=iris.cube.CubeList([c for C in CL_clim for c in C])
        
        self.data["data"]=CL_data.concatenate()
        self.data["clim"]=CL_clim.concatenate()
                
    def aggregate_by(self,coords,bins,aggregator=iris.analysis.MEAN):
        
        """Aggregates the coordinates of all cubes in Dataset into user defined
        bins. 
        
        Args:
            *coords - A list of strings which are the coordinates
        to be aggregated over.
        
            *bins - A corresponding list of lists 'bins'. bins[i]
        should contain the bounding values over which to group coords[i].
        
        Kwargs:
            *aggregator -A valid iris.analysis.Aggregator object which specifies
            how to aggregate entries together.
        """
        
        binlabels=[]
        for j,coord in enumerate(coords):
            binlabels.append(f"bin{j}")

        for key in self.data:
            for i,entry in enumerate(self.data[key]):
                for j,(coord,b) in enumerate(zip(coords,bins)):
                    
                    
                    #remove potential old bins:
                    if self.data[key][i].coords(f"bin{j}")!=[]:
                        self.data[key][i].remove_coord(f"bin{j}")
                    
                    if self.data[key][i].coords(coord)==[]:
                        raise(ValueError("No such coordinate in cube!"))
                        
                    label=np.digitize(entry.coord(coord).points,b)
                    coord_dim=entry.coord_dims(entry.coord(coord))
                    entry.add_aux_coord(iris.coords.AuxCoord(label,\
                                       var_name=f"bin{j}"),data_dims=coord_dim)
                    
                self.data[key][i]=entry.aggregated_by(binlabels,aggregator)
                for j,coord in enumerate(coords):
                    if self.data[key][i].coords(coord)!=[]:
                        self.data[key][i].remove_coord(f"bin{j}")

    def collapse_over(self,coord,aggregator=iris.analysis.MEAN):
        
        """Collapses all cubes in Dataset over a single coordinate.
        
        Args:
            *coords - A string which is the coordinate to collapse.
        
        Kwargs:
            *aggregator -A valid iris.analysis.Aggregator object which specifies
            how to collapse the coordinate.
        """
        
        for key in self.data:
            for i,entry in enumerate(self.data[key]):
                self.data[key][i]=self.data[key][i].collapsed(coord,aggregator)

    def apply_coslat_mean(self,mask=None):
        
        """Collapses the latitude and longitude coordinates of all cubes in 
        Dataset, using a cosine latitude weighting.
        
        Kwargs:
            *mask:
                A cube with matching latitude and longitude coordinates to
                the cubes in Dataset. Each gridpoint in 'mask' should vary between
                0 (totally masked) to 1 (totally unmasked).
        """
        
        for key in self.data:
            for i,entry in enumerate(self.data[key]):

                weights = cosine_latitude_weights(entry)
        
                #include the land sea mask in the weighting if one was passed.
                if mask is not None:
                    assert mask.coord("latitude")==self.data[key][i].coord("latitude")
                    assert mask.coord("longitude")==self.data[key][i].coord("longitude")
    
                    weights=weights*mask.data
                    
                self.data[key][i]=entry.collapsed(["latitude","longitude"],\
                                  iris.analysis.MEAN,weights=weights) 

    def regrid_to(self,dataset=None,cube=None,regridder=iris.analysis.Linear()):
        """regrids every cube in Dataset to match either those of another
        Dataset object, or an iris.Cube object."""
        
        if cube is None and dataset is None:
            raise(ValueError("No reference for regridding provided!"))
        elif cube is None:
            ref_cube=dataset.data["data"][0]
        else:
            ref_cube=cube
            
        for key in self.data:
            for i,entry in enumerate(self.data[key]):
                self.data[key][i]=entry.regrid(ref_cube,regridder)

    
    def apply(self,func,*args,in_place=True,keys=None,**kwargs):
        """A method which applies a function to every cube in Dataset
        
        Args:
            *func - A function of the type func(cube,*args,**kwargs).
            
        Kwargs:
            in_place - A boolean, specifying whether func returns an output or 
            not. If True, cube is set equal to func(cube), unless the output
            is None, in which case cube is removed from the CubeList.
        """
        if keys is None:
            keys=self.data
            
        for key in keys:
            for i,entry in enumerate(self.data[key]):
                result=func(entry,*args,**kwargs)
                if in_place:
                    pass
                else:
                    if result is not None:
                        self.data[key][i]=result
                    else:
                        self.data[key].remove(self.data[key][i])

            
    def apply_constraint(self,constraint,keys=None):
        
        """Apply a constraint to all cubes in Dataset"""
        if keys is None:
            keys=self.data
            
        for key in keys:
            self.data[key]=self.data[key].extract(constraint)
            
    def get_climatology(self,percentiles):
        
        """Finds the distribution of all values in the Dataset. 
        
        Args:
            * percentiles - A numpy array ([p_1,...,p_N]) where 0<=p_i<=100,
            which defines the percentiles of the data distribution to calculate.
            
            """
        self.percentiles=percentiles
        
        lat,lon=self.data["clim"][0].shape[-2:]
        dist=np.zeros([1,lat,lon])
        
        #We call the whole cubelist into memory
        self.data["clim"].realise_data()
        dist=np.concatenate([f.data.reshape([-1,lat,lon]) for f in self.data["clim"]])
        
        self.distribution=np.percentile(dist,percentiles,axis=0)
        self.distribution[0]-=0.01
        
        means=np.zeros([len(percentiles)-1,lat,lon])
        for i in range(len(percentiles)-1):
            for j in range(lat):
                for k in range(lon):
                    means[i,j,k]=dist[np.digitize(dist[:,j,k],\
                          self.distribution[:,j,k],right=True)==i+1,j,k].mean()
        #interpolates empty bins as being halfway between the distribution bounds            
        for i,j,k in np.argwhere(np.isnan(means)):
            means[i,j,k]=self.distribution[i:i+2,j,k].mean()
        self.dist_means=means
        
    def get_seasonal_cycle(self,N,keys=None):
        
        """Fits N sine modes to the data series, with frequencies of n/(365.25 days)
        for n in [1,...,N], in order to calculate a smooth seasonal cycle.
        
        Kwargs:
            *keys - A list of keys to self.data, specifying which data to use
            to calculate the cycle. If keys is None, all data in the dataset
            will be used.
        """
class _Deseasonaliser:

    def __init__(self,data,keys,N,period=365.25,coeffs=None):
        self.raw_data=[]
        self.t=[]
        self.t_unit=None
        self.tref=None
        
        self.keys=keys
        self.N=N
        self.pnum=2*(N+1)
        self.period=period
        self.coeffs=None
        
        for key in keys:
            for cube in data[key]:
                self.raw_data.append(cube.data)
                
                if self.t_unit is not None:
                    if self.t_unit!=cube.coord("time").units:
                        raise(ValueError("Clashing time units in data."))
                else:
                    self.t_unit=cube.coord("time").units
                    
                self.t.append(cube.coord("time").points)
                
        self.raw_data=np.concatenate(self.raw_data,axis=0)    
        self.t=np.concatenate(self.t,axis=0)
        
        self._setup_data()
        self.lat,self.lon=self.raw_data.shape[1:]
        
    def _setup_data(self):
        
        self.raw_data=self.raw_data[np.argsort(self.t)]
        self.t.sort()
        self.tref=self.t[0]
        self.t=(self.t-self.tref)%self.period
        
    #intelligently guesses initial parameters
    def _guess_p(self,tstd):
        
        p=np.zeros(self.pnum)
        
        for i in range(0,self.N):
            p[2+2*i]=tstd/(i+1.0)
        return p        
    
    #defines multimode sine function for fitting
    def _evaluate_fit(self,x,p,N):
        ans=p[0]*x+p[1]
        for i in range(0,N):
            ans+=p[2*i+2] * np.sin(2 * np.pi * (i+1)/365.25 * x + p[2*i+3])
        return ans
    
    #defines error function for optimisation
    def _get_residual(self,p,y,x,N):
        return y - self._evaluate_fit(x,p,N)
    
    def fit_cycle(self):
        from scipy.optimize import leastsq
        fit_coeffs=np.zeros([self.pnum,self.lat,self.lon])
        
        for i in range(self.lat):
            for j in range(self.lon):
                
                griddata=self.raw_data[:,i,j]
                tstd=griddata.std()
                p0=self._guess_p(tstd)
                
                plsq=leastsq(self._get_residual,p0,args=(griddata,self.t,self.N))
                fit_coeffs[:,i,j]=plsq[0]
        self.coeffs=fit_coeffs
        
    def evaluate_cycle(self,t):
        
        if self.coeffs is None:
            raise(ValueError("No coefficients for fitting have been calculated yet."))
            
        if t.units!=self.t_unit:
            if t.units.is_convertible(self.t_unit):
                t=t.convert_units(self.t_unit)
            else:
                raise(ValueError("Units of time series to evaluate are incompatible\
                                 with units of fitted time series."))
        t=t.points
        t=(t-self.tref)%self.period
        
        cycle=np.zeros([len(t),self.lat,self.lon])
        for i in range(self.lat):
            for j in range(self.lon):
                cycle[:,i,j]=self._evaluate_fit(t,self.coeffs[:,i,j],self.N)
                
        return cycle

            
        
"""Analysis is a subclass of Dataset that deals with reanalysis. At the moment
specific to era5, but that should be changed if more analyses start being used."""
class Analysis(Dataset):
    
    def _setup(self):
        
        self.path="/mnt/seasonal/reanalysis/era5/"+self.field+"/"
        self.type=Analysis

    def _clean_loaded_data(self):
        

        for i in range(len(self.data)):
            self.data[i].metadata.attributes.clear()
            self.data[i].coord("latitude").points=\
                self.data[i].coord("latitude").points.astype(np.float32)
            self.data[i].coord("longitude").points=\
                self.data[i].coord("longitude").points.astype(np.float32)
            
        self.data=self.data.concatenate_cube()
        self.data.coord(self.T).convert_units(self.U)
        
        iccat.add_hour(self.data,self.T)
        self.data=self.data.extract(self.constraints["hour"])

        iccat.add_day_of_year(self.data,self.T)
        self.data=self.data.extract(self.constraints["calendar"])
        self.data=iris.cube.CubeList([self.data])

class Forecast(Dataset):
    
    def _setup(self):
        
        self.T="forecast_reference_time"
        self.S="forecast_period"
        self.R="realisation"
        self._l_l,self._l_u=self.leads
        self.type=Forecast
        self.t=1
        self._fsetup()
        self.constraints["lead"]=iris.Constraint(coord_values={self.S:\
                        lambda cell:(self._l_l<=cell)and (cell<=self._l_u)})
            
        self.constraints["ens"]=iris.Constraint(coord_values={self.R:\
                        lambda cell: cell.point<self.max_ens})
        
    #Used by derived classes
    def _fsetup(self):
        pass
        
    def get_quantile_correction(self,analysis):
        if self.dist_means is None:
            raise(ValueError("Must get forecast climatology first."))
        if analysis.dist_means is None:
            raise(ValueError("Must get analysis climatology first."))
        if not np.all(analysis.percentiles == self.percentiles):
            raise(ValueError("These datasets have incomparable climatologies."))

        self.quantile_correction=analysis.dist_means-self.dist_means
    
    def apply_quantile_correction(self):
        
        lat,lon=self.data["data"][0].shape[-2:]
        
        for i,entry in enumerate(self.data["data"]):
            shape=entry.data.shape
            data=entry.data.reshape([-1,lat,lon])
            for x in range(lat):
                for y in range(lon):
                    which_bin=np.digitize(data[:,x,y],self.distribution[:,x,y],right=True)
                    which_bin[which_bin==0]+=1 #cold outliers put in 0-5% bin
                    which_bin[which_bin==len(self.percentiles)]-=1 #warm outliers in 95-100% bin
                    which_bin-=1 #indexing from zero
                    correction=self.quantile_correction[:,x,y][which_bin]
                    data[:,x,y]+=correction
            data=data.reshape(shape)
            self.data["data"][i].data=data
            self.data["data"][i].long_name="corrected "+self.data["data"][i].name()
            
class SubxForecast(Forecast):
    
    def _fsetup(self):
        self.path="/mnt/seasonal/subx/"+self.field+"/"
        self.R="realization"
        self.max_ens=11
        self.type=SubxForecast

    def _clean_loaded_data(self):
        
        CL=iris.cube.CubeList()
        for i,cube in enumerate(self.data):
            for entry in cube.slices_over(self.T):
                
                entry.coord(self.T).convert_units(self.U)
                
                T_ref=entry.coord(self.T)
                S=entry.coord(self.S).points
                t_coord=iris.coords.AuxCoord(S+T_ref.points[0],standard_name="time")
                t_coord.units=T_ref.units
                entry.add_aux_coord(t_coord,data_dims=1)
                
                iccat.add_hour(entry,"time")
                iccat.add_day_of_year(entry,"time")

                CL.append(entry)
        CL.sort(key=lambda cube:cube.coord(self.T).points[0])            
        self.data=CL
        
        self.data=self.data.extract(self.constraints["calendar"])
        self.data=self.data.extract(self.constraints["lead"])
        self.data=self.data.extract(self.constraints["hour"])
        self.data=self.data.extract(self.constraints["ens"])

    def remove_masked(self):
        for key in self.data:
            self.data[key].realise_data()   
            masked=[]
            
            for entry in self.data[key]:
                if not np.all(entry.data.mask==False):
                    masked.append(entry)
                    
            for entry in masked:
                self.data[key].remove(entry)
                
class EC45Forecast(Forecast):
    
    def _fsetup(self):
        self.path="/mnt/seasonal/ec45/netcdf/"+self.field+"/"
        self.max_ens=11
        self.U=cf_units.Unit(f"Days since {cf_units.EPOCH}",\
                         calendar=cf_units.CALENDAR_PROLEPTIC_GREGORIAN)
        self.type=EC45Forecast

    def _clean_loaded_data(self):
        
        CL=iris.cube.CubeList()
        for i,cube in enumerate(self.data):
            for entry in cube.slices_over(self.T):
                
                entry.coord(self.T).convert_units(self.U)
                entry.coord(self.S).convert_units(cf_units.Unit("Days"))
                T_ref=entry.coord(self.T)
                S=entry.coord(self.S).points
                t_coord=iris.coords.AuxCoord(S+T_ref.points[0],standard_name="time")
                t_coord.units=T_ref.units
                entry.add_aux_coord(t_coord,data_dims=1)
                
                iccat.add_hour(entry,"time")
                iccat.add_day_of_year(entry,"time")

                CL.append(entry)
        CL.sort(key=lambda cube:cube.coord(self.T).points[0])            
        self.data=CL
        
        self.data=self.data.extract(self.constraints["calendar"])
        self.data=self.data.extract(self.constraints["lead"])
        self.data=self.data.extract(self.constraints["hour"])
        self.data=self.data.extract(self.constraints["ens"])

class Seas5Forecast(Forecast):
    
    def _fsetup(self):
        self.path="/mnt/seasonal/seas5/"+self.field+"/"
        self.max_ens=25
        self.R="realization"
        self.U=cf_units.Unit(f"Days since {cf_units.EPOCH}",\
                calendar=cf_units.CALENDAR_PROLEPTIC_GREGORIAN)
        self.type=Seas5Forecast

        
    def _clean_loaded_data(self):
        
        CL=iris.cube.CubeList()
        for i,cube in enumerate(self.data):
            for entry in cube.slices_over(self.T):
                
                entry.coord(self.T).convert_units(self.U)
                entry.coord(self.S).convert_units(cf_units.Unit("Days"))

                T_ref=entry.coord(self.T)
                S=entry.coord(self.S).points
                t_coord=iris.coords.AuxCoord(S+T_ref.points[0],standard_name="time")
                t_coord.units=T_ref.units
                entry.add_aux_coord(t_coord,data_dims=1)
                
                iccat.add_hour(entry,"time")
                iccat.add_day_of_year(entry,"time")

                CL.append(entry)
        CL.sort(key=lambda cube:cube.coord(self.T).points[0])            
        self.data=CL
        
        self.data=self.data.extract(self.constraints["calendar"])
        self.data=self.data.extract(self.constraints["lead"])
        self.data=self.data.extract(self.constraints["hour"])
        self.data=self.data.extract(self.constraints["ens"])
    
"""An example script:
Here we want to look at week 3 forecasts around second week of June 2003. 
We use a 21 day window, including the first and third weeks. We want
weekly mean temperatures averaged over France, for the EC45, SUBX and SEAS5
forecast systems. We also want to debias our forecasts, using a climatology
of past dates.
"""        

#All bounds are inclusive.
#3 week period centred around 11/6/2003
dates=[dt.datetime(2003,6,1,12),dt.datetime(2003,6,21,12)]        
leads=[14.5,20.5] #Week 3, in days. We want midday, so we add .5
run=False
if run:
    #Load and restrict to the region around France
    import time 
    t0=time.time()
    
    A=Analysis("T2m",dates)
    A.load_data()
    A.restrict_area("France")
    
    Fx=SubxForecast("T2m",dates,leads)
    Fx.load_data()
    Fx.restrict_area("France")
    
    Fec=EC45Forecast("2T",dates,leads)
    Fec.load_data()
    Fec.restrict_area("France")
    
    Fs=Seas5Forecast("T2m",dates,leads)
    Fs.load_data()
    Fs.restrict_area("France")
    
    t1=time.time()
    print(f"loaded data (t={t1-t0:.1f}).")
    #Fx has the lowest spatial resolution
    A.regrid_to(Fx)
    Fec.regrid_to(Fx)
    Fs.regrid_to(Fx)
    
    t2=time.time()
    print(f"regridded data (t={t2-t1:.1f}).")

    #Backups of the uncorrected forecasts so we dont have to reload.
    Fx_bkp=Fx.copy()
    Fs_bkp=Fs.copy()
    Fec_bkp=Fec.copy()
    
    #Compute 5% bins for climatology calculations
    #We want our climatology to be computed for daily, gridpoint values:
    percentiles=np.linspace(0,100,21)
    A.get_climatology(percentiles)
    Fx.get_climatology(percentiles)
    Fs.get_climatology(percentiles)
    Fec.get_climatology(percentiles)
    
    Fx.get_quantile_correction(A)
    Fs.get_quantile_correction(A)
    Fec.get_quantile_correction(A)
    
    t3=time.time()
    print(f"computed corrections (t={t3-t2:.1f}).")

    Fx.apply_quantile_correction()
    Fs.apply_quantile_correction()
    Fec.apply_quantile_correction()
    
    t4=time.time()
    print(f"applied corrections (t={t4-t3:.1f}).")

    #After error correcting we want to take weekly means. We exclude any
    #forecasts that aren't 7 days long:
    full_week=iris.Constraint(cube_func=lambda cube: cube.coord("forecast_period").shape[0]==7)
    Fx.apply_constraint(full_week)
    Fs.apply_constraint(full_week)
    Fec.apply_constraint(full_week)
    
    #We then collapse the time axis to get weekly means:
    A.apply(lambda cube: cube.collapsed(A.T,iris.analysis.MEAN))
    Fx.apply(lambda cube: cube.collapsed(Fx.S,iris.analysis.MEAN))
    Fs.apply(lambda cube: cube.collapsed(Fs.S,iris.analysis.MEAN))
    Fec.apply(lambda cube: cube.collapsed(Fec.S,iris.analysis.MEAN))

    #We load the land/sea mask and apply the area reduction:
    MASK_PATH="/mnt/seasonal/land_sea_mask/NAVO_lsmask_1deg.nc"
    sea_mask=iris.load_cube(MASK_PATH)
    sea_mask=sea_mask.regrid(Fx.data["data"][0],iris.analysis.Linear())
    
    A_reduced=A.copy().apply_coslat_mean(mask=sea_mask)
    Fx.apply_coslat_mean(mask=sea_mask)
    Fs.apply_coslat_mean(mask=sea_mask)
    Fec.apply_coslat_mean(mask=sea_mask)
    
    t5=time.time()
    print(f"collapsed lat and lon (t={t5-t4:.1f}).")
    
    print("finished!")


