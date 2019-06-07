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

        
"""
Dataset is the base object shared by all analysis and forecast data sets. It defines
all functions that are generic between datasets"""
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
        
        self.field=field
        
        assert dates[0].hour==dates[1].hour
        self.T="time"
        self.dates=dates
        self.hour=dates[0].hour
        self._d_l,self._d_u=dates
        self.calendar_bounds=[d.timetuple().tm_yday for d in dates]
        self.leads=leads
        self.type=Dataset
        self.dist_means=None
        self.distribution=None
        self.t=0
        self.U=cf_units.Unit(f"Days since {cf_units.EPOCH}",\
                             calendar=cf_units.CALENDAR_GREGORIAN)
        
        self.constraints={
        "load":iris.Constraint(cube_func=lambda cube: cube.coords(self.T)!=[]),
        
        "calendar":iris.Constraint(coord_values={"day_of_year":lambda cell:\
                self._in_calendar_bounds(cell)}),
    
        "hour":iris.Constraint(hour=self.hour),
        
        "data":iris.Constraint(coord_values={self.T:lambda cell:\
                self._d_l<=cell<=self._d_u}),
    
        "clim":iris.Constraint(coord_values={self.T:lambda cell:\
                (self._d_l>cell)or (cell>self._d_u)})
                }
        self._setup()
        
    #used by derived classes
    def _setup(self):
        pass
    
    def set_path(self,path):
        if os.path.isdir(path):
            self.path=path
        else:
            raise(ValueError("Not a valid path."))

         
    def add_constraints(self,constr_dict):
        for key in constr_dict:
            self.constraints[key]=constr_dict[key]

        
    def load_data(self):
        
        self.data=iris.load([self.path+f for f in os.listdir(self.path) if f.endswith(".nc")],
                             constraints=self.constraints["load"])
        self._clean_loaded_data()
                
        a=self.data.extract(self.constraints["data"])
        c=self.data.extract(self.constraints["clim"])
        
        if a is None: raise(ValueError("No data after applying constraints."))
        if c is None: raise(ValueError("No climatology data after applying constraints."))

        self.data={"data":a,"clim":c}
        
    def _clean_loaded_data(self):
        pass

    #evaluates whether x is within the calendar bounds
    def _in_calendar_bounds(self,x):
        c0,c1=self.calendar_bounds
        if c1<c0:
            ans=(x<=c1) or (x>=c0)
        else:
            ans=(x<=c1) and (x>=c0)
        return ans     
           
    def restrict_area(self,key):
        
        if key.lower()=="europe":
            lons=[-15,20]
            lats=[32,60]
                
        elif key.lower()=="france":
            lons=[-5,8]
            lats=[42,51]
        
        else: raise(ValueError(f"Unrecognised key {key}."))
        
        #We use this over intersection, because it works for cubelists
        area_constr=iris.Constraint(longitude=lambda x: lons[0]<=x<=lons[1],\
                                    latitude=lambda x: lats[0]<=x<=lats[1])
        for key in self.data:
            self.data[key]=self.data[key].extract(area_constr)
       
    def add_cat_coord(self,iccat_function,coordname,base_coord):
        for key in self.data:
            for i,entry in enumerate(self.data[key]):
                if entry.coords(coordname)==[]:
                    iccat_function(entry,base_coord)
                    
    def aggregate_by(self,coords,bins,aggregator=iris.analysis.MEAN):
        
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
                print(binlabels)
                print(entry.coords(*binlabels))
                self.data[key][i]=entry.aggregated_by(binlabels,aggregator)
                for j,coord in enumerate(coords):
                    if self.data[key][i].coords(coord)!=[]:
                        self.data[key][i].remove_coord(f"bin{j}")


    def apply_coslat_mean(self,mask=None):
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

    def regrid_to(self,dataset=None,cube=None):
        
        if cube is None and dataset is None:
            raise(ValueError("No reference for regridding provided!"))
        elif cube is None:
            ref_cube=dataset.data["data"][0]
        else:
            ref_cube=cube
            
        for key in self.data:
            for i,entry in enumerate(self.data[key]):
                self.data[key][i]=entry.regrid(ref_cube,iris.analysis.Linear())
                
    def copy(self):
        copy=self.type(self.field,self.dates,self.leads)
        
        copy.data=cp.deepcopy(self.data)
        return copy
    
    def apply(self,func,*args,in_place=True,**kwargs):
        
        for key in self.data:
            for i,entry in enumerate(self.data[key]):
                result=func(entry,*args,**kwargs)
                
                if result is not None:
                    self.data[key][i]=func(entry,*args,**kwargs)
                    
                elif in_place:
                    self.data[key].remove(self.data[key][i])
                else:
                    pass
            
    def apply_constraint(self,constraint):
        for key in self.data:
            self.data[key]=self.data[key].extract(constraint)
            
    #A method that finds the gridpointwise distribution of the dataset.
    #self.distribution 
    def get_climatology(self,percentiles):
        
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
        self.dist_means=means
        
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
        
        lat,lon=self.data["clim"][0].shape[-2:]
        
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


                
class EC45Forecast(Forecast):
    
    def _fsetup(self):
        self.path="/mnt/seasonal/ec45/netcdf/"+self.field+"/"
        self.max_ens=10
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
