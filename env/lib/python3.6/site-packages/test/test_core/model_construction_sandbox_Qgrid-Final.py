#!/usr/bin/env python
# coding: utf-8

# ## Importing used libraries

# In[1]:


# These three lines are necessary only if GemPy is not installed
import sys, os
sys.path.append('../..')
sys.path.append('../../../gempy/')

# Importing GemPy
import gempy as gp

# Importing aux libraries
from ipywidgets import interact
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pn
import matplotlib
import theano
import qgrid
import pickle

# Imort SandBox
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN"
#import sandbox.sandbox as sb


# In[2]:


# calibrationdata = sb.CalibrationData(file='my_calibration.json')
# kinect = sb.KinectV2(calibrationdata)
# projector = sb.Projector(calibrationdata)


# ### Initializing the model:
# 
# The first step to create a GemPy model is create a gempy.Model object that will contain all the other data structures and necessary functionality.
# 
# In addition for this example we will define a regular grid since the beginning. This is the grid where we will interpolate the 3D geological model. GemPy comes with an array of different grids for different pourposes as we will see below. For visualization usually a regular grid is the one that makes more sense.

# In[3]:


geo_model = gp.create_model('Geological_Model1')
geo_model = gp.init_data(geo_model, extent= [0, 4000, 0, 2775, 200, 1200], resolution=[100, 10, 100])


# In[4]:


geo_model.rename_series(['Cycle1'])
geo_model.add_series(['Fault2'])
geo_model.add_series(['Fault1'])
geo_model.add_series(['Cycle2'])
geo_model.reorder_series(['Fault2','Cycle2','Fault1','Cycle1'])


# In[5]:


geo_model.add_surfaces(['D','C','B', 'A', 'F1', 'F2', 'H', 'G'])


# In[6]:


gp.map_series_to_surfaces(geo_model, {'Cycle2':['G','H'], 'Fault1':'F1', 'Fault2': 'F2'})


# In[7]:


geo_model.set_is_fault(['Fault1', 'Fault2'])


# GemPy core code is written in Python. However for efficiency (and other reasons) most of heavy computations happend in optimize compile code, either C or CUDA for GPU. To do so, GemPy rely on the library theano. To guarantee maximum optimization theano requires to compile the code for every Python kernel. The compilation is done by calling the following line at any point (before computing the model):

# In[8]:


gp.set_interpolator(geo_model, theano_optimizer='fast_compile', update_kriging=True,
                    verbose=[])


# ### Creating figure:
# 
# GemPy uses matplotlib and pyvista-vtk libraries for 2d and 3d visualization of the model respectively. One of the design decisions of GemPy is to allow real time construction of the model. What this means is that you can start adding input data and see in real time how the 3D surfaces evolve. Lets initialize the visualization windows.
# 
# The first one is the 2d figure. Just place the window where you can see it (maybe move the jupyter notebook to half screen and use the other half for the renderers).

# In[9]:


#Visualization Widgets - Conflicts with bokeh visualization
from gempy.plot import visualization_2d_pro as vv
from gempy.plot import vista


# In[10]:


get_ipython().run_line_magic('matplotlib', 'qt5')

p2d = vv.Plot2D(geo_model)
p2d.create_figure((15, 8))


# #### Add model section
# 
# In the 2d renderer we can add several cross section of the model. In this case, for simplicity we are just adding one perpendicular to z.

# In[11]:


# In this case perpendicular to the z axes
ax = p2d.add_section(direction='z', ax_pos=121)
#ax.imshow()


# In[12]:


ax2 = p2d.add_section(direction='y', ax_pos=122)
ax2.set_xlim(geo_model.grid.regular_grid.extent[0], geo_model.grid.regular_grid.extent[1])
ax2.set_ylim(geo_model.grid.regular_grid.extent[4], geo_model.grid.regular_grid.extent[5])


# #### Loading geological map image:
# 
# Remember that gempy is simply using matplotlib and therofe the ax object created above is a standard matplotlib axes. This allow to manipulate it freely. Lets load an image with the information of geological map

# In[13]:


# Reading image
img = mpimg.imread('geological_model.png')
# Plotting it inplace
ax.imshow(img, origin='upper', alpha=.8, extent = (0, 4000, 0,2775))


# In[14]:


p3d = vista.Vista(geo_model, plotter_type='background', notebook=False, real_time=False)


# ## Building the model
# 
# Now that we have everything initialize we can start the construction of the geological model. 
# ### Cycle1:
# 
# #### Surfaces
# 
# GemPy is a surface based interpolator. This means that all the input data we add has to be refered to a surface. The surfaces always mark the bottom of a unit. By default GemPy surfaces are empty:

# In[15]:


geo_model._surfaces


# We can create the first surfaces for the Cycle1 of the sedimentary layers:

# Series is the object that contains the properties associated with each independent scalar field. The name by default is "Default series" but we can rename it and create new ones as we advance in the constructin of the model

# Now we can start adding data. GemPy input data consist on surface points and orientations (perpendicular to the layers). The 2D plot gives you the X and Y coordinates when hovering the mouse over. We can add a surface point as follows:

# In[16]:


#surface B
geo_model.add_surface_points(X=584, Y=285, Z=500, surface='B')
geo_model.add_surface_points(X=494, Y=696, Z=500, surface='B')
geo_model.add_surface_points(X=197, Y=1898, Z=500, surface='B')
geo_model.add_surface_points(X=473, Y=2180, Z=400, surface='B')
geo_model.add_surface_points(X=435, Y=2453, Z=400, surface='B')
#surface C
geo_model.add_surface_points(X=946, Y=188, Z=600, surface='C')
geo_model.add_surface_points(X=853, Y=661, Z=600, surface='C')
geo_model.add_surface_points(X=570, Y=1845, Z=600, surface='C')
geo_model.add_surface_points(X=832, Y=2132, Z=500, surface='C')
geo_model.add_surface_points(X=767, Y=2495, Z=500, surface='C')
#Surface D
geo_model.add_surface_points(X=967, Y=1638, Z=800, surface='D')
geo_model.add_surface_points(X=1095, Y=996, Z=800, surface='D')


# The minimum amount of data to interpolate anything in gempy is
# 
# a) 2 surface points per surface
# b) One orientation per series.
# 
# Lets add an orientation for the first cycle:

# In[17]:


# Adding orientation
geo_model.add_orientations(X=832, Y=2132, Z=500, surface='C', orientation = [98,17.88,1])


# In[18]:


geo_model._series


# In[19]:


# Plot in 2D
p2d.plot_data(ax, direction='z')
p2d.plot_data(ax2, direction='y', )


# In[20]:


geo_model._series


# In[21]:


gp.compute_model(geo_model)


# In[22]:


geo_model._series


# In[23]:


# In 2D
p2d.plot_contacts(ax, direction='z', cell_number=-10)
p2d.plot_lith(ax, direction='z', cell_number=-10)


p2d.plot_contacts(ax2, direction='y', cell_number=5)
p2d.plot_lith(ax2, direction='y', cell_number=5)
# In 3D
#p3d.plot_surfaces()


# In[24]:


p3d.plot_data()


# In[25]:


p3d.plot_surfaces()


# In[26]:


geo_model._additional_data


# ### Fault1 :
# So far the model is simply a depositional unit. GemPy allows for unconformities and faults to build complex models. This input is given by categorical data. In general:
# 
# input data (surface points/ orientations) <belong to< surface <belong to< series
# 
# And series can be a fault---i.e. offset the rest of surface--- or not. We are going to show how to add a fault.
# 
# First we need to add a series:

# Then define that is a fault:

# But we also need to add a new surface:

# And finally assign the new surface to the new series/fault

# Now we can just add input data as before (remember the minimum amount of input data to compute a model):

# In[27]:


# Add input data of the fault
geo_model.add_surface_points(X=1203, Y=138, Z=600, surface='F1')
geo_model.add_surface_points(X=1250, Y=1653, Z=800, surface='F1')
#geo_model.add_surface_points(X=1280, Y=2525, Z=500, surface='F1')
#Add orientation
geo_model.add_orientations(X=1280, Y=2525, Z=500, surface='F1', orientation = [272,90,-1])


# In[28]:


geo_model._surfaces


# In[29]:


# Compute
gp.compute_model(geo_model)


# In[30]:


p2d.remove(ax)
p2d.plot_data(ax, direction='z', cell_number=-10)
p2d.plot_contacts(ax, direction='z', cell_number=-10)
p2d.plot_lith(ax, direction='z', cell_number=-10)

p2d.remove(ax2)
# Plot
p2d.plot_data(ax2, cell_number=5)
p2d.plot_lith(ax2, cell_number=5)
p2d.plot_contacts(ax2,cell_number=5)
#p3d.plot_structured_grid(opacity=.2, annotations = {2: 'surface1', 3:'surface2', 4:'surface3', 5:'basement'})


# In[31]:


geo_model._additional_data


# In[31]:


p3d.plot_surfaces()


# As you can see now instead of having dipping layers we have a sharp jump. But there is no information on the other side of the fault. That is because we now are going to add the information on the afected block. 

# In[32]:


#surface B
geo_model.add_surface_points(X=1447, Y=2554, Z=500, surface='B')
geo_model.add_surface_points(X=1511, Y=2200, Z=500, surface='B')
geo_model.add_surface_points(X=1549, Y=629, Z=600, surface='B')
geo_model.add_surface_points(X=1630, Y=287, Z=600, surface='B')
#surface C
geo_model.add_surface_points(X=1891, Y=2063, Z=600, surface='C')
geo_model.add_surface_points(X=1605, Y=1846, Z=700, surface='C')
geo_model.add_surface_points(X=1306, Y=1641, Z=800, surface='C')
geo_model.add_surface_points(X=1476, Y=979, Z=800, surface='C')
geo_model.add_surface_points(X=1839, Y=962, Z=700, surface='C')
geo_model.add_surface_points(X=2185, Y=893, Z=600, surface='C')
geo_model.add_surface_points(X=2245, Y=547, Z=600, surface='C')
#Surface D
geo_model.add_surface_points(X=2809, Y=2567, Z=600, surface='D')
geo_model.add_surface_points(X=2843, Y=2448, Z=600, surface='D')
geo_model.add_surface_points(X=2873, Y=876, Z=700, surface='D')


# In[33]:


# Compute
gp.compute_model(geo_model)


# In[34]:


# Plot
p2d.remove(ax)
p2d.plot_data(ax, direction='z', cell_number=-10)
p2d.plot_contacts(ax, direction='z', cell_number=-10)
p2d.plot_lith(ax, direction='z', cell_number=-10)

p2d.remove(ax2)
p2d.plot_lith(ax2, cell_number=5)
p2d.plot_data(ax2, cell_number=5)


# In[35]:


p3d.update_surfaces()


# In[36]:


p3d.plot_data()
p3d.plot_surfaces()


# Now all the first sequence is complete, with the deposition of some sedimentary layes and its posterior faulting

# ### Cycle2 :
# For the second cycle, we have the disconformity of layer G on top of the old layers and the fault. This order is important to take into account for the modelling.
# First we create a new serie with tis cycle:

# In[37]:


# Surface G
geo_model.add_surface_points(X=1012, Y=1493, Z=900, surface='G')
geo_model.add_surface_points(X=1002, Y=1224, Z=900, surface='G')
geo_model.add_surface_points(X=1996, Y=47, Z=800, surface='G')
geo_model.add_surface_points(X=300, Y=907, Z=700, surface='G')
#Surface H
geo_model.add_surface_points(X=3053, Y=727, Z=800, surface='G')
#Orientation
geo_model.add_orientations(X=1996, Y=47, Z=800, surface='G', orientation = [272,5.54,1])


# In[38]:


# Compute
gp.compute_model(geo_model)


# In[39]:


# Plot
# Plot
p2d.remove(ax)
p2d.plot_data(ax, direction='z', cell_number=-10)
p2d.plot_contacts(ax, direction='z', cell_number=-10)
p2d.plot_lith(ax, direction='z', cell_number=-10)

p2d.remove(ax2)
p2d.plot_lith(ax2, cell_number=5)
p2d.plot_data(ax2, cell_number=5)
p2d.plot_contacts(ax2, cell_number=5)


# In[40]:


p3d.plot_surfaces()
#p3d.plot_structured_grid()


# In[41]:


p3d.plot_data()


# ### Fault2: 
# The model includes 2 different faultings and different ages. For this we are going to do the same steps as before to input the data:

# In[42]:


geo_model.add_surface_points(X=3232, Y=178, Z=1000, surface='F2')
geo_model.add_surface_points(X=3132, Y=951, Z=700, surface='F2')
#geo_model.add_surface_points(X=2962, Y=2184, Z=700, surface='F2')

geo_model.add_orientations(X=3132, Y=951, Z=700, surface='F2', orientation = [95,90,1])


# In[43]:


# Compute
gp.compute_model(geo_model)


# In[44]:


geo_model._surfaces


# In[45]:


# Plot
p2d.remove(ax)
p2d.plot_data(ax, direction='z', cell_number=5, legend='force')
p2d.plot_lith(ax, direction='z', cell_number=5)
p2d.plot_contacts(ax, direction='z',cell_number=5)

p2d.remove(ax2)
p2d.plot_lith(ax2, cell_number=5)
p2d.plot_data(ax2, cell_number=5)
p2d.plot_contacts(ax2, cell_number=5)


# In[46]:


p3d.plot_surfaces()


# finally, we will add the information of the right side of fault2 to complete the model

# In[47]:


geo_model.add_surface_points(X=3135, Y=1300, Z=700, surface='D')
geo_model.add_surface_points(X=3190, Y=969, Z=700, surface='D')

geo_model.add_surface_points(X=3031, Y=2725, Z=800, surface='G')
geo_model.add_surface_points(X=3018, Y=1990, Z=800, surface='G')
geo_model.add_surface_points(X=3194, Y=965, Z=700, surface='G')

geo_model.add_surface_points(X=3218, Y=1818, Z=890, surface='H')
geo_model.add_surface_points(X=3934, Y=1207, Z=810, surface='H')


# In[48]:


# Compute
gp.compute_model(geo_model)


# In[49]:


# Plot
p2d.remove(ax)
p2d.plot_data(ax, direction='z', cell_number=5, legend='force')
p2d.plot_lith(ax, direction='z', cell_number=5)
p2d.plot_contacts(ax, direction='z',cell_number=5)

p2d.remove(ax2)
p2d.plot_lith(ax2, cell_number=5)
p2d.plot_data(ax2, cell_number=5)
p2d.plot_contacts(ax2, cell_number=5)


# In[50]:


p3d.plot_surfaces()


# ## Iteractive DataFrame

# ### Activating Qgrid
# 
# Qgrid is only a gempy dependency. Therefore to use it, first we need to activate it in a given model by using:

# In[ ]:


gp.activate_interactive_df(geo_model)


# This will create the interactive dataframes objects. This dataframes are tightly linked to the main dataframes of each data class.

# #### Series

# In[ ]:


geo_model.qi.qgrid_se


# #### Faults

# In[ ]:


geo_model.qi.qgrid_fa


# #### surfaces

# In[ ]:


geo_model.qi.qgrid_fo


# ### surface points

# In[ ]:


geo_model.qi.qgrid_in


# ### Orientations

# In[ ]:


geo_model.qi.qgrid_or


# Remember we are always changing the main df as well!

# ### Plot

# In[ ]:


# Compute
gp.compute_model(geo_model)

# Plot
p2d.plot_lith(ax, cell_number=5)
p2d.plot_contacts(ax, cell_number=5)
p3d.plot_surfaces()


# ### Sandbox ??

# In[ ]:


gpsb=sb.GemPyModule(geo_model, calibrationdata, kinect, projector)


# In[ ]:


gpsb.setup()


# In[ ]:


gpsb.run()


# In[ ]:


gpsb.stop() 


# In[ ]:


fig=gp.plot.plot_section(geo_model,90, direction ='x')

fig.fig.axes[0].set_xlim(800,0)


# In[ ]:




