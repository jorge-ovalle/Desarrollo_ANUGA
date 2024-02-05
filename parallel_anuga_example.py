import numpy as np
import matplotlib.pyplot as plt
import os
import anuga




if anuga.myid == 0:
    print('I am processor ', anuga.myid, ' of ', anuga.numprocs)
    # Parameters
    DATA_FOLDER = 'data_notebook2'

    # Polygon defining broad area of interest
    bounding_polygon = anuga.read_polygon(os.path.join(DATA_FOLDER,'extent.csv'))


    # Polygon defining particular area of interest
    merewether_polygon = anuga.read_polygon(os.path.join(DATA_FOLDER,'merewether.csv'))


    # Elevation Data
    topography_file = os.path.join(DATA_FOLDER,'topography1.asc')


    # Resolution for most of the mesh
    base_resolution = 80.0  # m^2

    # Resolution in particular area of interest 
    merewether_resolution = 25.0 # m^2

    interior_regions = [[merewether_polygon, merewether_resolution]]

    # CREATE AND VIEW DOMAIN
    domain = anuga.create_domain_from_regions(
            bounding_polygon,
            boundary_tags={'bottom': [0],
                            'right':  [1],
                            'top':    [2],
                            'left':   [3]},
            maximum_triangle_area=base_resolution,
            interior_regions=interior_regions)


    domain.set_name('merewether1') # Name of sww file

    # plt.triplot(dplotter.triang, linewidth = 0.4)

    # plt.tripcolor(dplotter.triang, 
    #           facecolors = dplotter.elev, 
    #           cmap='Greys_r')
    # plt.colorbar()
    # plt.title("Elevation")

else:
    domain = None
    print('I am processor ', anuga.myid, ' of ', anuga.numprocs)


domain = anuga.distribute(domain)

# INITIAL_CONDITIONS
domain.set_quantity('elevation', filename=topography_file, location='centroids') # Use function for elevation
domain.set_quantity('friction', 0.025, location='centroids')                        # Constant friction 
domain.set_quantity('stage', expression='elevation', location='centroids')         # Dry Bed 

# BOUNDARY CONDITIONS
Br = anuga.Reflective_boundary(domain)
Bt = anuga.Transmissive_boundary(domain)

domain.set_boundary({'bottom':   Br,
                    'right':    Bt, # outflow
                    'top':      Bt, # outflow
                    'left':     Br})

# INFLOW
center = (382270.0, 6354285.0)
radius = 10.0
region0 = anuga.Region(domain, center=center, radius=radius)
fixed_inflow = anuga.Inlet_operator(domain, region0 , Q=0.7)

anuga.barrier()
# EVOLVE THE DOMAIN
for t in domain.evolve(yieldstep=400, duration=2000):

    #dplotter.plot_depth_frame()
    # dplotter.save_depth_frame(vmin=0.0, vmax=1.0)
    if anuga.myid == 0:
        domain.print_timestepping_statistics()


# Read in the png files stored during the evolve loop
# dplotter.make_depth_animation() 

anuga.barrier()

domain.sww_merge(delete_old=True)
anuga.finalize()