# gastrometry
astrometry residual analysis of HSC

# example run:

make some config file config_file.yaml

    # To make the input data
    gastrogp config_file.yaml --read_input_only

    # submit it all to the batch farm of cc in2p3>:)
    gastrogp config_file.yaml --interp_only
    
    # to compute the mean function once the gp is run across all visits
    gastrogp config_file.yaml --comp_meanify

    # to gather outputs for plotting purpose
    gastrogp config_file.yaml --write_output_only

    # to get the plot for the paper:
    gastrogp config_file.yaml --comp_plot_paper

    # to run only on a single exposure 
    
    gastrify single_shot.yaml
