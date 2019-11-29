# gastrometry
astrometry residual analysis of HSC

# example run:

make some config file config_file.yaml

    # To make the input data
    gastrogp config_file.yaml --read_input_only

    # submit it all to the batch farm of cc in2p3>:)
    gastrogp config_file.yaml --interp_only

    # drink coffee, probably eat a meal and sleep

    # to run only on a single exposure 
    
    gastrify single_shot.yaml
